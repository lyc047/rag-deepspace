from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.fiesta import discover_fiesta_files, summarize_fiesta_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Fiesta spectrum-measurement semantic payload savings.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "fiesta" / "fiesta_dataset")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--threshold-margin-db", type=float, default=8.0)
    parser.add_argument("--min-bins", type=int, default=2)
    args = parser.parse_args()

    files = discover_fiesta_files(args.root)
    if not files:
        raise SystemExit(f"No Fiesta files found under {args.root}")

    all_rows = []
    for file in files:
        all_rows.extend(
            summarize_fiesta_file(
                file,
                max_rows=args.max_rows_per_file,
                threshold_margin_db=args.threshold_margin_db,
                min_bins=args.min_bins,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "fiesta_payload_summary.csv"
    json_path = args.out_dir / "fiesta_payload_summary.json"
    md_path = args.out_dir / "fiesta_payload_report.md"

    fieldnames = [
        "device",
        "center_hz",
        "bandwidth_hz",
        "timestamp_s",
        "latitude",
        "longitude",
        "n_bins",
        "n_events",
        "raw_float32_bits",
        "raw_int16_bits",
        "semantic_event_bits",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.__dict__)

    n = len(all_rows)
    total_float32 = sum(row.raw_float32_bits for row in all_rows)
    total_int16 = sum(row.raw_int16_bits for row in all_rows)
    total_semantic = sum(row.semantic_event_bits for row in all_rows)
    total_events = sum(row.n_events for row in all_rows)
    nonempty = sum(1 for row in all_rows if row.n_events > 0)
    devices = sorted({row.device for row in all_rows})
    centers_mhz = sorted({round(row.center_hz / 1e6) for row in all_rows})

    summary = {
        "dataset": "Fiesta spectrum crowdsensing dataset",
        "source": "https://www.kaggle.com/datasets/neutrinoliu/fiesta",
        "n_files": len(files),
        "n_frames": n,
        "devices": devices,
        "center_frequencies_mhz": centers_mhz,
        "threshold_margin_db": args.threshold_margin_db,
        "min_bins": args.min_bins,
        "total_events": total_events,
        "mean_events_per_frame": total_events / max(n, 1),
        "nonempty_frame_ratio": nonempty / max(n, 1),
        "mean_raw_float32_bits_per_frame": total_float32 / max(n, 1),
        "mean_raw_int16_bits_per_frame": total_int16 / max(n, 1),
        "mean_semantic_event_bits_per_frame": total_semantic / max(n, 1),
        "float32_to_semantic_ratio": total_float32 / max(total_semantic, 1),
        "int16_to_semantic_ratio": total_int16 / max(total_semantic, 1),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fiesta Spectrum Crowdsensing Payload Report",
        "",
        "## 1. 数据定位",
        "",
        "- 数据集：Fiesta spectrum crowdsensing dataset",
        "- 来源：https://www.kaggle.com/datasets/neutrinoliu/fiesta",
        f"- 文件数：{len(files)}",
        f"- 设备数：{len(devices)}，设备：{', '.join(devices)}",
        f"- 中心频点：{', '.join(str(v) for v in centers_mhz)} MHz",
        "- 单帧格式：latitude、longitude、timestamp + 256 个 PSD bin",
        "",
        "## 2. 当前语义化方法",
        "",
        (
            "本脚本把每个 PSD 快照中高于“该快照中位数 + margin”的连续频率 bin "
            "合并为占用事件。该方法不是最终检测器，而是用于验证现实频谱测量中："
            "如果任务只需要占用频段、峰值功率和位置时间元数据，完整上传 256-bin 频谱并非总是必要。"
        ),
        "",
        "## 3. Payload 统计",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 帧数 | {n} |",
        f"| 事件总数 | {total_events} |",
        f"| 平均事件/帧 | {summary['mean_events_per_frame']:.3f} |",
        f"| 非空帧比例 | {summary['nonempty_frame_ratio']:.3f} |",
        f"| 原始 float32 频谱 bits/帧 | {summary['mean_raw_float32_bits_per_frame']:.1f} |",
        f"| 原始 int16 频谱 bits/帧 | {summary['mean_raw_int16_bits_per_frame']:.1f} |",
        f"| 语义事件 bits/帧 | {summary['mean_semantic_event_bits_per_frame']:.1f} |",
        f"| float32 原始/语义压缩倍数 | {summary['float32_to_semantic_ratio']:.2f}x |",
        f"| int16 原始/语义压缩倍数 | {summary['int16_to_semantic_ratio']:.2f}x |",
        "",
        "## 4. 对主项目的意义",
        "",
        "1. Fiesta 不是完整 I/Q 数据，但它证明了真实频谱众包/移动频谱测量存在大量连续频谱读数上传场景。",
        "2. 当下游任务是频谱占用监测、异常频段告警或低空平台频谱地图更新时，事件级语义比完整频谱读数更贴近任务目标。",
        "3. 该结果可作为论文中“现实数据容量瓶颈”的辅助证据；主线验证仍应以 SigMF/RadDet/真实 SDR I/Q 的时频目标检测为核心。",
        "",
        f"CSV: `{csv_path}`",
        f"JSON: `{json_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
