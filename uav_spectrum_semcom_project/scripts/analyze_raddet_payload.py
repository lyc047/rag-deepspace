from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.raddet import RADDET_CLASS_NAMES, iter_raddet_frames, read_raddet_metadata


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RadDet 128 payload and annotation statistics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--read-all-metadata", action="store_true")
    parser.add_argument("--default-sequence-length", type=int, default=1_000_000)
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=2000,
        help="Default keeps the quick analysis responsive; pass 0 for all frames.",
    )
    args = parser.parse_args()

    frames = []
    for split in ("train", "val", "test"):
        frames.extend(
            iter_raddet_frames(
                args.root,
                split,
                read_metadata=args.read_all_metadata,
                default_sequence_length=args.default_sequence_length,
                max_frames=None if args.max_frames_per_split == 0 else args.max_frames_per_split,
            )
        )
    if not frames:
        raise SystemExit(f"No RadDet frames found under {args.root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "raddet_payload_summary.csv"
    json_path = args.out_dir / "raddet_payload_summary.json"
    md_path = args.out_dir / "raddet_payload_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "stem",
                "n_boxes",
                "classes",
                "snr_db",
                "image_compressed_bits",
                "raw_spectrogram_8bit_bits",
                "raw_iq_12bit_bits",
                "semantic_box_bits",
            ],
        )
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "split": frame.split,
                    "stem": frame.stem,
                    "n_boxes": len(frame.boxes),
                    "classes": " ".join(box.class_name for box in frame.boxes),
                    "snr_db": frame.snr_db,
                    "image_compressed_bits": frame.image_compressed_bits,
                    "raw_spectrogram_8bit_bits": frame.raw_spectrogram_8bit_bits,
                    "raw_iq_12bit_bits": frame.raw_iq_12bit_bits,
                    "semantic_box_bits": frame.semantic_box_bits,
                }
            )

    split_counts = Counter(frame.split for frame in frames)
    class_counts = Counter(box.class_name for frame in frames for box in frame.boxes)
    n_boxes = [len(frame.boxes) for frame in frames]
    raw_iq_bits = [frame.raw_iq_12bit_bits for frame in frames if frame.raw_iq_12bit_bits]
    total_semantic = sum(frame.semantic_box_bits for frame in frames)
    total_png = sum(frame.image_compressed_bits for frame in frames)
    total_spec = sum(frame.raw_spectrogram_8bit_bits for frame in frames)
    total_iq = sum(raw_iq_bits)

    sampled_snr_values = []
    if not args.read_all_metadata:
        for split in ("train", "val", "test"):
            for path in sorted((args.root / "metadata" / split).glob("*.json"))[:200]:
                for item in read_raddet_metadata(path):
                    if "SNR" in item:
                        sampled_snr_values.append(float(item["SNR"]))

    summary = {
        "dataset": "RadDet40k128HW001Tv2",
        "source": "https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025",
        "license": "CC-BY-NC-SA-4.0",
        "n_frames": len(frames),
        "max_frames_per_split": None if args.max_frames_per_split == 0 else args.max_frames_per_split,
        "split_counts": dict(split_counts),
        "n_labeled_frames": sum(1 for value in n_boxes if value > 0),
        "mean_boxes_per_frame": mean([float(v) for v in n_boxes]),
        "class_counts": dict(class_counts),
        "mean_png_bits_per_frame": mean([float(frame.image_compressed_bits) for frame in frames]),
        "raw_spectrogram_8bit_bits_per_frame": frames[0].raw_spectrogram_8bit_bits,
        "mean_raw_iq_12bit_bits_per_frame": mean([float(v) for v in raw_iq_bits]),
        "mean_semantic_box_bits_per_frame": mean([float(frame.semantic_box_bits) for frame in frames]),
        "png_to_semantic_ratio": total_png / max(total_semantic, 1),
        "spectrogram8_to_semantic_ratio": total_spec / max(total_semantic, 1),
        "iq12_to_semantic_ratio": total_iq / max(total_semantic, 1),
        "snr_values": sorted({frame.snr_db for frame in frames if frame.snr_db is not None})
        or sorted(set(sampled_snr_values)),
        "snr_metadata_mode": "all" if args.read_all_metadata else "sampled_first_200_per_split",
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    class_rows = "\n".join(
        f"| {name} | {class_counts.get(name, 0)} |" for _, name in sorted(RADDET_CLASS_NAMES.items())
    )
    split_rows = "\n".join(f"| {split} | {split_counts.get(split, 0)} |" for split in ("train", "val", "test"))

    lines = [
        "# RadDet 128 Payload and Annotation Report",
        "",
        "## 1. 数据定位",
        "",
        "- 数据集：RadDet40k128HW001Tv2",
        "- 来源：https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025",
        "- 许可证：CC-BY-NC-SA-4.0",
        "- 数据形态：128×128 时频图 PNG、YOLO 目标框标签、每帧 metadata",
        "- metadata 显示典型原始序列长度为 1,000,000 complex I/Q samples，采样率为 500 MHz。",
        "",
        "## 2. Split 规模",
        "",
        "| split | frames |",
        "|---|---:|",
        split_rows,
        "",
        "## 3. 类别分布",
        "",
        "| class | boxes |",
        "|---|---:|",
        class_rows,
        "",
        "## 4. Payload 对比",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 帧数 | {summary['n_frames']} |",
        f"| 有标签帧数 | {summary['n_labeled_frames']} |",
        f"| 平均目标框/帧 | {summary['mean_boxes_per_frame']:.3f} |",
        f"| PNG 压缩图像 bits/帧 | {summary['mean_png_bits_per_frame']:.1f} |",
        f"| 8-bit 128×128 谱图 bits/帧 | {summary['raw_spectrogram_8bit_bits_per_frame']:.1f} |",
        f"| 12+12 bit I/Q bits/帧 | {summary['mean_raw_iq_12bit_bits_per_frame']:.1f} |",
        f"| hard box semantic bits/帧 | {summary['mean_semantic_box_bits_per_frame']:.1f} |",
        f"| PNG/语义压缩倍数 | {summary['png_to_semantic_ratio']:.2f}x |",
        f"| 8-bit 谱图/语义压缩倍数 | {summary['spectrogram8_to_semantic_ratio']:.2f}x |",
        f"| 12-bit I/Q/语义压缩倍数 | {summary['iq12_to_semantic_ratio']:.2f}x |",
        "",
        "## 5. 对主项目的意义",
        "",
        "1. RadDet 比 RadioML 更贴近本项目主线，因为任务是时频目标检测，而不是单窗口调制分类。",
        "2. 该数据集可用于训练/验证从宽带观测中提取目标框、类别、置信度等任务语义。",
        "3. 当前 payload 统计只证明“标签级语义开销极低”；下一步必须训练检测器并评估不同 SNR 下的 detection F1。",
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
