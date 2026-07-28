from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_DIR / "results" / "phase1" / "real_sigmf_energy_baseline.json"
CNN_PATH = PROJECT_DIR / "results" / "phase1" / "tiny_cnn_full_frame_result.json"
FRAME_BASELINE_PATH = PROJECT_DIR / "results" / "phase1" / "frame_baseline_tuned_val_test.json"
FRAME_CNN_PATH = PROJECT_DIR / "results" / "phase1" / "frame_cnn_train_val_test.json"
REPORT_PATH = PROJECT_DIR / "results" / "phase1" / "phase1_method_comparison.md"


def _fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def _best_by_f1(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["f1"])


def main() -> None:
    frame_baseline = json.loads(FRAME_BASELINE_PATH.read_text(encoding="utf-8"))
    frame_cnn = json.loads(FRAME_CNN_PATH.read_text(encoding="utf-8"))
    baseline_test = frame_baseline["test"]
    cnn_test = frame_cnn["test"]["aggregate"]

    rows = [
        {
            "method": "Tuned traditional STFT baseline",
            "setting": (
                f"{frame_baseline['best_params']['enhancement']} / "
                f"sigma={frame_baseline['best_params']['sigma']} / "
                f"min_cells={frame_baseline['best_params']['min_cells']}"
            ),
            "precision": baseline_test["precision"],
            "recall": baseline_test["recall"],
            "f1": baseline_test["f1"],
            "iou": None,
            "payload_bps": baseline_test["mean_semantic_packet_bps"],
        },
        {
            "method": "Frame-level Tiny STFT-CNN",
            "setting": f"device={frame_cnn['device']} / epochs={frame_cnn['args']['epochs']} / patches={frame_cnn['args']['patches_per_frame']}",
            "precision": cnn_test["precision"],
            "recall": cnn_test["recall"],
            "f1": cnn_test["f1"],
            "iou": None,
            "payload_bps": cnn_test["mean_semantic_packet_bps"],
        },
    ]

    lines = [
        "# Phase 1 Method Comparison",
        "",
        "## 1. 当前方法对比",
        "",
        "| 方法 | 设置 | Test Precision | Test Recall | Test F1 | 语义 payload |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | `{row['setting']}` | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {_fmt_bps(row['payload_bps'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. 传输瓶颈基准",
            "",
            "| 数据表示 | 等效速率 |",
            "|---|---:|",
            "| 原始 32+32 bit I/Q | 491.520 Mbps |",
            "| 重量化 12+12 bit I/Q | 184.320 Mbps |",
            "| 8-bit STFT | 约 245.535 Mbps，取决于 STFT 参数 |",
            "",
            "## 3. 阶段性判断",
            "",
            "1. frame-level CNN 在 GPU 环境下已经超过 tuned traditional baseline，但当前数据量很小，不能作为最终泛化结论。",
            "2. 传统 baseline 仍然很重要，它提供了可解释、低成本、强对照的下界。",
            "3. 两类语义候选框 payload 都是 kbps 级，显著低于原始 I/Q 和 STFT 上传。",
            "4. 下一阶段需要增加更多真实/半真实数据，并比较硬判决、软概率、候选框和 feature token 的 payload。",
            "",
            "## 4. 下一步工程任务",
            "",
            "- 将一个长 SigMF 录制切成多个 frame-level 样本，建立 train/val/test manifest。",
            "- 对每个 frame 生成 STFT、mask、标注和 payload 统计。",
            "- 训练 CNN 时按 frame 划分，而不是在同一整帧内随机采 patch 后又整帧测试。",
            "- 同时保留传统 baseline，作为后续语义 token 和多 UAV 融合实验的强对照。",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
