from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_DIR / "results" / "phase1" / "frame_baseline_tuned_val_test.json"
CNN_PATH = PROJECT_DIR / "results" / "phase1" / "frame_cnn_train_val_test.json"
REPORT_PATH = PROJECT_DIR / "results" / "phase1" / "frame_level_phase1_report.md"


def _fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def main() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    cnn = json.loads(CNN_PATH.read_text(encoding="utf-8"))
    b_val = baseline["best_val"]
    b_test = baseline["test"]
    c_val = cnn["best_val"]["aggregate"]
    c_test = cnn["test"]["aggregate"]

    lines = [
        "# Frame-Level Phase 1 Report",
        "",
        "## 1. 实验设置",
        "",
        "- 数据：短 5G NR SigMF 真实 I/Q 录制",
        "- 切分方式：按 SigMF 中的 `Frame N` 标注切分为 frame-level 样本",
        "- 划分：train 8 frames，val 3 frames，test 3 frames",
        "- 目的：避免在同一整帧上随机采 patch 后又在同一整帧测试，建立更规范的第一阶段基准",
        "",
        "## 2. 方法对比",
        "",
        "| 方法 | 参数选择 | Val F1 | Test Precision | Test Recall | Test F1 | Test 平均语义速率 |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            f"| Tuned traditional baseline | `{baseline['best_params']['enhancement']}`, "
            f"sigma={baseline['best_params']['sigma']}, min_cells={baseline['best_params']['min_cells']} "
            f"| {b_val['f1']:.4f} | {b_test['precision']:.4f} | {b_test['recall']:.4f} | {b_test['f1']:.4f} "
            f"| {_fmt_bps(b_test['mean_semantic_packet_bps'])} |"
        ),
        (
            f"| Tiny STFT-CNN | frame-level smoke training, device={cnn['device']} "
            f"| {c_val['f1']:.4f} | {c_test['precision']:.4f} | {c_test['recall']:.4f} | {c_test['f1']:.4f} "
            f"| {_fmt_bps(c_test['mean_semantic_packet_bps'])} |"
        ),
        "",
        "## 3. 关键结论",
        "",
        "1. 传统 `freq_median + connected components` 在当前小数据上仍是最可靠 baseline。",
        "2. Tiny STFT-CNN 已完成 train/val/test 工程闭环，但小配置下尚未超过传统 baseline。",
        "3. 这说明学习式方法后续需要更多训练样本、更合理的标签设计和 GPU 训练，而不是简单增加网络复杂度。",
        "4. 当前 test 上传统语义候选框速率仍是 kbps 级，而 12+12 bit I/Q 是约 184.32 Mbps，容量瓶颈和语义压缩价值仍然成立。",
        "",
        "## 4. 下一步建议",
        "",
        "- 在 VSCode 的 GPU PyTorch 环境中重新运行 `train_frame_cnn.py --device cuda`，把 epoch 和 patches-per-frame 加大。",
        "- 增加更多小规模真实/半真实 I/Q 数据，不能只依赖一条 5G SigMF 录制。",
        "- 训练前先固定传统 baseline，后续学习式方法必须超过 tuned baseline 才作为主线结果。",
        "- 下一阶段加入 payload 类型对比：硬判决、软概率、候选框语义包、局部 feature token。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

