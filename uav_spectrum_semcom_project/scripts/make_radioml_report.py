from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ALL_SNR = PROJECT_DIR / "results" / "phase1" / "radioml_classifier_result.json"
MID_HIGH = PROJECT_DIR / "results" / "phase1" / "radioml_classifier_result_minsnr0.json"
OUT = PROJECT_DIR / "results" / "phase1" / "radioml_semantic_validation_report.md"


def main() -> None:
    all_snr = json.loads(ALL_SNR.read_text(encoding="utf-8"))
    mid_high = json.loads(MID_HIGH.read_text(encoding="utf-8"))
    payload = all_snr["payload_bits"]

    lines = [
        "# RadioML Medium Dataset Semantic Validation",
        "",
        "## 1. 数据集",
        "",
        "- 数据集：RadioML 2016.10A",
        "- 大小：压缩包约 212.7 MB，解压后约 225.3 MB",
        "- 样本数：220,000",
        "- 类别数：11",
        "- SNR：-20 dB 到 18 dB，步长 2 dB",
        "- 样本形状：`(2, 128)` I/Q",
        "",
        "## 2. 分类语义提取结果",
        "",
        "| 实验 | 训练样本 | Val Acc | Test Acc | 说明 |",
        "|---|---:|---:|---:|---|",
        f"| 全 SNR | {all_snr['n_train']} | {all_snr['val_accuracy']:.4f} | {all_snr['test_accuracy']:.4f} | 包含 -20 到 18 dB，低 SNR 难度高 |",
        f"| SNR >= 0 dB | {mid_high['n_train']} | {mid_high['val_accuracy']:.4f} | {mid_high['test_accuracy']:.4f} | 中高 SNR 下类别语义更稳定 |",
        "",
        "## 3. Payload 开销",
        "",
        "| Payload | bit/sample | 相比 12+12 bit I/Q 的压缩倍数 |",
        "|---|---:|---:|",
        f"| 原始 I/Q | {payload['raw_iq_12bit_iq_bits_per_sample']} | 1.0x |",
        f"| hard class semantic | {payload['hard_class_bits_per_sample']} | {payload['raw_to_hard_ratio']:.1f}x |",
        f"| soft probability semantic | {payload['soft_prob_8bit_bits_per_sample']} | {payload['raw_to_soft_ratio']:.1f}x |",
        f"| 16d/6bit feature token | {payload['feature_token_16d_6bit_bits_per_sample']} | {payload['raw_to_token_ratio']:.1f}x |",
        "",
        "## 4. 阶段性结论",
        "",
        "1. RadioML 验证了 class-level I/Q 语义在中高 SNR 下可以被轻量 CNN 提取。",
        "2. 低 SNR 会显著降低语义提取准确率，因此后续必须做 SNR sweep 和鲁棒性建模。",
        "3. 类别语义 payload 的 bit 数远低于原始 I/Q：hard class 约 768x，soft probability 约 34.9x，feature token 约 21.2x。",
        "4. 该数据集验证的是调制类别语义，不验证宽带时频框定位；它与 SigMF/RadDet 类数据互补。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

