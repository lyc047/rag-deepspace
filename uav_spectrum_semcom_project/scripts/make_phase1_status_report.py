from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_DIR / "results" / "phase1" / "phase1_status_report.md"


def _load(path: str):
    return json.loads((PROJECT_DIR / path).read_text(encoding="utf-8"))


def _fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def main() -> None:
    real_baseline = _load("results/phase1/frame_baseline_tuned_val_test.json")
    real_cnn = _load("results/phase1/frame_cnn_train_val_test.json")
    synthetic = _load("results/phase1/synthetic_frame_baseline_test.json")
    radioml_all = _load("results/phase1/radioml_classifier_result.json")
    radioml_midsnr = _load("results/phase1/radioml_classifier_result_minsnr0.json")

    rb = real_baseline["test"]
    rc = real_cnn["test"]["aggregate"]
    syn = synthetic["aggregate"]

    lines = [
        "# Phase 1 Status Report",
        "",
        "## 1. 当前已完成内容",
        "",
        "- 已建立独立课题目录，与原深空遥测代码分离。",
        "- 已接入一个小规模真实 5G NR SigMF I/Q 数据，并完成 frame-level train/val/test 划分。",
        "- 已建立可控半真实合成 I/Q frame 数据，用于补充训练和压力测试。",
        "- 已下载并接入 RadioML 2016.10A 中型 I/Q 调制识别数据集，用于 class-level 语义验证。",
        "- 已实现传统 STFT 连通域 baseline、轻量 STFT-CNN、bit 账本、payload 方案对比。",
        "- 已确认本机 GPU 环境可用：`D:\\anaconda\\envs\\pytorch\\python.exe`，PyTorch CUDA，NVIDIA MX450。",
        "",
        "## 2. 真实 SigMF frame-level 结果",
        "",
        "| 方法 | Precision | Recall | F1 | 平均语义速率 | I/Q 基准速率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Tuned traditional baseline | {rb['precision']:.4f} | {rb['recall']:.4f} | {rb['f1']:.4f} | {_fmt_bps(rb['mean_semantic_packet_bps'])} | {_fmt_bps(rb['mean_raw_iq_12bit_iq_bps'])} |",
        f"| GPU Tiny STFT-CNN | {rc['precision']:.4f} | {rc['recall']:.4f} | {rc['f1']:.4f} | {_fmt_bps(rc['mean_semantic_packet_bps'])} | 184.320 Mbps |",
        "",
        "## 3. 半真实合成数据 test 结果",
        "",
        "| 数据 | Precision | Recall | F1 | 平均语义速率 | I/Q 基准速率 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| synthetic controlled IQ | {syn['precision']:.4f} | {syn['recall']:.4f} | {syn['f1']:.4f} | {_fmt_bps(syn['mean_semantic_packet_bps'])} | {_fmt_bps(syn['mean_raw_iq_12bit_iq_bps'])} |",
        "",
        "## 4. RadioML 中型数据集语义分类验证",
        "",
        "| 实验 | Val Acc | Test Acc | 语义类型 | 开销结论 |",
        "|---|---:|---:|---|---|",
        f"| 全 SNR (-20 到 18 dB) | {radioml_all['val_accuracy']:.4f} | {radioml_all['test_accuracy']:.4f} | modulation class | hard class 4 bit/sample，soft probability 88 bit/sample |",
        f"| SNR >= 0 dB | {radioml_midsnr['val_accuracy']:.4f} | {radioml_midsnr['test_accuracy']:.4f} | modulation class | 原始 I/Q 为 3072 bit/sample，soft probability 约 34.9x 压缩 |",
        "",
        "## 5. 当前判断",
        "",
        "1. I/Q 传输瓶颈已经在真实数据和合成数据上都被量化出来：完整 I/Q 是 Mbps 到百 Mbps 级，语义候选框是 kbps 级。",
        "2. RadioML 中型数据集说明：类别语义在中高 SNR 下可被轻量 CNN 提取，但低 SNR 会显著降低准确率。",
        "3. GPU 训练后的轻量 CNN 在当前小规模真实 test split 上超过传统 baseline，但数据量太小，不能作为最终泛化结论。",
        "4. 传统 baseline 仍应保留为强对照，因为它可解释、训练成本低，并能检验学习式方法是否真的有必要。",
        "5. 合成数据入口已经可用，下一步可以用于扩大训练集、做 SNR sweep 和信道扰动实验。",
        "",
        "## 6. 下一步计划",
        "",
        "- 增加更多真实或公开小规模 I/Q 数据，优先 SigMF/RadioML/RadDet 小子集。",
        "- 在合成数据上做 SNR sweep，观察 semantic payload 在低 SNR 下的稳定性。",
        "- 引入上报信道扰动：bit erasure、packet loss、AWGN soft payload。",
        "- 开始实现多 UAV 观测仿真：同一发射源，多 UAV 不同 SNR/遮挡，地面融合 soft box 或 feature token。",
    ]

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
