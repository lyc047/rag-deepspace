from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_DIR / "results" / "phase1" / "phase1_integrated_evidence_report.md"


def load_json(relative_path: str) -> dict:
    return json.loads((PROJECT_DIR / relative_path).read_text(encoding="utf-8"))


def fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def main() -> None:
    sigmf_baseline = load_json("results/phase1/frame_baseline_tuned_val_test.json")
    sigmf_cnn = load_json("results/phase1/frame_cnn_train_val_test.json")
    synthetic = load_json("results/phase1/synthetic_frame_baseline_test.json")
    radioml_high = load_json("results/phase1/radioml_classifier_result_minsnr0.json")
    fiesta = load_json("results/phase1/fiesta_payload_summary.json")
    raddet = load_json("results/phase1/raddet_payload_summary.json")
    raddet_mask = load_json("results/phase1/raddet_occupancy_mask_result.json")
    raddet_snr = load_json("results/phase1/raddet_occupancy_mask_by_snr.json")
    raddet_class_snr = load_json("results/phase1/raddet_occupancy_mask_by_class_snr.json")
    raddet_mask_class = load_json("results/phase1/raddet_mask_class_result.json")
    raddet_two_stage = load_json("results/phase1/raddet_two_stage_mask_class_result.json")
    raddet_grid = load_json("results/phase1/raddet_grid_detector_result.json")
    semantic_link = load_json("results/phase1/semantic_link_simulation.json")

    rb = sigmf_baseline["test"]
    rc = sigmf_cnn["test"]["aggregate"]
    syn = synthetic["aggregate"]

    lines = [
        "# Phase 1 Integrated Evidence Report",
        "",
        "## 1. 当前结论",
        "",
        (
            "当前实验已经可以支持一个谨慎结论：面向低空/低轨平台的频谱感知任务中，"
            "如果下游目标是频谱占用、异常频段告警、目标框或类别摘要，那么传输任务语义"
            "可以显著低于完整 I/Q 或完整频谱读数的开销；但“语义提取是否可靠”仍然是核心风险，"
            "需要继续用更贴近宽带检测的数据集和更系统的 SNR/channel sweep 验证。"
        ),
        "",
        "换句话说：项目目前已经证明了“开销差距很大”，但还没有充分证明“在复杂真实场景下语义提取稳定可靠”。",
        "",
        "## 2. 数据与实验角色划分",
        "",
        "| 数据/实验 | 角色 | 适合证明什么 | 不适合证明什么 |",
        "|---|---|---|---|",
        "| Annotated 5G SigMF | 小规模真实 I/Q 主线验证 | 真实 I/Q 到时频候选框/语义 packet 的端到端流程 | 泛化能力，数据太小 |",
        "| Synthetic controlled IQ | 可控补充验证 | SNR、带宽、占用形态、信道扰动 sweep | 真实世界复杂性 |",
        "| RadioML 2016.10A | 中型 I/Q 辅助验证 | class-level 语义在 I/Q 上可被学习，payload 极小 | 宽带占用定位，不应作为主线结果 |",
        "| Fiesta crowdsensing | 现实频谱读数侧证 | 真实移动频谱测量存在持续上传压力，事件语义可减少上报量 | 完整 I/Q 压缩，因它是 PSD 读数不是 I/Q |",
        "| RadDet 128 | 已接入的主线中型数据 | 宽带时频目标检测、目标框/类别语义 payload | 当前仅完成标签统计，尚未训练检测器 |",
        "| DeepSense | 后续候选 | 真实 SDR 多信道频谱占用检测 | 当前数据入口 403，暂未接入 |",
        "",
        "## 3. 已有定量结果",
        "",
        "### 3.1 真实 SigMF I/Q：时频目标检测",
        "",
        "| 方法 | Precision | Recall | F1 | 语义上报速率 | 12+12 bit I/Q 基准 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Tuned traditional baseline | {rb['precision']:.4f} | {rb['recall']:.4f} | {rb['f1']:.4f} | {fmt_bps(rb['mean_semantic_packet_bps'])} | {fmt_bps(rb['mean_raw_iq_12bit_iq_bps'])} |",
        f"| Tiny STFT-CNN | {rc['precision']:.4f} | {rc['recall']:.4f} | {rc['f1']:.4f} | {fmt_bps(rc['mean_semantic_packet_bps'])} | 184.320 Mbps |",
        "",
        "解释：这里的开销差距已经非常明显，语义 packet 是 kbps 级，而完整 I/Q 是百 Mbps 级。但测试集很小，因此 F1 不能被包装成最终性能结论。",
        "",
        "### 3.2 合成 I/Q：可控检测基线",
        "",
        "| 数据 | Precision | Recall | F1 | 语义上报速率 | 12+12 bit I/Q 基准 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| synthetic controlled IQ | {syn['precision']:.4f} | {syn['recall']:.4f} | {syn['f1']:.4f} | {fmt_bps(syn['mean_semantic_packet_bps'])} | {fmt_bps(syn['mean_raw_iq_12bit_iq_bps'])} |",
        "",
        "解释：合成数据用于可控 sweep，而不是代替真实数据。后续应在它上面系统改变 SNR、占用密度、频偏、多径和 packet loss。",
        "",
        "### 3.3 RadioML：调制类别语义的辅助证据",
        "",
        "| 设置 | Val Acc | Test Acc | 原始 I/Q | hard class | soft probability |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| SNR >= 0 dB | {radioml_high['val_accuracy']:.4f} | {radioml_high['test_accuracy']:.4f} "
            f"| {radioml_high['payload_bits']['raw_iq_12bit_iq_bits_per_sample']} bit/sample "
            f"| {radioml_high['payload_bits']['hard_class_bits_per_sample']} bit/sample "
            f"| {radioml_high['payload_bits']['soft_prob_8bit_bits_per_sample']} bit/sample |"
        ),
        "",
        "解释：0.75 左右的中高 SNR 准确率并不理想，所以 RadioML 只能说明类别语义 payload 很小，不能说明主项目已经有效。",
        "",
        "### 3.4 Fiesta：真实频谱读数的容量瓶颈侧证",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 文件数 | {fiesta['n_files']} |",
        f"| 帧数 | {fiesta['n_frames']} |",
        f"| 设备数 | {len(fiesta['devices'])} |",
        f"| 中心频点数 | {len(fiesta['center_frequencies_mhz'])} |",
        f"| 平均事件/帧 | {fiesta['mean_events_per_frame']:.3f} |",
        f"| 非空帧比例 | {fiesta['nonempty_frame_ratio']:.3f} |",
        f"| float32 频谱读数/语义事件压缩倍数 | {fiesta['float32_to_semantic_ratio']:.2f}x |",
        f"| int16 频谱读数/语义事件压缩倍数 | {fiesta['int16_to_semantic_ratio']:.2f}x |",
        "",
        "解释：Fiesta 不是 I/Q 数据，但它帮助回答老师关心的关键问题：哪些现实数据确实有容量瓶颈？移动频谱众包、低空频谱地图更新和持续频谱监测就是合理场景。",
        "",
        "### 3.5 RadDet 128：主线宽带时频检测数据",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 抽样帧数 | {raddet['n_frames']} |",
        f"| 有标签帧数 | {raddet['n_labeled_frames']} |",
        f"| 平均目标框/帧 | {raddet['mean_boxes_per_frame']:.3f} |",
        f"| PNG 压缩图像 bits/帧 | {raddet['mean_png_bits_per_frame']:.1f} |",
        f"| 8-bit 128×128 谱图 bits/帧 | {raddet['raw_spectrogram_8bit_bits_per_frame']:.1f} |",
        f"| 12+12 bit I/Q bits/帧 | {raddet['mean_raw_iq_12bit_bits_per_frame']:.1f} |",
        f"| hard box semantic bits/帧 | {raddet['mean_semantic_box_bits_per_frame']:.1f} |",
        f"| PNG/语义压缩倍数 | {raddet['png_to_semantic_ratio']:.2f}x |",
        f"| 8-bit 谱图/语义压缩倍数 | {raddet['spectrogram8_to_semantic_ratio']:.2f}x |",
        f"| 12-bit I/Q/语义压缩倍数 | {raddet['iq12_to_semantic_ratio']:.2f}x |",
        "",
        (
        "解释：RadDet 是目前最适合推进主线的中型数据集。它已经清楚显示目标框级语义的开销很低，"
        "但这只是理想标签的上限对比；真正的研究问题是检测器在不同 SNR 和传输扰动下能否稳定恢复这些语义。"
        ),
        "",
        "### 3.6 RadDet 128：初步 occupancy-mask 语义提取",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 训练帧数 | {raddet_mask['train_frames']} |",
        f"| 测试帧数 | {raddet_mask['test_frames']} |",
        f"| Epochs | {raddet_mask['epochs']} |",
        f"| Best validation F1 | {raddet_mask['best_val']['f1']:.4f} |",
        f"| Selected threshold | {raddet_mask['threshold']:.2f} |",
        f"| IoU 阈值 | {raddet_mask['iou_threshold']} |",
        f"| Test Precision | {raddet_mask['test']['precision']:.4f} |",
        f"| Test Recall | {raddet_mask['test']['recall']:.4f} |",
        f"| Test F1 | {raddet_mask['test']['f1']:.4f} |",
        f"| Mean matched IoU | {raddet_mask['test']['mean_iou']:.4f} |",
        f"| 语义 bits/帧 | {raddet_mask['test']['mean_semantic_bits_per_frame']:.1f} |",
        f"| 8-bit 谱图/语义压缩倍数 | {raddet_mask['test']['spectrogram8_to_semantic_ratio']:.2f}x |",
        f"| 12-bit I/Q/语义压缩倍数 | {raddet_mask['test']['iq12_to_semantic_ratio']:.2f}x |",
        "",
        (
            "解释：初步 mask baseline 已经能提取一部分时频占用语义，但 F1 仍低，主要问题是误检多、类别尚未建模。"
            "这说明项目真正的挑战不在 payload 计算，而在低信噪比/复杂时频图上的可靠语义提取。"
        ),
        "",
        "### 3.7 RadDet 128：按 SNR 分组的语义提取可靠性",
        "",
        "| SNR (dB) | Frames | Precision | Recall | F1 | Mean IoU | Semantic bits/frame |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {row['snr_db']:.0f} | {row['n_frames']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | {row['mean_iou']:.4f} | "
                f"{row['mean_semantic_bits_per_frame']:.1f} |"
            )
            for row in raddet_snr["rows"]
        ],
        "",
        (
            "解释：当前 SNR 分组结果并不呈现简单的“高 SNR 更好”单调关系，"
            "说明 RadDet 的检测难度还受到类别、目标形态、图像生成方式和样本分布影响。"
            "因此后续应继续做 class × SNR 交叉分析，而不是只用单一 SNR 曲线下结论。"
        ),
        "",
        "### 3.8 RadDet 128：类别维度的定位召回率",
        "",
        "| Class | Truth | Matched | Recall |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['class_name']} | {row['truth']} | {row['matched']} | {row['recall']:.4f} |"
            for row in raddet_class_snr["class_totals"]
        ],
        "",
        (
            "解释：类别召回率显示，当前 mask baseline 对 ZadoffChu、FMCW、P3、P1 等类别定位较好，"
            "对 Rect、Frank、Px 等类别较弱。结合 SNR 表可以看出，性能波动并非单纯由信噪比决定，"
            "还受到波形形态和类别分布影响。"
        ),
        "",
        "### 3.9 RadDet 128：类别语义初步验证",
        "",
        "| 方法 | Detection F1 | Recall | Precision | Matched class acc | Hard bits/frame | Soft bits/frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| End-to-end mask+class | {raddet_mask_class['test']['f1']:.4f} | "
            f"{raddet_mask_class['test']['recall']:.4f} | {raddet_mask_class['test']['precision']:.4f} | "
            f"{raddet_mask_class['test']['matched_class_accuracy']:.4f} | "
            f"{raddet_mask_class['test']['mean_hard_semantic_bits_per_frame']:.1f} | "
            f"{raddet_mask_class['test']['mean_soft_semantic_bits_per_frame']:.1f} |"
        ),
        (
            f"| Two-stage mask + crop classifier | {raddet_two_stage['test']['f1']:.4f} | "
            f"{raddet_two_stage['test']['recall']:.4f} | {raddet_two_stage['test']['precision']:.4f} | "
            f"{raddet_two_stage['test']['matched_class_accuracy']:.4f} | "
            f"{raddet_two_stage['test']['mean_hard_semantic_bits_per_frame']:.1f} | "
            f"{raddet_two_stage['test']['mean_soft_semantic_bits_per_frame']:.1f} |"
        ),
        "",
        (
            "解释：两阶段方法保留了较好的定位性能，并把 matched class accuracy 从约 0.162 提升到约 0.207，"
            "但类别语义仍明显不足。当前项目可以较有把握地说“位置/占用语义可提取且开销极低”，"
            "但还不能声称“雷达波形类别语义已经可靠”。下一步需要更强的类别特征、区域级分类器或直接采用轻量 YOLO/DETR 类检测头。"
        ),
        "",
        "### 3.10 RadDet 128：YOLO-style grid detector 初步结果",
        "",
        "| 方法 | Detection F1 | Recall | Precision | Matched class acc | Hard bits/frame | Soft bits/frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| TinyGridDetectorCNN | {raddet_grid['test']['f1']:.4f} | "
            f"{raddet_grid['test']['recall']:.4f} | {raddet_grid['test']['precision']:.4f} | "
            f"{raddet_grid['test']['matched_class_accuracy']:.4f} | "
            f"{raddet_grid['test']['mean_hard_semantic_bits_per_frame']:.1f} | "
            f"{raddet_grid['test']['mean_soft_semantic_bits_per_frame']:.1f} |"
        ),
        "",
        (
            "解释：grid detector 的输出形式最接近最终语义包，可以直接产生 box、class 和 confidence。"
            "cell-relative 坐标回归后性能明显改善，但当前 F1 仍低于 occupancy-mask baseline，"
            "说明 one-anchor 小网格检测头还需要更好的正负样本设计、anchor/尺度建模和更长训练。"
        ),
        "",
        "### 3.11 语义通信链路仿真",
        "",
        "当前仿真将检测器输出作为语义源，比较 hard semantic、soft semantic、重复语义包、8-bit 谱图转发和 12-bit I/Q 转发在 packet loss / BER 下的任务性能。大 payload 转发采用保守模型：一帧需要多个 packet，若该帧 payload 未完整到达则地面端不能运行检测。",
        "",
        "| 条件 | Scheme | F1 | Recall | Precision | bits/frame | vs 8-bit 谱图 | vs 12-bit I/Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| PL={row['packet_loss']:.2g}, BER={row['ber']:.0e} | {row['scheme']} | "
                f"{row['f1']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} | "
                f"{row['mean_bits_per_frame']:.1f} | {row['compression_vs_spectrogram8']:.1f}x | "
                f"{row['compression_vs_iq12']:.1f}x |"
            )
            for row in semantic_link["rows"]
            if (
                (abs(row["packet_loss"] - 0.05) < 1e-12 and abs(row["ber"] - 1e-5) < 1e-15)
                or (abs(row["packet_loss"] - 0.1) < 1e-12 and abs(row["ber"] - 1e-4) < 1e-15)
            )
            and row["scheme"] in {"hard", "hard_rep3", "spectrogram8", "iq12"}
        ],
        "",
        (
            "解释：语义包把通信对象从大规模 I/Q/谱图变成少量 box 级信息。"
            "在轻中等丢包和误码下，hard semantic 只出现温和退化；3 次重复语义包可以显著增强链路鲁棒性，"
            "同时 payload 仍远低于完整谱图或 I/Q。该结果使项目从“检测算法”推进到了“语义通信系统仿真”。"
        ),
        "",
        "## 4. 对“高 SNR 0.75 准确率不理想”的处理",
        "",
        "这个判断是对的。当前不应该把 RadioML 作为主线亮点。后续应把研究重点放回宽带时频目标检测：",
        "",
        "1. 用已接入的 RadDet 128 子集验证“时频目标框/占用图”语义，而不是只识别调制方式。",
        "2. 在合成 I/Q 上做 SNR sweep，明确语义提取在低 SNR 下何时失效。",
        "3. 加入 soft semantic payload，例如概率图、置信度框、低维 feature token，而不是只传 hard label。",
        "4. 引入上报信道丢包/误码，比较 hard semantic、soft semantic、压缩谱图和原始 I/Q 的任务性能-开销曲线。",
        "",
        "## 5. 下一步执行建议",
        "",
        "优先级最高的是在已接入的 RadDet 128 小包上训练轻量检测/分割模型，因为它的数据规模适中、任务形态接近目标检测，且比 RadioML 更能支撑论文主线。",
        "",
        "- 先做 RadDet 的轻量检测 baseline：输入 128×128 时频图，输出目标框和类别。",
        "- 再做 RadDet 的 SNR 分组评估：按 metadata 中的 SNR 分别报告 mAP/F1 和语义 payload。",
        "- 同步在 synthetic controlled IQ 上完成 SNR/channel sweep，用可控数据解释低 SNR 失败边界。",
        "- 不建议继续花主要时间堆 RadioML 准确率；它可以保留为辅助章节，但不是项目核心贡献。",
    ]

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
