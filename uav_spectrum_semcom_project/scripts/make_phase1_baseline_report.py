from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_DIR / "results" / "phase1" / "real_sigmf_energy_baseline.json"
REPORT_PATH = PROJECT_DIR / "results" / "phase1" / "phase1_baseline_report.md"


def _fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def main() -> None:
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    best = max(rows, key=lambda row: row["f1"])
    truth_counts = best.get("truth_label_counts", {})
    matched_counts = best.get("matched_truth_label_counts", {})

    lines = [
        "# Phase 1 Baseline Report: Real SigMF I/Q",
        "",
        "## 1. 数据概况",
        "",
        f"- 数据源：短 5G NR SigMF 真实 I/Q 录制",
        f"- 样本数：{data['n_samples']:,}",
        f"- 时长：{data['duration_s']:.6f} s",
        f"- 采样率：{data['sample_rate_hz'] / 1e6:.3f} MHz",
        f"- 中心频率：{data['center_freq_hz'] / 1e6:.3f} MHz",
        f"- 参与评估的标注数：{data['truth_count_eval']}",
        "",
        "## 2. 最佳传统 baseline",
        "",
        f"- 检测增强方式：`{best['enhancement']}`",
        f"- 阈值 sigma：{best['threshold_sigma']}",
        f"- 最小连通域单元数：{best['min_cells']}",
        f"- 预测框数：{best['n_pred']}",
        f"- Precision：{best['precision']:.4f}",
        f"- Recall：{best['recall']:.4f}",
        f"- F1：{best['f1']:.4f}",
        f"- 平均匹配 IoU：{best['mean_matched_iou']:.4f}",
        f"- 最佳 IoU：{best['best_iou']:.4f}",
        "",
        "## 3. 传输开销对比",
        "",
        "| 方案 | 等效速率 | 说明 |",
        "|---|---:|---|",
        f"| 原始 32+32 bit I/Q | {_fmt_bps(best['raw_iq_32bit_iq_bps'])} | SigMF 原始量化格式估计 |",
        f"| 重量化 12+12 bit I/Q | {_fmt_bps(best['raw_iq_12bit_iq_bps'])} | 常见 SDR 量化位宽估计 |",
        f"| 8-bit STFT | {_fmt_bps(best['stft_8bit_bps'])} | 直接上传时频图仍然很贵 |",
        f"| 语义候选框包 | {_fmt_bps(best['semantic_packet_bps'])} | 当前 baseline 的候选框 payload |",
        "",
        "## 4. 类别命中情况",
        "",
        "| 标签 | 标注数 | 命中数 | 召回率 |",
        "|---|---:|---:|---:|",
    ]

    for label, count in sorted(truth_counts.items()):
        matched = matched_counts.get(label, 0)
        recall = matched / count if count else 0.0
        lines.append(f"| {label} | {count} | {matched} | {recall:.4f} |")

    lines.extend(
        [
            "",
            "## 5. 当前结论",
            "",
            "1. 真实 I/Q 的传输瓶颈非常明显：即使是 7.68 MHz 的短录制，完整 I/Q 的等效速率也达到百 Mbps 量级。",
            "2. 简单语义候选框 payload 只有 kbps 量级，说明任务语义压缩具有很大的通信节省空间。",
            "3. 频率方向背景归一化 `freq_median` 显著提高了召回率，说明弱小同步/控制结构不能只靠原始能量阈值检测。",
            "4. 当前传统 baseline 仍存在频率边界偏宽和漏检 CSI-RS 的问题，后续需要引入更细粒度的时频特征或轻量学习式检测器。",
            "",
            "## 6. 下一步",
            "",
            "- 构建轻量 STFT-CNN 检测器，先做二分类占用/候选区域热力图，不直接上复杂 Transformer。",
            "- 保持相同 bit 账本，与当前传统 baseline 公平比较。",
            "- 增加硬判决、软信息、语义候选框三种 payload 方案的统一比较。",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

