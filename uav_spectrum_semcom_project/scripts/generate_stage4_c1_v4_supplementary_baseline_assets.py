#!/usr/bin/env python
"""Generate thesis-ready assets for the post-Final supplementary baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage4/c1_v4_supplementary_baseline_assets_v1",
    )
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    result_path = (
        PROJECT_DIR
        / "results/stage4/c1_v4_supplementary_baselines_v1"
        / "supplementary_baselines_result.json"
    )
    protocol_path = PROJECT_DIR / "configs/stage4_c1_v4_supplementary_baselines.json"
    result = load(result_path)
    protocol = load(protocol_path)
    governance = result["governance"]
    if governance["final_measurement_values_loaded"] or governance["final_metrics_loaded"]:
        raise RuntimeError("Supplementary assets must not depend on Final measurements or metrics")
    if governance["confirmatory"] or governance["changes_final_claim"]:
        raise RuntimeError("Supplementary development baselines cannot be confirmatory")

    rep = result["representation_baselines"]["summary"]
    rep_order = [
        "hard_mean_energy_1bit",
        "occupancy_fraction_4bit",
        "soft_channel_power_4bit",
        "soft_channel_power_8bit",
        "proposed_best_block_indicator_1bit",
    ]
    rep_labels = [
        "Hard energy\n1-bit",
        "Occupancy\n4-bit",
        "Soft power\n4-bit",
        "Soft power\n8-bit",
        "Proposed block\nindicator",
    ]
    rep_colors = ["#7f7f7f", "#b07aa1", "#4e79a7", "#76b7b2", "#2ca25f"]
    rep_rows = [
        [
            method,
            rep[method]["application_bits"],
            rep[method]["mean_regret_db"],
            rep[method]["cvar_0_9_regret_db"],
            rep[method]["mean_actual_bits"],
            rep[method]["mean_frame_success_rate"],
            rep[method]["mean_link_latency_ms"],
        ]
        for method in rep_order
    ]
    write_csv(
        out / "table_representation_baselines.csv",
        [
            "method",
            "application_bits",
            "mean_regret_db",
            "cvar_0_9_regret_db",
            "mean_actual_bits",
            "frame_success_rate",
            "mean_link_latency_ms",
        ],
        rep_rows,
    )

    rep_comparisons = result["representation_baselines"][
        "comparisons_proposed_minus_baseline"
    ]
    rep_inference_rows = [
        [
            baseline,
            value["cluster_count"],
            value["repetitions"],
            value["mean_regret_difference"],
            value["mean_regret_upper_bound"],
            value["mean_regret_one_sided_p"],
            value["cvar_difference"],
            value["cvar_upper_bound"],
            value["cvar_one_sided_p"],
            value["actual_bit_difference"],
            value["actual_bit_upper_bound"],
        ]
        for baseline, value in rep_comparisons.items()
    ]
    write_csv(
        out / "table_representation_inference.csv",
        [
            "baseline",
            "clusters",
            "bootstrap_repetitions",
            "mean_regret_diff",
            "mean_regret_one_sided_95_upper",
            "mean_regret_one_sided_p",
            "cvar_diff",
            "cvar_one_sided_95_upper",
            "cvar_one_sided_p",
            "actual_bit_diff",
            "actual_bit_one_sided_95_upper",
        ],
        rep_inference_rows,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    for axis, key, ylabel in (
        (axes[0], "mean_regret_db", "Mean block regret (dB)"),
        (axes[1], "cvar_0_9_regret_db", r"CVaR$_{0.9}$ block regret (dB)"),
        (axes[2], "mean_actual_bits", "Mean transmitted bits"),
    ):
        values = [rep[method][key] for method in rep_order]
        bars = axis.bar(np.arange(len(rep_order)), values, color=rep_colors, width=0.72)
        axis.set_xticks(np.arange(len(rep_order)), rep_labels, fontsize=7.6)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        labels = [
            f"{value:.3f}" if key != "mean_actual_bits" else f"{value:.1f}"
            for value in values
        ]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=7.4)
        axis.set_ylim(0, max(values) * 1.22)
    save(fig, out / "figure_representation_baselines.png")

    reliability = result["reliability_baselines"]["summary"]
    rel_order = [
        "no_arq_stateless",
        "standard_arq_stateless",
        "extra_arq_stateless",
        "standard_arq_unlimited_hold",
        "standard_arq_age60_hold",
    ]
    rel_labels = [
        "No ARQ\nstateless",
        "Standard ARQ\nstateless",
        "Extra ARQ\nstateless",
        "Standard ARQ\nunlimited hold",
        "Standard ARQ\nage-60 hold",
    ]
    rel_colors = ["#9c755f", "#7f7f7f", "#f28e2b", "#4e79a7", "#2ca25f"]
    rel_rows = [
        [
            method,
            reliability[method]["mean_regret_db"],
            reliability[method]["cvar_0_9_regret_db"],
            reliability[method]["mean_actual_bits"],
            reliability[method]["mean_frame_success_rate"],
            reliability[method]["mean_link_latency_ms"],
        ]
        for method in rel_order
    ]
    write_csv(
        out / "table_reliability_baselines.csv",
        [
            "method",
            "mean_regret_db",
            "cvar_0_9_regret_db",
            "mean_actual_bits",
            "frame_success_rate",
            "mean_link_latency_ms",
        ],
        rel_rows,
    )

    rel_comparisons = result["reliability_baselines"][
        "comparisons_age60_minus_baseline"
    ]
    rel_inference_rows = [
        [
            baseline,
            value["cluster_count"],
            value["repetitions"],
            value["mean_regret_difference"],
            value["mean_regret_upper_bound"],
            value["mean_regret_one_sided_p"],
            value["cvar_difference"],
            value["cvar_upper_bound"],
            value["cvar_one_sided_p"],
            value["actual_bit_difference"],
            value["actual_bit_upper_bound"],
        ]
        for baseline, value in rel_comparisons.items()
    ]
    write_csv(
        out / "table_reliability_inference.csv",
        [
            "baseline",
            "clusters",
            "bootstrap_repetitions",
            "mean_regret_diff",
            "mean_regret_one_sided_95_upper",
            "mean_regret_one_sided_p",
            "cvar_diff",
            "cvar_one_sided_95_upper",
            "cvar_one_sided_p",
            "actual_bit_diff",
            "actual_bit_one_sided_95_upper",
        ],
        rel_inference_rows,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    for axis, key, ylabel in (
        (axes[0, 0], "mean_regret_db", "Mean block regret (dB)"),
        (axes[0, 1], "cvar_0_9_regret_db", r"CVaR$_{0.9}$ block regret (dB)"),
        (axes[1, 0], "mean_actual_bits", "Mean transmitted bits"),
        (axes[1, 1], "mean_link_latency_ms", "Mean link latency (ms)"),
    ):
        values = [reliability[method][key] for method in rel_order]
        bars = axis.bar(np.arange(len(rel_order)), values, color=rel_colors, width=0.72)
        axis.set_xticks(np.arange(len(rel_order)), rel_labels, fontsize=7.2)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        labels = [
            f"{value:.3f}"
            if key in {"mean_regret_db", "cvar_0_9_regret_db", "mean_link_latency_ms"}
            else f"{value:.1f}"
            for value in values
        ]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=7.2)
        axis.set_ylim(0, max(values) * 1.22)
    save(fig, out / "figure_reliability_baselines.png")

    proposed = rep["proposed_best_block_indicator_1bit"]
    age60 = reliability["standard_arq_age60_hold"]
    standard = reliability["standard_arq_stateless"]
    extra = reliability["extra_arq_stateless"]
    unlimited = reliability["standard_arq_unlimited_hold"]
    hard = rep["hard_mean_energy_1bit"]
    occ = rep["occupancy_fraction_4bit"]
    soft4 = rep["soft_channel_power_4bit"]
    soft8 = rep["soft_channel_power_8bit"]

    lines = [
        "# C1-v4补充基线表（自动生成）",
        "",
        "> 仅使用已经看过的开发验证缓存；属于Final后的解释性补充实验，",
        "> 不读取Final测量值或Final指标，不改变已完成的单次Final结论。",
        "",
        "## 表示基线",
        "",
        "| 方法 | 应用层bit | 平均regret/dB | CVaR0.9/dB | 实际bit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rep_rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.3f} |"
        )
    lines.extend(
        [
            "",
            "最优块指示相对各基线的直观变化：",
            "",
            f"- 相对硬能量判决：平均regret降低{reduction(hard['mean_regret_db'], proposed['mean_regret_db']):.2f}%，"
            f"CVaR降低{reduction(hard['cvar_0_9_regret_db'], proposed['cvar_0_9_regret_db']):.2f}%；"
            "应用层少3 bit，但分包/FEC对齐后实际bit相同。",
            f"- 相对4-bit occupancy：平均regret降低{reduction(occ['mean_regret_db'], proposed['mean_regret_db']):.2f}%，"
            f"CVaR降低{reduction(occ['cvar_0_9_regret_db'], proposed['cvar_0_9_regret_db']):.2f}%，"
            f"实际bit降低{reduction(occ['mean_actual_bits'], proposed['mean_actual_bits']):.2f}%。",
            f"- 相对4-bit软功率：平均regret降低{reduction(soft4['mean_regret_db'], proposed['mean_regret_db']):.2f}%，"
            f"CVaR降低{reduction(soft4['cvar_0_9_regret_db'], proposed['cvar_0_9_regret_db']):.2f}%，"
            f"实际bit降低{reduction(soft4['mean_actual_bits'], proposed['mean_actual_bits']):.2f}%。",
            f"- 相对8-bit软功率：平均regret降低{reduction(soft8['mean_regret_db'], proposed['mean_regret_db']):.2f}%，"
            f"CVaR降低{reduction(soft8['cvar_0_9_regret_db'], proposed['cvar_0_9_regret_db']):.2f}%，"
            f"实际bit降低{reduction(soft8['mean_actual_bits'], proposed['mean_actual_bits']):.2f}%。",
            "",
            "## 可靠性基线",
            "",
            "| 方法 | 平均regret/dB | CVaR0.9/dB | 实际bit | 链路时延/ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rel_rows:
        lines.append(
            f"| {row[0]} | {row[1]:.6f} | {row[2]:.6f} | {row[3]:.3f} | {row[5]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- 60分钟保护相对标准ARQ无状态：平均regret降低{reduction(standard['mean_regret_db'], age60['mean_regret_db']):.2f}%，"
            f"CVaR降低{reduction(standard['cvar_0_9_regret_db'], age60['cvar_0_9_regret_db']):.2f}%，实际bit不变。",
            f"- 60分钟保护相对额外ARQ：平均regret高{100.0 * (age60['mean_regret_db'] - extra['mean_regret_db']) / extra['mean_regret_db']:.2f}%，"
            f"但CVaR低{reduction(extra['cvar_0_9_regret_db'], age60['cvar_0_9_regret_db']):.2f}%，"
            f"实际bit低{reduction(extra['mean_actual_bits'], age60['mean_actual_bits']):.2f}%。",
            f"- 固定站点慢变化开发集上，无限保持的平均regret和CVaR比60分钟保护分别低"
            f"{reduction(age60['mean_regret_db'], unlimited['mean_regret_db']):.2f}%和"
            f"{reduction(age60['cvar_0_9_regret_db'], unlimited['cvar_0_9_regret_db']):.2f}%；"
            "这说明年龄保护付出了保守性代价，而当前数据没有覆盖其要防范的移动、突变或长失联风险。",
            "",
        ]
    )
    (out / "supplementary_baseline_tables.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    sources = [result_path, protocol_path]
    generated = sorted(
        path
        for path in out.iterdir()
        if path.is_file() and path.name != "supplementary_assets_manifest.json"
    )
    manifest = {
        "asset_id": "stage4_c1_v4_supplementary_baseline_assets_v1",
        "source_files": [
            {
                "path": path.relative_to(PROJECT_DIR).as_posix(),
                "sha256": sha256(path),
            }
            for path in sources
        ],
        "generated_files": [
            {
                "path": path.relative_to(PROJECT_DIR).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "representation_scene_count": result["inputs"]["representation_scene_count"],
        "reliability_scene_count": result["inputs"]["reliability_scene_count"],
        "governance": governance,
        "claim_boundary": result["claim_boundary"],
        "protocol_claim_boundary": protocol["claim_boundary"],
    }
    (out / "supplementary_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "generated": [path.name for path in generated],
                "final_measurement_values_loaded": governance[
                    "final_measurement_values_loaded"
                ],
                "confirmatory": governance["confirmatory"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
