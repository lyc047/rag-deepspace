#!/usr/bin/env python
"""Generate tables and figures for the C1-v4 development scalability experiment."""

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
METHODS = ("soft_power_fixed4", "best_block_onehot", "best_block_binary_index")
LABELS = ("4-bit soft power", "Best-block one-hot", "Best-block binary index")
COLORS = ("#4e79a7", "#2ca25f", "#f28e2b")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage4/c1_v4_scalability_assets_v1",
    )
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = (
        PROJECT_DIR / "results/stage4/c1_v4_scalability_v1/scalability_result.json"
    )
    protocol_path = PROJECT_DIR / "configs/stage4_c1_v4_scalability_experiment.json"
    result, protocol = load(result_path), load(protocol_path)
    governance = result["governance"]
    if (
        governance["final_measurement_values_loaded"]
        or governance["final_metrics_loaded"]
        or governance["confirmatory"]
    ):
        raise RuntimeError("scalability assets require development-only results")

    summary_rows = []
    inference_rows = []
    for task in result["tasks"]:
        identifier = task["task_id"]
        for method in METHODS:
            row = result["summary"][identifier][method]
            summary_rows.append(
                [
                    identifier,
                    task["n_channels"],
                    task["demand_channels"],
                    task["demand_ratio"],
                    task["candidate_blocks"],
                    method,
                    row["payload_bits"],
                    row["application_bits"],
                    row["mean_clean_regret_db"],
                    row["mean_regret_db"],
                    row["cvar_0_9_regret_db"],
                    row["mean_actual_bits"],
                    row["mean_frame_success_rate"],
                    row["mean_link_latency_ms"],
                ]
            )
        for method in METHODS[1:]:
            value = result["comparisons_vs_soft_power_fixed4"][identifier][method]
            inference_rows.append(
                [
                    identifier,
                    method,
                    value["cluster_count"],
                    value["repetitions"],
                    value["mean_regret_difference"],
                    value["mean_regret_upper_bound"],
                    value["mean_regret_one_sided_p"],
                    value["cvar_difference"],
                    value["cvar_upper_bound"],
                    value["actual_bit_difference"],
                    value["actual_bit_upper_bound"],
                    value["actual_bit_reduction_pct"],
                    value["mean_regret_noninferior"],
                    value["reaches_20pct_strong_target"],
                ]
            )
    write_csv(
        out / "table_scalability_summary.csv",
        [
            "task_id",
            "n_channels",
            "demand_channels",
            "demand_ratio",
            "candidate_blocks",
            "method",
            "payload_bits",
            "application_bits",
            "mean_clean_regret_db",
            "mean_regret_db",
            "cvar_0_9_regret_db",
            "mean_actual_bits",
            "frame_success_rate",
            "mean_link_latency_ms",
        ],
        summary_rows,
    )
    write_csv(
        out / "table_scalability_inference.csv",
        [
            "task_id",
            "method",
            "clusters",
            "bootstrap_repetitions",
            "mean_regret_difference",
            "mean_regret_one_sided_95_upper",
            "mean_regret_one_sided_p",
            "cvar_difference",
            "cvar_one_sided_95_upper",
            "actual_bit_difference",
            "actual_bit_one_sided_95_upper",
            "actual_bit_reduction_pct",
            "mean_regret_noninferior",
            "reaches_20pct_strong_target",
        ],
        inference_rows,
    )
    condition_rows = [
        [
            row["task_id"],
            row["n_channels"],
            row["demand_channels"],
            row["method"],
            row["channel"],
            row["ebn0_db"],
            row["mean_regret_db"],
            row["mean_actual_bits"],
            row["mean_frame_success_rate"],
            row["mean_link_latency_ms"],
        ]
        for row in result["condition_rows"]
    ]
    write_csv(
        out / "table_scalability_link_conditions.csv",
        [
            "task_id",
            "n_channels",
            "demand_channels",
            "method",
            "channel",
            "ebn0_db",
            "mean_regret_db",
            "mean_actual_bits",
            "frame_success_rate",
            "mean_link_latency_ms",
        ],
        condition_rows,
    )

    ratios = protocol["demand_ratios"]
    channel_counts = protocol["channel_counts"]
    fig, axes = plt.subplots(1, len(ratios), figsize=(12.8, 3.8), sharey=True)
    for axis, ratio in zip(axes, ratios):
        tasks = [
            task for task in result["tasks"] if task["demand_ratio"] == ratio
        ]
        for method, label, color in zip(METHODS, LABELS, COLORS):
            values = [
                result["summary"][task["task_id"]][method]["mean_actual_bits"]
                for task in tasks
            ]
            axis.plot(
                channel_counts,
                values,
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.axvline(16, color="black", alpha=0.15, linewidth=0.8)
        axis.set_title(f"D/N = {ratio:.2f}")
        axis.set_xlabel("Number of channels N")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Mean transmitted bits")
    axes[-1].legend(fontsize=8)
    save(fig, out / "figure_scalability_actual_bits.png")

    fig, axes = plt.subplots(1, len(ratios), figsize=(12.8, 3.8), sharey=False)
    for axis, ratio in zip(axes, ratios):
        tasks = [
            task for task in result["tasks"] if task["demand_ratio"] == ratio
        ]
        for method, label, color in zip(METHODS, LABELS, COLORS):
            values = [
                result["summary"][task["task_id"]][method]["mean_regret_db"]
                for task in tasks
            ]
            axis.plot(
                channel_counts,
                values,
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set_title(f"D/N = {ratio:.2f}")
        axis.set_xlabel("Number of channels N")
        axis.set_ylabel("Mean block regret (dB)")
        axis.grid(alpha=0.22)
    axes[-1].legend(fontsize=8)
    save(fig, out / "figure_scalability_regret.png")

    representative = "N64_D32"
    channels = protocol["link"]["channel_models"]
    ebn0_values = protocol["link"]["ebn0_db"]
    fig, axes = plt.subplots(1, len(channels), figsize=(12.8, 3.8), sharey=True)
    for axis, channel in zip(axes, channels):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            rows = sorted(
                (
                    row
                    for row in result["condition_rows"]
                    if row["task_id"] == representative
                    and row["channel"] == channel
                    and row["method"] == method
                ),
                key=lambda row: row["ebn0_db"],
            )
            axis.plot(
                ebn0_values,
                [row["mean_regret_db"] for row in rows],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set_title(channel.upper())
        axis.set_xlabel(r"$E_b/N_0$ (dB)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Mean block regret (dB)")
    axes[-1].legend(fontsize=8)
    save(fig, out / "figure_scalability_link_degradation_n64_d32.png")

    lines = [
        "# C1-v4规模扩展实验表（自动生成）",
        "",
        "> 仅使用已经看过的开发日期扫频；不读取Final测量值或指标，",
        "> 不修改Final算法和结论。紧凑二进制块编号仍是开发方案。",
        "",
        "## 任务结果",
        "",
        "| 任务 | 方法 | 应用层bit | 实际bit | 平均regret/dB | CVaR0.9/dB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in result["tasks"]:
        identifier = task["task_id"]
        for method in METHODS:
            row = result["summary"][identifier][method]
            lines.append(
                f"| {identifier} | {method} | {row['application_bits']} | "
                f"{row['mean_actual_bits']:.3f} | {row['mean_regret_db']:.6f} | "
                f"{row['cvar_0_9_regret_db']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## 相对4-bit软功率的实际bit降幅",
            "",
            "| 任务 | 当前独热指示 | 紧凑二进制编号 | 独热达到20%强目标 | 编号达到20%强目标 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for task in result["tasks"]:
        identifier = task["task_id"]
        onehot = result["comparisons_vs_soft_power_fixed4"][identifier][
            "best_block_onehot"
        ]
        binary = result["comparisons_vs_soft_power_fixed4"][identifier][
            "best_block_binary_index"
        ]
        lines.append(
            f"| {identifier} | {onehot['actual_bit_reduction_pct']:.2f}% | "
            f"{binary['actual_bit_reduction_pct']:.2f}% | "
            f"{onehot['reaches_20pct_strong_target']} | "
            f"{binary['reaches_20pct_strong_target']} |"
        )
    lines.extend(
        [
            "",
            f"- 当前独热指示达到强目标的任务数："
            f"{result['strong_target_task_count']['best_block_onehot']}/12。",
            f"- 紧凑二进制编号达到强目标的任务数："
            f"{result['strong_target_task_count']['best_block_binary_index']}/12。",
            "",
        ]
    )
    (out / "scalability_tables.md").write_text("\n".join(lines), encoding="utf-8")

    sources = [result_path, protocol_path]
    generated = sorted(
        path
        for path in out.iterdir()
        if path.is_file() and path.name != "scalability_assets_manifest.json"
    )
    manifest = {
        "asset_id": "stage4_c1_v4_scalability_assets_v1",
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
        "task_count": len(result["tasks"]),
        "scene_count": result["inputs"]["scene_count"],
        "strong_target_task_count": result["strong_target_task_count"],
        "governance": governance,
        "claim_boundary": result["claim_boundary"],
    }
    (out / "scalability_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "generated": [path.name for path in generated],
                "strong_target_task_count": result["strong_target_task_count"],
                "final_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
