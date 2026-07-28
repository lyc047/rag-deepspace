#!/usr/bin/env python
"""Generate thesis-ready C1-v4 Final figures and tables from frozen JSON."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_paper_assets_v1")
    args = parser.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    result_path = PROJECT_DIR / "results/stage4/c1_v4_final_v1/c1_v4_final_result.json"
    protocol_path = PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json"
    access_path = PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json"
    inventory_path = PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json"
    result, protocol, access, inventory = map(load, (result_path, protocol_path, access_path, inventory_path))
    if access.get("access_count") != 1 or not result.get("overall_final_success"):
        raise RuntimeError("Final assets require one consumed access and a completed Final result")

    order = ["occupancy_stateless", "indicator_stateless", "indicator_guarded_last_success"]
    labels = ["Occupancy\nstateless", "Block indicator\nstateless", "Block indicator\nage-guarded"]
    colors = ["#8c8c8c", "#2f6f9f", "#2ca25f"]
    summary_rows = []
    for method in order:
        row = result["summary"][method]
        summary_rows.append([method, row["application_bits"], row["mean_regret_db"], row["cvar_0_9_regret_db"], row["mean_actual_bits"], row["mean_frame_success_rate"]])
    write_csv(out / "table_c1_v4_final_summary.csv", ["method", "application_bits", "mean_regret_db", "cvar_0_9_regret_db", "mean_actual_bits", "frame_success_rate"], summary_rows)

    inference_rows = []
    for hypothesis, value in result["hypotheses"].items():
        x = value["inference"]
        inference_rows.append([hypothesis, x["cluster_count"], x["repetitions"], x["mean_regret_difference"], x["mean_regret_upper_bound"], x["mean_regret_one_sided_p"], x["cvar_difference"], x["cvar_upper_bound"], x["cvar_one_sided_p"], x["actual_bit_difference"], x["actual_bit_upper_bound"]])
    write_csv(out / "table_c1_v4_final_inference.csv", ["hypothesis", "clusters", "bootstrap_repetitions", "mean_regret_diff", "mean_regret_one_sided_95_upper", "mean_regret_one_sided_p", "cvar_diff", "cvar_one_sided_95_upper", "cvar_one_sided_p", "actual_bit_diff", "actual_bit_one_sided_95_upper"], inference_rows)

    # Figure 1: task performance and communication cost.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for axis, key, ylabel in (
        (axes[0], "mean_regret_db", "Mean block regret (dB)"),
        (axes[1], "cvar_0_9_regret_db", r"CVaR$_{0.9}$ block regret (dB)"),
        (axes[2], "mean_actual_bits", "Mean transmitted bits"),
    ):
        values = [result["summary"][method][key] for method in order]
        bars = axis.bar(np.arange(3), values, color=colors, width=.68)
        axis.set_xticks(np.arange(3), labels, fontsize=8)
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.22)
        axis.bar_label(bars, labels=[f"{value:.3f}" if key != "mean_actual_bits" else f"{value:.1f}" for value in values], padding=3, fontsize=8)
        axis.set_ylim(0, max(values) * 1.20)
    save(fig, out / "figure_c1_v4_final_overall.png")

    # Figure 2: site consistency.
    sites = sorted(result["site_summary_mean_regret_db"])
    x = np.arange(len(sites)); width = .24
    fig, axis = plt.subplots(figsize=(6.6, 4.0))
    for index, (method, label, color) in enumerate(zip(order, labels, colors)):
        values = [result["site_summary_mean_regret_db"][site][method] for site in sites]
        axis.bar(x + (index - 1) * width, values, width, label=label.replace("\n", " "), color=color)
    axis.set_xticks(x, sites); axis.set_ylabel("Mean block regret (dB)")
    axis.grid(axis="y", alpha=.22); axis.legend(fontsize=8)
    save(fig, out / "figure_c1_v4_final_sites.png")

    # Figure 3: paired effects and frozen one-sided upper bounds.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    hypothesis_labels = ["H1: block semantics", "H2: age-guarded fallback"]
    hypotheses = [result["hypotheses"]["H1_task_sufficient_representation"]["inference"], result["hypotheses"]["H2_age_guarded_zero_bit_fallback"]["inference"]]
    for axis, point_key, upper_key, xlabel in (
        (axes[0], "mean_regret_difference", "mean_regret_upper_bound", "Mean regret difference (dB)"),
        (axes[1], "cvar_difference", "cvar_upper_bound", r"CVaR$_{0.9}$ difference (dB)"),
    ):
        points = np.asarray([item[point_key] for item in hypotheses])
        uppers = np.asarray([item[upper_key] for item in hypotheses])
        lower_error = np.zeros_like(points)
        upper_error = uppers - points
        y = np.arange(2)
        axis.errorbar(points, y, xerr=np.vstack((lower_error, upper_error)), fmt="o", capsize=4, color="#2f6f9f")
        axis.axvline(0, color="black", linestyle="--", linewidth=.9)
        axis.set_yticks(y, hypothesis_labels if axis is axes[0] else [])
        axis.set_xlabel(xlabel); axis.grid(axis="x", alpha=.22); axis.invert_yaxis()
    save(fig, out / "figure_c1_v4_final_effects.png")

    occ = result["summary"]["occupancy_stateless"]
    ind = result["summary"]["indicator_stateless"]
    full = result["summary"]["indicator_guarded_last_success"]
    pct = lambda before, after: 100.0 * (before - after) / before
    lines = [
        "# C1-v4 Final论文表格（自动生成）", "",
        "> 数据来自已冻结的单次Final结果；不得据此重新调参或重跑Final。", "",
        "## 总体结果", "",
        "| 方法 | 应用层bit | 平均regret/dB | CVaR0.9/dB | 实际发送bit | 帧成功率 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.3f} | {row[5]:.6f} |")
    lines.extend(["", "## 关键相对改善", "", f"- H1相对occupancy：平均regret降低{pct(occ['mean_regret_db'], ind['mean_regret_db']):.2f}%，CVaR降低{pct(occ['cvar_0_9_regret_db'], ind['cvar_0_9_regret_db']):.2f}%，实际bit降低{pct(occ['mean_actual_bits'], ind['mean_actual_bits']):.2f}%。", f"- H2相对无状态块指示：平均regret继续降低{pct(ind['mean_regret_db'], full['mean_regret_db']):.2f}%，CVaR继续降低{pct(ind['cvar_0_9_regret_db'], full['cvar_0_9_regret_db']):.2f}%，bit差为0。", f"- 完整V4相对occupancy：平均regret降低{pct(occ['mean_regret_db'], full['mean_regret_db']):.2f}%，CVaR降低{pct(occ['cvar_0_9_regret_db'], full['cvar_0_9_regret_db']):.2f}%。", ""])
    (out / "c1_v4_final_paper_tables.md").write_text("\n".join(lines), encoding="utf-8")

    sources = [result_path, protocol_path, access_path, inventory_path]
    generated = sorted(path for path in out.iterdir() if path.is_file() and path.name != "paper_assets_manifest.json")
    manifest = {
        "asset_id": "stage4_c1_v4_final_paper_assets_v1",
        "source_files": [{"path": path.relative_to(PROJECT_DIR).as_posix(), "sha256": sha256(path)} for path in sources],
        "generated_files": [{"path": path.relative_to(PROJECT_DIR).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in generated],
        "final_access_count": access["access_count"], "overall_final_success": result["overall_final_success"],
        "scene_count": inventory["scene_count"], "cluster_count": inventory["cluster_count"],
        "claim_boundary": protocol["claim_boundary"],
    }
    (out / "paper_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "generated": [path.name for path in generated], "final_access_count": access["access_count"]}, indent=2))


if __name__ == "__main__":
    main()
