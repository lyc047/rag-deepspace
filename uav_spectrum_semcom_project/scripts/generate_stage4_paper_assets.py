#!/usr/bin/env python
"""Generate thesis tables and figures directly from frozen stage-4 JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.paper_assets import assert_development_only, markdown_table


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/paper_assets_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "gate_a": root / "results/stage4/gate_a_training_v1/gate_a_paired_analysis.json",
        "value": root / "results/stage4/value_diagnostics_v1/value_diagnostics_result.json",
        "quality": root / "results/stage4/quality_diagnostics_v1/quality_diagnostics_result.json",
        "c3": root / "results/stage4/reliable_minority_v2/reliable_minority_result.json",
        "end_to_end": root / "results/stage4/end_to_end_sweep_v1/end_to_end_sweep_result.json",
        "complexity": root / "results/stage4/complexity_profile_v1/complexity_result.json",
        "matrix": root / "results/stage4/matrix_coverage_v1/matrix_coverage_result.json",
        "protocol": root / "configs/stage4_protocol.json",
    }
    data = {name: load(path) for name, path in source_paths.items()}
    access = load(root / "configs/stage4_final_access_state.json")
    assert_development_only(data, access)

    # Table 1: Gate A paired effects.
    gate_rows = []
    for row in data["gate_a"]["comparisons"]:
        gate_rows.append([
            row["method"],
            f"{row['regret']['difference']:.6f}", f"[{row['regret']['ci95_low']:.6f}, {row['regret']['ci95_high']:.6f}]",
            f"{row['cvar_0_9']['difference']:.6f}", f"[{row['cvar_0_9']['ci95_low']:.6f}, {row['cvar_0_9']['ci95_high']:.6f}]",
            f"{row['nominal_bits']['difference']:.2f}", f"[{row['nominal_bits']['ci95_low']:.2f}, {row['nominal_bits']['ci95_high']:.2f}]",
        ])
    gate_headers = ["Method vs detection-only", "Regret diff", "Regret 95% CI", "CVaR diff", "CVaR 95% CI", "Bit diff", "Bit 95% CI"]
    write_csv(output / "table_gate_a.csv", gate_headers, gate_rows)

    # Figure 1: Gate A forest plot, each panel retains its natural units.
    methods = [x["method"].replace("detection_plus_", "+").replace("full_joint_loss", "full joint") for x in data["gate_a"]["comparisons"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    metrics = [("regret", "Mean regret difference"), ("cvar_0_9", "CVaR$_{0.9}$ difference"), ("nominal_bits", "Nominal bit difference")]
    for axis, (key, label) in zip(axes, metrics):
        values = np.asarray([row[key]["difference"] for row in data["gate_a"]["comparisons"]])
        low = np.asarray([row[key]["ci95_low"] for row in data["gate_a"]["comparisons"]])
        high = np.asarray([row[key]["ci95_high"] for row in data["gate_a"]["comparisons"]])
        y = np.arange(len(methods))
        axis.errorbar(values, y, xerr=np.vstack((values - low, high - values)), fmt="o", capsize=3, color="#1f77b4")
        axis.axvline(0, color="black", linewidth=.9, linestyle="--")
        axis.set_yticks(y, methods if axis is axes[0] else [])
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=.25)
    axes[0].invert_yaxis()
    save_figure(fig, output / "figure_gate_a_forest.png")

    # Table/Figure 2: selective G2 minus all G2 under actual decoder paths.
    e2e_rows = []
    channel_order = ["awgn", "rayleigh", "rician"]
    for channel in channel_order:
        for ebn0 in (0, 3, 6, 9):
            item = data["end_to_end"]["paired_scene_bootstrap"][f"{channel}_{ebn0}"]["all_G2"]
            e2e_rows.append([channel, ebn0, item["regret"]["difference"], *item["regret"]["ci95"], item["transmitted_bits"]["difference"], *item["transmitted_bits"]["ci95"]])
    e2e_headers = ["channel", "EbN0_dB", "regret_diff", "regret_CI_low", "regret_CI_high", "actual_bit_diff", "bit_CI_low", "bit_CI_high"]
    write_csv(output / "table_end_to_end_selective_vs_all_g2.csv", e2e_headers, e2e_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    colors = {"awgn": "#1f77b4", "rayleigh": "#d62728", "rician": "#2ca02c"}
    for channel in channel_order:
        rows = [x for x in e2e_rows if x[0] == channel]
        x = np.asarray([r[1] for r in rows])
        for axis, indices, label in ((axes[0], (2, 3, 4), channel), (axes[1], (5, 6, 7), channel)):
            y = np.asarray([r[indices[0]] for r in rows], dtype=float)
            lo = np.asarray([r[indices[1]] for r in rows], dtype=float)
            hi = np.asarray([r[indices[2]] for r in rows], dtype=float)
            axis.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), marker="o", capsize=3, label=label, color=colors[channel])
    axes[0].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[0].axhline(data["protocol"]["final_evaluation_protocol"]["primary_claims"]["selective_G2_digital_reporting"]["regret_noninferiority_margin"], color="#9467bd", linestyle=":", linewidth=1.2, label="frozen NI margin")
    axes[0].set_ylabel("Regret difference")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Actual transmitted-bit difference")
    for axis in axes:
        axis.set_xlabel("Reporting $E_b/N_0$ (dB)")
        axis.set_xticks([0, 3, 6, 9])
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    save_figure(fig, output / "figure_end_to_end_selective_g2.png")

    # Figure 3: calibration reliability before/after.
    calibration = data["quality"]["calibration_metrics"]
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    for key, label, color in (("raw_reliability_table", "raw confidence", "#d62728"), ("calibrated_reliability_table", "calibrated", "#1f77b4")):
        table = calibration[key]
        axis.plot([x["mean_prediction"] for x in table], [x["mean_target"] for x in table], marker="o", label=label, color=color)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=.8, label="ideal")
    axis.set_xlim(0, 1); axis.set_ylim(0, 1)
    axis.set_xlabel("Predicted task reliability"); axis.set_ylabel(r"Observed $1-\mathrm{MAE}$")
    axis.grid(alpha=.25); axis.legend(fontsize=8)
    save_figure(fig, output / "figure_quality_reliability.png")

    # Figure 4: value ranking failure and available oracle gap.
    ranking = data["value"]["ranking"]
    effect = data["value"]["scheduler_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].bar(["top-1", "top-2", "top-3"], [ranking["top_1_recall"], ranking["top_2_recall"], ranking["top_3_recall"]], color="#ff7f0e")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Oracle-action recall"); axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(["learned selection", "oracle"], [effect["selected_regret_reduction"], effect["oracle_regret_reduction"]], color=["#1f77b4", "#2ca02c"])
    axes[1].set_ylabel("Mean regret reduction vs all-G1"); axes[1].grid(axis="y", alpha=.25)
    save_figure(fig, output / "figure_value_diagnostic.png")

    # Human-readable tables kept in one generated document.
    complexity = data["complexity"]
    c1, c2 = complexity["c1_variable_rate_head"], complexity["c2_value_network"]
    complexity_table = markdown_table(
        ["Component", "Parameters", "FP32 bytes", "Linear MACs", "Median ms", "p95 ms"],
        [
            ["C1 head / scene", c1["total_parameters"], c1["parameter_bytes_fp32"], c1["linear_MACs_per_scene"], f"{c1['discrete_infer_latency']['median_ms']:.4f}", f"{c1['discrete_infer_latency']['p95_ms']:.4f}"],
            ["C2 / 8 candidates / 1 seed", c2["total_parameters"], c2["parameter_bytes_fp32"], c2["linear_MACs_for_8_candidates_one_seed"], f"{c2['eight_candidate_one_seed_latency']['median_ms']:.4f}", f"{c2['eight_candidate_one_seed_latency']['p95_ms']:.4f}"],
            ["C2 / 8 candidates / 5 seeds", c2["total_parameters"] * 5, c2["parameter_bytes_fp32"] * 5, c2["linear_MACs_for_8_candidates_5_seed_ensemble"], f"{c2['eight_candidate_five_seed_latency']['median_ms']:.4f}", f"{c2['eight_candidate_five_seed_latency']['p95_ms']:.4f}"],
        ],
    )
    value_table = markdown_table(
        ["Diagnostic", "Value"],
        [["Value/bit MAE", f"{data['value']['regression']['value_per_expected_bit_MAE']:.8g}"], ["Pooled Spearman", f"{ranking['pooled_spearman']:.5f}"], ["Top-1 / Top-2 / Top-3", f"{ranking['top_1_recall']:.4f} / {ranking['top_2_recall']:.4f} / {ranking['top_3_recall']:.4f}"], ["Selected regret reduction", f"{effect['selected_regret_reduction']:.6f}"], ["Oracle regret reduction", f"{effect['oracle_regret_reduction']:.6f}"]],
    )
    tables = ["# Stage-4 thesis tables (generated)", "", "> Development evidence only; final holdout access count is 0.", "", "## Gate A paired effects", "", markdown_table(gate_headers, gate_rows), "", "## C2 value diagnostic", "", value_table, "", "## Complexity", "", complexity_table, "", "## Coverage status", "", markdown_table(["Status", "Count"], [[key, value] for key, value in data["matrix"]["counts"].items()]), ""]
    (output / "paper_tables.md").write_text("\n".join(tables), encoding="utf-8")

    generated = sorted(path for path in output.iterdir() if path.is_file() and path.name != "paper_assets_manifest.json")
    manifest = {
        "asset_id": "stage4_paper_assets_v1",
        "source_files": [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path)} for path in source_paths.values()],
        "generated_files": [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in generated],
        "claim_boundary": "All assets are generated from development/calibration/validation aggregates. Final H2/H3 remains untested.",
        "final_access_count": access["access_count"],
        "acceptance": {"completed": True, "final_flags_found": False},
    }
    (output / "paper_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": [path.name for path in generated], "final_access_count": access["access_count"]}, indent=2))


if __name__ == "__main__":
    main()
