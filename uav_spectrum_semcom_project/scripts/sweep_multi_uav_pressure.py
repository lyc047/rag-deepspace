from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from simulate_resource_optimization import aggregate_rows, load_or_generate_predictions, run_trial


PLOT_SCHEMES = ["random", "fixed_ch0", "semantic_hard", "semantic_hard_rep3", "semantic_soft_rep3", "oracle"]
SCHEME_LABELS = {
    "random": "Random",
    "fixed_ch0": "Fixed channel",
    "semantic_hard": "Semantic hard",
    "semantic_hard_rep3": "Semantic hard rep3",
    "semantic_soft_rep3": "Semantic soft rep3",
    "oracle": "Oracle",
}


def make_run_args(base: argparse.Namespace, uavs_per_step: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=base.seed,
        n_channels=base.n_channels,
        demand_channels=base.demand_channels,
        channel_axis=base.channel_axis,
        clean_threshold=base.clean_threshold,
        low_interference_threshold=base.low_interference_threshold,
        switch_cost=base.switch_cost,
        decision_steps=base.decision_steps,
        uavs_per_step=uavs_per_step,
        packet_bits=base.packet_bits,
        large_fec_overhead=base.large_fec_overhead,
        large_min_recovery_fraction=base.large_min_recovery_fraction,
    )


def plot_metric(rows: list[dict], metric: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for scheme in PLOT_SCHEMES:
        selected = [row for row in rows if row["scheme"] == scheme]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: row["uavs_per_step"])
        ax.plot(
            [row["uavs_per_step"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=2,
            label=SCHEME_LABELS.get(scheme, scheme),
        )
    ax.set_xlabel("Number of simultaneous UAV observations")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({row["uavs_per_step"] for row in rows}))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep multi-UAV pressure for spectrum resource optimization.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-loss", type=float, default=0.2)
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--uav-list", type=str, default="1,2,4,8")
    parser.add_argument("--decision-steps", type=int, default=1000)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--demand-channels", type=int, default=2)
    parser.add_argument("--channel-axis", type=str, default="y", choices=["x", "y"])
    parser.add_argument("--clean-threshold", type=float, default=0.02)
    parser.add_argument("--low-interference-threshold", type=float, default=0.10)
    parser.add_argument("--switch-cost", type=float, default=0.005)
    parser.add_argument("--large-fec-overhead", type=float, default=1.25)
    parser.add_argument("--large-min-recovery-fraction", type=float, default=0.25)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions, truths = load_or_generate_predictions(args)
    uav_values = [int(x) for x in args.uav_list.split(",") if x.strip()]

    all_rows: list[dict] = []
    for uavs in uav_values:
        run_args = make_run_args(args, uavs)
        detailed = []
        for trial in range(args.trials):
            detailed.extend(run_trial(predictions, truths, args.packet_loss, args.ber, run_args, trial))
        aggregated = aggregate_rows(detailed)
        for row in aggregated:
            row["uavs_per_step"] = uavs
            row["decision_steps"] = args.decision_steps
        all_rows.extend(aggregated)

    csv_path = args.out_dir / "multi_uav_pressure_sweep.csv"
    json_path = args.out_dir / "multi_uav_pressure_sweep.json"
    md_path = args.out_dir / "multi_uav_pressure_sweep.md"
    clean_plot = args.out_dir / "multi_uav_pressure_clean_rate.png"
    regret_plot = args.out_dir / "multi_uav_pressure_regret.png"

    fieldnames = [
        "uavs_per_step",
        "scheme",
        "packet_loss",
        "ber",
        "trials",
        "decision_steps",
        "mean_bits_per_frame",
        "compression_vs_spectrogram8",
        "compression_vs_iq12",
        "mean_true_occupancy",
        "mean_throughput_score",
        "mean_net_utility",
        "oracle_gap",
        "mean_regret_occupancy",
        "clean_channel_rate",
        "low_interference_rate",
        "oracle_equivalent_rate",
        "exact_oracle_channel_rate",
        "switch_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "dataset": "RadDet40k128HW001Tv2",
        "split": args.split,
        "frames": len(predictions),
        "packet_loss": args.packet_loss,
        "ber": args.ber,
        "uav_values": uav_values,
        "decision_steps": args.decision_steps,
        "trials": args.trials,
        "n_channels": args.n_channels,
        "demand_channels": args.demand_channels,
        "channel_axis": args.channel_axis,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plot_metric(all_rows, "clean_channel_rate", "Clean resource selection rate", clean_plot)
    plot_metric(all_rows, "mean_regret_occupancy", "Mean occupancy regret vs oracle", regret_plot)

    lines = [
        "# Multi-UAV Spectrum Pressure Sweep",
        "",
        "## Setup",
        "",
        f"- Packet loss: {args.packet_loss:g}",
        f"- BER: {args.ber:g}",
        f"- Candidate channels: {args.n_channels}",
        f"- Demand: contiguous block of {args.demand_channels} channel(s)",
        f"- Switch cost: {args.switch_cost:g}",
        f"- Decision steps per trial: {args.decision_steps}",
        f"- Trials: {args.trials}",
        "",
        "## Clean resource selection rate",
        "",
        "| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 | Oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for uavs in uav_values:
        by_scheme = {row["scheme"]: row for row in all_rows if row["uavs_per_step"] == uavs}
        lines.append(
            f"| {uavs} | "
            f"{by_scheme['random']['clean_channel_rate']:.4f} | "
            f"{by_scheme['fixed_ch0']['clean_channel_rate']:.4f} | "
            f"{by_scheme['semantic_hard']['clean_channel_rate']:.4f} | "
            f"{by_scheme['semantic_hard_rep3']['clean_channel_rate']:.4f} | "
            f"{by_scheme['semantic_soft_rep3']['clean_channel_rate']:.4f} | "
            f"{by_scheme['oracle']['clean_channel_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean occupancy regret vs oracle",
            "",
            "| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for uavs in uav_values:
        by_scheme = {row["scheme"]: row for row in all_rows if row["uavs_per_step"] == uavs}
        lines.append(
            f"| {uavs} | "
            f"{by_scheme['random']['mean_regret_occupancy']:.4f} | "
            f"{by_scheme['fixed_ch0']['mean_regret_occupancy']:.4f} | "
            f"{by_scheme['semantic_hard']['mean_regret_occupancy']:.4f} | "
            f"{by_scheme['semantic_hard_rep3']['mean_regret_occupancy']:.4f} | "
            f"{by_scheme['semantic_soft_rep3']['mean_regret_occupancy']:.4f} |"
        )
    best_uavs = max(uav_values)
    best = {row["scheme"]: row for row in all_rows if row["uavs_per_step"] == best_uavs}
    improvement = best["semantic_hard_rep3"]["clean_channel_rate"] - best["random"]["clean_channel_rate"]
    bit_ratio = best["semantic_hard_rep3"]["compression_vs_iq12"]
    lines.extend(
        [
            "",
            "## Net utility after switch cost",
            "",
            "| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 | Oracle |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for uavs in uav_values:
        by_scheme = {row["scheme"]: row for row in all_rows if row["uavs_per_step"] == uavs}
        lines.append(
            f"| {uavs} | "
            f"{by_scheme['random']['mean_net_utility']:.4f} | "
            f"{by_scheme['fixed_ch0']['mean_net_utility']:.4f} | "
            f"{by_scheme['semantic_hard']['mean_net_utility']:.4f} | "
            f"{by_scheme['semantic_hard_rep3']['mean_net_utility']:.4f} | "
            f"{by_scheme['semantic_soft_rep3']['mean_net_utility']:.4f} | "
            f"{by_scheme['oracle']['mean_net_utility']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"At {best_uavs} simultaneous UAV observations, semantic hard repetition improves clean resource selection "
                f"by {improvement:.4f} absolute over random while remaining {bit_ratio:.1f}x smaller than 12-bit I/Q forwarding."
            ),
            "This sweep is intended to show that the resource-optimization value of spectrum semantics becomes clearer as the spectrum-sharing pressure increases.",
            "",
            f"Clean-rate plot: `{clean_plot}`",
            f"Regret plot: `{regret_plot}`",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {clean_plot}")
    print(f"wrote {regret_plot}")


if __name__ == "__main__":
    main()
