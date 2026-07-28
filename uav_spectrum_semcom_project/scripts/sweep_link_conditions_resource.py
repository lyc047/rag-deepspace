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


PLOT_SCHEMES = [
    "random",
    "fixed_ch0",
    "semantic_hard",
    "semantic_hard_rep3",
    "spectrogram8_partial",
    "iq12_partial",
    "oracle",
]


def make_run_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        n_channels=args.n_channels,
        demand_channels=args.demand_channels,
        channel_axis=args.channel_axis,
        clean_threshold=args.clean_threshold,
        low_interference_threshold=args.low_interference_threshold,
        switch_cost=args.switch_cost,
        decision_steps=args.decision_steps,
        uavs_per_step=args.uavs_per_step,
        packet_bits=args.packet_bits,
        large_fec_overhead=args.large_fec_overhead,
        large_min_recovery_fraction=args.large_min_recovery_fraction,
    )


def plot_by_packet_loss(rows: list[dict], ber: float, metric: str, ylabel: str, output_path: Path) -> None:
    selected_rows = [row for row in rows if abs(row["ber"] - ber) < 1e-15 and row["scheme"] in PLOT_SCHEMES]
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    for scheme in PLOT_SCHEMES:
        scheme_rows = sorted([row for row in selected_rows if row["scheme"] == scheme], key=lambda row: row["packet_loss"])
        if not scheme_rows:
            continue
        ax.plot([row["packet_loss"] for row in scheme_rows], [row[metric] for row in scheme_rows], marker="o", label=scheme)
    ax.set_xlabel("Packet loss probability")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs packet loss at BER={ber:g}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep packet loss and BER for spectrum resource optimization.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-losses", type=str, default="0,0.03,0.05,0.1,0.2,0.3")
    parser.add_argument("--bers", type=str, default="0,1e-5,1e-4,1e-3")
    parser.add_argument("--plot-ber", type=float, default=1e-4)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--decision-steps", type=int, default=600)
    parser.add_argument("--uavs-per-step", type=int, default=4)
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
    run_args = make_run_args(args)
    packet_losses = [float(x) for x in args.packet_losses.split(",") if x.strip()]
    bers = [float(x) for x in args.bers.split(",") if x.strip()]

    detailed = []
    for packet_loss in packet_losses:
        for ber in bers:
            for trial in range(args.trials):
                detailed.extend(run_trial(predictions, truths, packet_loss, ber, run_args, trial))
    rows = aggregate_rows(detailed)

    csv_path = args.out_dir / "link_condition_resource_sweep.csv"
    json_path = args.out_dir / "link_condition_resource_sweep.json"
    md_path = args.out_dir / "link_condition_resource_sweep.md"
    clean_plot = args.out_dir / "link_condition_clean_rate.png"
    utility_plot = args.out_dir / "link_condition_net_utility.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "RadDet40k128HW001Tv2",
        "packet_losses": packet_losses,
        "bers": bers,
        "plot_ber": args.plot_ber,
        "trials": args.trials,
        "decision_steps": args.decision_steps,
        "uavs_per_step": args.uavs_per_step,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_by_packet_loss(rows, args.plot_ber, "clean_channel_rate", "Clean resource selection rate", clean_plot)
    plot_by_packet_loss(rows, args.plot_ber, "mean_net_utility", "Net utility", utility_plot)

    def pick(scheme: str, pl: float, ber: float) -> dict:
        return next(row for row in rows if row["scheme"] == scheme and abs(row["packet_loss"] - pl) < 1e-12 and abs(row["ber"] - ber) < 1e-15)

    lines = [
        "# Link-Condition Resource Optimization Sweep",
        "",
        "## Setup",
        "",
        f"- UAV observations per decision: {args.uavs_per_step}",
        f"- Decision steps/trial: {args.decision_steps}",
        f"- Trials: {args.trials}",
        f"- Large-payload partial recovery: FEC overhead {args.large_fec_overhead:g}, minimum recovery fraction {args.large_min_recovery_fraction:g}",
        "",
        f"## Representative results at BER={args.plot_ber:g}",
        "",
        "| Packet loss | Random clean | Semantic hard rep3 clean | Spectrogram partial clean | I/Q partial clean | Semantic hard rep3 net utility |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for pl in packet_losses:
        random_row = pick("random", pl, args.plot_ber)
        semantic_row = pick("semantic_hard_rep3", pl, args.plot_ber)
        spec_row = pick("spectrogram8_partial", pl, args.plot_ber)
        iq_row = pick("iq12_partial", pl, args.plot_ber)
        lines.append(
            f"| {pl:g} | {random_row['clean_channel_rate']:.4f} | {semantic_row['clean_channel_rate']:.4f} | "
            f"{spec_row['clean_channel_rate']:.4f} | {iq_row['clean_channel_rate']:.4f} | {semantic_row['mean_net_utility']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This sweep evaluates whether the spectrum resource decision remains robust under changing packet loss and BER. "
            "The partial large-payload baselines are intentionally more favorable than the earlier all-or-nothing gate, so they provide a fairer comparison point for semantic reporting.",
            "",
            f"Clean-rate plot: `{clean_plot}`",
            f"Net-utility plot: `{utility_plot}`",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {clean_plot}")
    print(f"wrote {utility_plot}")


if __name__ == "__main__":
    main()
