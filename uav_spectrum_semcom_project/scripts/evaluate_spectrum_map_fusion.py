from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from simulate_resource_optimization import (
    aggregate_group_boxes,
    all_channel_occupancy,
    block_occupancy,
    flatten_group_boxes,
    load_or_generate_predictions,
    regroup_flat_boxes,
    sample_multi_uav_groups,
    transmit_large_payload_partial,
)
from simulate_semantic_link import transmit_semantic_boxes
from spectrum_semcom.bit_budget import iq_payload_bits, stft_payload_bits


SCHEMES = ["semantic_hard", "semantic_hard_rep3", "spectrogram8_partial", "iq12_partial"]


def make_args(args: argparse.Namespace, uavs: int) -> SimpleNamespace:
    return SimpleNamespace(
        n_channels=args.n_channels,
        demand_channels=args.demand_channels,
        channel_axis=args.channel_axis,
        packet_bits=args.packet_bits,
        large_fec_overhead=args.large_fec_overhead,
        large_min_recovery_fraction=args.large_min_recovery_fraction,
        uavs_per_step=uavs,
        decision_steps=args.decision_steps,
    )


def delivered_for_scheme(
    scheme: str,
    flat_predictions: list[list[np.ndarray]],
    packet_loss: float,
    ber: float,
    args: SimpleNamespace,
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], float]:
    raw_spectrogram_bits = stft_payload_bits((128, 128), 8)
    raw_iq_bits = iq_payload_bits(1_000_000, 12, 12)
    if scheme == "semantic_hard":
        return transmit_semantic_boxes(flat_predictions, packet_loss, ber, "hard", rng)
    if scheme == "semantic_hard_rep3":
        return transmit_semantic_boxes(flat_predictions, packet_loss, ber, "hard_rep3", rng)
    if scheme == "spectrogram8_partial":
        return transmit_large_payload_partial(
            flat_predictions,
            packet_loss,
            ber,
            raw_spectrogram_bits,
            args.packet_bits,
            rng,
            repeats=1,
            fec_overhead=args.large_fec_overhead,
            min_recovery_fraction=args.large_min_recovery_fraction,
        )
    if scheme == "iq12_partial":
        return transmit_large_payload_partial(
            flat_predictions,
            packet_loss,
            ber,
            raw_iq_bits,
            args.packet_bits,
            rng,
            repeats=1,
            fec_overhead=args.large_fec_overhead,
            min_recovery_fraction=args.large_min_recovery_fraction,
        )
    raise ValueError(scheme)


def map_metrics(truth_occ: np.ndarray, sensed_occ: np.ndarray, truth_block: np.ndarray, sensed_block: np.ndarray, clean_threshold: float) -> dict[str, float]:
    truth_clean = truth_block <= clean_threshold
    sensed_clean = sensed_block <= clean_threshold
    return {
        "channel_occupancy_mae": float(np.mean(np.abs(truth_occ - sensed_occ))),
        "block_occupancy_mae": float(np.mean(np.abs(truth_block - sensed_block))),
        "clean_block_accuracy": float(np.mean(truth_clean == sensed_clean)),
        "mean_truth_occupancy": float(np.mean(truth_occ)),
        "mean_sensed_occupancy": float(np.mean(sensed_occ)),
    }


def plot_mae(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for scheme in SCHEMES:
        selected = sorted([row for row in rows if row["scheme"] == scheme], key=lambda row: row["uavs_per_step"])
        ax.plot([row["uavs_per_step"] for row in selected], [row["block_occupancy_mae"] for row in selected], marker="o", label=scheme)
    ax.set_xlabel("Number of simultaneous UAV observations")
    ax.set_ylabel("Block occupancy MAE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-UAV spectrum-map fusion accuracy.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-loss", type=float, default=0.2)
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--uav-list", type=str, default="1,2,4,8")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--decision-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--demand-channels", type=int, default=2)
    parser.add_argument("--channel-axis", type=str, default="y", choices=["x", "y"])
    parser.add_argument("--clean-threshold", type=float, default=0.02)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--large-fec-overhead", type=float, default=1.25)
    parser.add_argument("--large-min-recovery-fraction", type=float, default=0.25)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions, truths = load_or_generate_predictions(args)
    uav_values = [int(x) for x in args.uav_list.split(",") if x.strip()]

    detailed: list[dict] = []
    for uavs in uav_values:
        local_args = make_args(args, uavs)
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + trial + uavs * 1000)
            groups = sample_multi_uav_groups(len(predictions), args.decision_steps, uavs, rng)
            grouped_truths = aggregate_group_boxes(truths, groups)
            flat_predictions = flatten_group_boxes(predictions, groups)
            truth_occ = all_channel_occupancy(grouped_truths, args.n_channels, args.channel_axis)
            truth_block = block_occupancy(truth_occ, args.demand_channels)
            for scheme in SCHEMES:
                delivered_flat, mean_bits_per_uav = delivered_for_scheme(
                    scheme, flat_predictions, args.packet_loss, args.ber, local_args, rng
                )
                delivered_grouped = regroup_flat_boxes(delivered_flat, groups)
                sensed_occ = all_channel_occupancy(delivered_grouped, args.n_channels, args.channel_axis)
                sensed_block = block_occupancy(sensed_occ, args.demand_channels)
                detailed.append(
                    {
                        "uavs_per_step": uavs,
                        "scheme": scheme,
                        "trial": trial,
                        "mean_bits_per_decision": mean_bits_per_uav * uavs,
                        **map_metrics(truth_occ, sensed_occ, truth_block, sensed_block, args.clean_threshold),
                    }
                )

    rows = []
    for key in sorted({(row["uavs_per_step"], row["scheme"]) for row in detailed}):
        items = [row for row in detailed if (row["uavs_per_step"], row["scheme"]) == key]
        row = {"uavs_per_step": key[0], "scheme": key[1], "trials": len(items)}
        for metric in [
            "mean_bits_per_decision",
            "channel_occupancy_mae",
            "block_occupancy_mae",
            "clean_block_accuracy",
            "mean_truth_occupancy",
            "mean_sensed_occupancy",
        ]:
            row[metric] = float(np.mean([item[metric] for item in items]))
        rows.append(row)

    csv_path = args.out_dir / "spectrum_map_fusion_eval.csv"
    json_path = args.out_dir / "spectrum_map_fusion_eval.json"
    md_path = args.out_dir / "spectrum_map_fusion_eval.md"
    plot_path = args.out_dir / "spectrum_map_fusion_block_mae.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "packet_loss": args.packet_loss, "ber": args.ber}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_mae(rows, plot_path)

    lines = [
        "# Multi-UAV Spectrum Map Fusion Evaluation",
        "",
        f"- Packet loss: {args.packet_loss:g}",
        f"- BER: {args.ber:g}",
        f"- Decision steps/trial: {args.decision_steps}",
        f"- Trials: {args.trials}",
        "",
        "| UAVs | Scheme | Block MAE | Channel MAE | Clean-block accuracy | bits/decision |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['uavs_per_step']} | {row['scheme']} | {row['block_occupancy_mae']:.4f} | "
            f"{row['channel_occupancy_mae']:.4f} | {row['clean_block_accuracy']:.4f} | "
            f"{row['mean_bits_per_decision']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This evaluates the spectrum-map layer directly, before resource selection. Lower MAE means the fused received reports better approximate the true multi-UAV occupancy map.",
            "",
            f"Block-MAE plot: `{plot_path}`",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
