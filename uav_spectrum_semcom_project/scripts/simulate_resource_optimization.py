from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import iq_payload_bits, stft_payload_bits
from simulate_semantic_link import (
    load_or_generate_predictions,
    repetition_ber,
    repetition_packet_loss,
    transmit_large_payload_gate,
    transmit_semantic_boxes,
)


def _box_intersection_area(box: np.ndarray, rect: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    rx0, ry0, rx1, ry1 = rect
    ix0 = max(min(x0, x1), rx0)
    iy0 = max(min(y0, y1), ry0)
    ix1 = min(max(x0, x1), rx1)
    iy1 = min(max(y0, y1), ry1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def channel_rect(index: int, n_channels: int, axis: str) -> tuple[float, float, float, float]:
    lo = index / n_channels
    hi = (index + 1) / n_channels
    if axis == "y":
        return (0.0, lo, 1.0, hi)
    if axis == "x":
        return (lo, 0.0, hi, 1.0)
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def frame_channel_occupancy(boxes: list[np.ndarray], n_channels: int, axis: str) -> np.ndarray:
    values = np.zeros(n_channels, dtype=np.float32)
    channel_area = 1.0 / n_channels
    for idx in range(n_channels):
        rect = channel_rect(idx, n_channels, axis)
        occupied_area = sum(_box_intersection_area(box, rect) for box in boxes)
        values[idx] = min(1.0, occupied_area / channel_area)
    return values


def all_channel_occupancy(frames: list[list[np.ndarray]], n_channels: int, axis: str) -> np.ndarray:
    return np.stack([frame_channel_occupancy(frame, n_channels, axis) for frame in frames], axis=0)


def transmit_large_payload_partial(
    predictions: list[list[np.ndarray]],
    packet_loss: float,
    ber: float,
    payload_bits_per_frame: int,
    packet_bits: int,
    rng: np.random.Generator,
    repeats: int = 1,
    fec_overhead: float = 1.0,
    min_recovery_fraction: float = 0.25,
) -> tuple[list[list[np.ndarray]], float]:
    """A less conservative large-payload baseline with partial recovery.

    The payload is split into packets. Each packet is useful only if it arrives
    and has no bit error after optional repetition. Instead of dropping the
    whole frame unless every packet survives, this model keeps each detected
    box with a probability proportional to the recovered packet fraction. This
    approximates partial spectrogram recovery, simple source-channel coding, or
    region-level decoding without assuming an implementation-specific codec.
    """

    effective_payload_bits = int(np.ceil(payload_bits_per_frame * float(fec_overhead)))
    packets = max(1, int(np.ceil(effective_payload_bits / packet_bits)))
    eff_packet_loss = repetition_packet_loss(packet_loss, repeats)
    eff_ber = repetition_ber(ber, repeats)
    packet_success = (1.0 - eff_packet_loss) * ((1.0 - eff_ber) ** packet_bits)
    delivered: list[list[np.ndarray]] = []
    for boxes in predictions:
        recovered_packets = rng.binomial(packets, packet_success)
        recovered_fraction = recovered_packets / packets
        if recovered_fraction < min_recovery_fraction:
            delivered.append([])
            continue
        keep_prob = min(1.0, recovered_fraction)
        delivered.append([box for box in boxes if rng.random() < keep_prob])
    return delivered, float(effective_payload_bits * repeats)


def sample_multi_uav_groups(
    n_frames: int,
    n_steps: int,
    uavs_per_step: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Sample frame indices that represent simultaneous observations from multiple UAVs."""

    if n_frames <= 0:
        return []
    uavs = max(1, int(uavs_per_step))
    steps = max(1, int(n_steps))
    return [rng.integers(0, n_frames, size=uavs).astype(int).tolist() for _ in range(steps)]


def aggregate_group_boxes(frames: list[list[np.ndarray]], groups: list[list[int]]) -> list[list[np.ndarray]]:
    aggregated: list[list[np.ndarray]] = []
    for group in groups:
        boxes: list[np.ndarray] = []
        for idx in group:
            boxes.extend(frames[idx])
        aggregated.append(boxes)
    return aggregated


def flatten_group_boxes(frames: list[list[np.ndarray]], groups: list[list[int]]) -> list[list[np.ndarray]]:
    return [frames[idx] for group in groups for idx in group]


def regroup_flat_boxes(flat_frames: list[list[np.ndarray]], groups: list[list[int]]) -> list[list[np.ndarray]]:
    regrouped: list[list[np.ndarray]] = []
    cursor = 0
    for group in groups:
        boxes: list[np.ndarray] = []
        for _ in group:
            boxes.extend(flat_frames[cursor])
            cursor += 1
        regrouped.append(boxes)
    return regrouped


def block_occupancy(channel_occupancy: np.ndarray, demand_channels: int) -> np.ndarray:
    demand = max(1, min(int(demand_channels), int(channel_occupancy.shape[-1])))
    if demand == 1:
        return channel_occupancy
    blocks = []
    for start in range(channel_occupancy.shape[-1] - demand + 1):
        blocks.append(np.mean(channel_occupancy[..., start : start + demand], axis=-1))
    return np.stack(blocks, axis=-1)


def choose_min_resource(
    sensed_occupancy: np.ndarray,
    rng: np.random.Generator,
    margin: float = 1e-9,
) -> int:
    best = float(np.min(sensed_occupancy))
    candidates = np.flatnonzero(sensed_occupancy <= best + margin)
    return int(rng.choice(candidates))


def evaluate_decisions(
    chosen: np.ndarray,
    truth_resource_occ: np.ndarray,
    oracle_chosen: np.ndarray,
    clean_threshold: float,
    low_interference_threshold: float,
    switch_cost: float = 0.0,
) -> dict[str, float]:
    chosen_occ = truth_resource_occ[np.arange(len(chosen)), chosen]
    oracle_occ = truth_resource_occ[np.arange(len(oracle_chosen)), oracle_chosen]
    throughput = 1.0 - chosen_occ
    oracle_throughput = 1.0 - oracle_occ
    switch_events = np.zeros(len(chosen), dtype=np.float32)
    if len(chosen) > 1:
        switch_events[1:] = chosen[1:] != chosen[:-1]
    switches = float(np.mean(switch_events)) if len(chosen) else 0.0
    net_utility = throughput - float(switch_cost) * switch_events
    return {
        "mean_true_occupancy": float(np.mean(chosen_occ)),
        "mean_throughput_score": float(np.mean(throughput)),
        "mean_net_utility": float(np.mean(net_utility)),
        "oracle_gap": float(np.mean(oracle_throughput - throughput)),
        "mean_regret_occupancy": float(np.mean(chosen_occ - oracle_occ)),
        "clean_channel_rate": float(np.mean(chosen_occ <= clean_threshold)),
        "low_interference_rate": float(np.mean(chosen_occ <= low_interference_threshold)),
        "oracle_equivalent_rate": float(np.mean(np.abs(chosen_occ - oracle_occ) <= 1e-6)),
        "exact_oracle_channel_rate": float(np.mean(chosen == oracle_chosen)),
        "switch_rate": float(switches),
    }


def run_trial(
    predictions: list[list[np.ndarray]],
    truths: list[list[np.ndarray]],
    packet_loss: float,
    ber: float,
    args: argparse.Namespace,
    trial: int,
) -> list[dict[str, float | str]]:
    seed = args.seed + trial + int(packet_loss * 1e6) + int(ber * 1e9)
    rng = np.random.default_rng(seed)

    groups = sample_multi_uav_groups(
        n_frames=len(predictions),
        n_steps=args.decision_steps,
        uavs_per_step=args.uavs_per_step,
        rng=rng,
    )
    grouped_truths = aggregate_group_boxes(truths, groups)
    grouped_predictions = aggregate_group_boxes(predictions, groups)
    flat_predictions = flatten_group_boxes(predictions, groups)

    truth_occ = all_channel_occupancy(grouped_truths, args.n_channels, args.channel_axis)
    truth_resource_occ = block_occupancy(truth_occ, args.demand_channels)
    oracle_chosen = np.array([choose_min_resource(row, rng) for row in truth_resource_occ], dtype=np.int64)
    raw_spectrogram_bits = stft_payload_bits((128, 128), 8)
    raw_iq_bits = iq_payload_bits(1_000_000, 12, 12)

    schemes: list[tuple[str, list[list[np.ndarray]], float]] = []
    schemes.append(("oracle", grouped_truths, 0.0))

    fixed_boxes: list[list[np.ndarray]] = [[] for _ in grouped_predictions]
    schemes.append(("fixed_ch0", fixed_boxes, 0.0))
    schemes.append(("random", fixed_boxes, 0.0))

    for scheme in ["hard", "hard_rep3", "soft", "soft_rep3"]:
        delivered_flat, mean_bits_per_uav = transmit_semantic_boxes(flat_predictions, packet_loss, ber, scheme, rng)
        delivered_grouped = regroup_flat_boxes(delivered_flat, groups)
        schemes.append((f"semantic_{scheme}", delivered_grouped, mean_bits_per_uav * args.uavs_per_step))

    delivered_flat, mean_bits = transmit_large_payload_gate(
        flat_predictions, packet_loss, ber, raw_spectrogram_bits, args.packet_bits, rng
    )
    schemes.append(("spectrogram8_gate", regroup_flat_boxes(delivered_flat, groups), mean_bits * args.uavs_per_step))
    delivered_flat, mean_bits = transmit_large_payload_partial(
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
    schemes.append(("spectrogram8_partial", regroup_flat_boxes(delivered_flat, groups), mean_bits * args.uavs_per_step))
    delivered_flat, mean_bits = transmit_large_payload_partial(
        flat_predictions,
        packet_loss,
        ber,
        raw_spectrogram_bits,
        args.packet_bits,
        rng,
        repeats=3,
        fec_overhead=args.large_fec_overhead,
        min_recovery_fraction=args.large_min_recovery_fraction,
    )
    schemes.append(("spectrogram8_rep3_partial", regroup_flat_boxes(delivered_flat, groups), mean_bits * args.uavs_per_step))
    delivered_flat, mean_bits = transmit_large_payload_gate(flat_predictions, packet_loss, ber, raw_iq_bits, args.packet_bits, rng)
    schemes.append(("iq12_gate", regroup_flat_boxes(delivered_flat, groups), mean_bits * args.uavs_per_step))
    delivered_flat, mean_bits = transmit_large_payload_partial(
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
    schemes.append(("iq12_partial", regroup_flat_boxes(delivered_flat, groups), mean_bits * args.uavs_per_step))

    rows: list[dict[str, float | str]] = []
    for name, sensed_boxes, bits in schemes:
        if name == "oracle":
            chosen = oracle_chosen
        elif name == "fixed_ch0":
            chosen = np.zeros(len(grouped_predictions), dtype=np.int64)
        elif name == "random":
            chosen = rng.integers(0, truth_resource_occ.shape[1], size=len(grouped_predictions), dtype=np.int64)
        else:
            sensed_occ = all_channel_occupancy(sensed_boxes, args.n_channels, args.channel_axis)
            sensed_resource_occ = block_occupancy(sensed_occ, args.demand_channels)
            chosen = np.array([choose_min_resource(row, rng) for row in sensed_resource_occ], dtype=np.int64)
        metrics = evaluate_decisions(
            chosen=chosen,
            truth_resource_occ=truth_resource_occ,
            oracle_chosen=oracle_chosen,
            clean_threshold=args.clean_threshold,
            low_interference_threshold=args.low_interference_threshold,
            switch_cost=args.switch_cost,
        )
        rows.append(
            {
                "scheme": name,
                "packet_loss": packet_loss,
                "ber": ber,
                "trial": trial,
                "mean_bits_per_frame": float(bits),
                "compression_vs_spectrogram8": raw_spectrogram_bits / max(float(bits), 1.0) if bits else 0.0,
                "compression_vs_iq12": raw_iq_bits / max(float(bits), 1.0) if bits else 0.0,
                **metrics,
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    keys = [
        "mean_bits_per_frame",
        "compression_vs_spectrogram8",
        "compression_vs_iq12",
        "mean_true_occupancy",
        "mean_throughput_score",
        "oracle_gap",
        "mean_regret_occupancy",
        "mean_net_utility",
        "clean_channel_rate",
        "low_interference_rate",
        "oracle_equivalent_rate",
        "exact_oracle_channel_rate",
        "switch_rate",
    ]
    groups: dict[tuple[str, float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["scheme"]), float(row["packet_loss"]), float(row["ber"])), []).append(row)
    out = []
    for (scheme, packet_loss, ber), items in sorted(groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        merged: dict[str, float | str] = {"scheme": scheme, "packet_loss": packet_loss, "ber": ber, "trials": len(items)}
        for key in keys:
            merged[key] = float(np.mean([float(item[key]) for item in items]))
        out.append(merged)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate spectrum-resource optimization from semantic occupancy reports.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-losses", type=str, default="0,0.05,0.1,0.2")
    parser.add_argument("--bers", type=str, default="0,1e-5,1e-4")
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--demand-channels", type=int, default=2)
    parser.add_argument("--uavs-per-step", type=int, default=4)
    parser.add_argument("--decision-steps", type=int, default=1000)
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

    detailed_rows: list[dict[str, float | str]] = []
    for packet_loss in [float(x) for x in args.packet_losses.split(",") if x.strip()]:
        for ber in [float(x) for x in args.bers.split(",") if x.strip()]:
            for trial in range(args.trials):
                detailed_rows.extend(run_trial(predictions, truths, packet_loss, ber, args, trial))

    rows = aggregate_rows(detailed_rows)

    csv_path = args.out_dir / "resource_optimization_simulation.csv"
    json_path = args.out_dir / "resource_optimization_simulation.json"
    md_path = args.out_dir / "resource_optimization_simulation.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "RadDet40k128HW001Tv2",
        "source_detector": "TinyOccupancyCNN occupancy mask",
        "split": args.split,
        "frames": len(predictions),
        "decision_steps": args.decision_steps,
        "uavs_per_step": args.uavs_per_step,
        "n_channels": args.n_channels,
        "demand_channels": args.demand_channels,
        "channel_axis": args.channel_axis,
        "clean_threshold": args.clean_threshold,
        "low_interference_threshold": args.low_interference_threshold,
        "switch_cost": args.switch_cost,
        "large_fec_overhead": args.large_fec_overhead,
        "large_min_recovery_fraction": args.large_min_recovery_fraction,
        "trials": args.trials,
        "notes": [
            "Frequency resources are approximated by splitting the selected normalized image axis into candidate channels.",
            "Each decision step aggregates simultaneous observations from multiple sampled UAVs to create a high-pressure spectrum-sharing scenario.",
            "The decision module selects the contiguous resource block with the lowest sensed occupancy; oracle uses ground truth boxes.",
            "Throughput score is a normalized proxy defined as 1 - true occupancy of the selected resource block.",
            "Net utility subtracts a per-switch cost from the throughput score to penalize frequent resource changes.",
            "Semantic, spectrogram, and I/Q schemes transmit one packet stream per UAV and then fuse delivered reports at the receiver.",
            "Large-payload spectrogram/IQ forwarding uses the same conservative gate as the semantic-link simulation.",
            "Partial large-payload baselines approximate packet-level partial recovery and simple FEC overhead.",
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def selected(pl: float, ber: float) -> list[dict[str, float | str]]:
        return [row for row in rows if abs(float(row["packet_loss"]) - pl) < 1e-12 and abs(float(row["ber"]) - ber) < 1e-15]

    lines = [
        "# Spectrum Resource Optimization Simulation",
        "",
        "## Setup",
        "",
        f"- Dataset: RadDet40k128HW001Tv2 `{args.split}` split, {len(predictions)} frames.",
        f"- Multi-UAV pressure: {args.uavs_per_step} UAV observations are fused per decision step, {args.decision_steps} decision steps.",
        f"- Candidate channels: {args.n_channels}, generated by splitting normalized `{args.channel_axis}` axis.",
        f"- Transmission demand: contiguous block of {args.demand_channels} channel(s).",
        f"- Decision rule: choose the contiguous resource block with minimum sensed occupancy.",
        f"- Clean channel threshold: true occupancy <= {args.clean_threshold:g}.",
        f"- Low-interference threshold: true occupancy <= {args.low_interference_threshold:g}.",
        f"- Switch cost: {args.switch_cost:g} normalized utility per resource change.",
        "",
        "## Representative operating points",
        "",
    ]
    for pl, ber in [(0.0, 0.0), (0.05, 1e-5), (0.1, 1e-4), (0.2, 1e-4)]:
        sample = selected(pl, ber)
        if not sample:
            continue
        lines.extend(
            [
                f"### packet_loss={pl:g}, BER={ber:g}",
                "",
                "| Scheme | Throughput score | Net utility | Clean rate | Low-interference rate | Oracle-equivalent rate | Regret | Switch rate | bits/decision-step | vs 8-bit spectrogram | vs 12-bit IQ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sample:
            lines.append(
                f"| {row['scheme']} | {float(row['mean_throughput_score']):.4f} | "
                f"{float(row['mean_net_utility']):.4f} | "
                f"{float(row['clean_channel_rate']):.4f} | {float(row['low_interference_rate']):.4f} | "
                f"{float(row['oracle_equivalent_rate']):.4f} | {float(row['mean_regret_occupancy']):.4f} | "
                f"{float(row['switch_rate']):.4f} | {float(row['mean_bits_per_frame']):.1f} | "
                f"{float(row['compression_vs_spectrogram8']):.1f}x | {float(row['compression_vs_iq12']):.1f}x |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This experiment closes the loop from spectrum semantic extraction to a resource decision. "
            "The current detector is not yet a high-accuracy classifier, so the resource-optimization metric focuses on whether the semantic occupancy report can help avoid occupied channels under limited link budget. "
            "Semantic repetition is expected to be useful when packet erasure exists because it keeps the decision input available while remaining far smaller than full spectrogram or I/Q forwarding.",
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
