from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import packet_header_bits
from spectrum_semcom.visdrone import VisDroneFrame, iter_visdrone_frames
from simulate_multimodal_policy import PRIORITY_CLASSES, estimate_roi_bits, roi_area_fraction, visual_priority


ACTION_BITS = {
    "summary_only": 0,
    "semantic_only": 1,
    "semantic_plus_roi": 2,
    "lowres_plus_semantic": 3,
    "jpeg_full": 4,
}


def load_spectrum_operating_points(path: Path, ber: float, scheme_filter: set[str]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in data["rows"]:
        if abs(float(row["ber"]) - ber) > 1e-15:
            continue
        if row["scheme"] not in scheme_filter:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No spectrum operating points found in {path}")
    return sorted(rows, key=lambda row: (float(row["packet_loss"]), row["scheme"]))


def priority_object_count(frame: VisDroneFrame) -> int:
    return sum(1 for box in frame.boxes if box.category in PRIORITY_CLASSES)


def lowres_bits(frame: VisDroneFrame, fraction: float) -> float:
    return frame.jpeg_bits * float(fraction)


def action_bits(frame: VisDroneFrame, action: str, lowres_fraction: float) -> float:
    summary_bits = packet_header_bits(metadata_bits=128)
    semantic_bits = frame.semantic_box_bits
    if action == "summary_only":
        return float(summary_bits)
    if action == "semantic_only":
        return float(semantic_bits)
    if action == "semantic_plus_roi":
        return float(semantic_bits + estimate_roi_bits(frame))
    if action == "lowres_plus_semantic":
        return float(semantic_bits + lowres_bits(frame, lowres_fraction))
    if action == "jpeg_full":
        return float(frame.jpeg_bits)
    raise ValueError(action)


def choose_rule_action(
    frame: VisDroneFrame,
    spectrum_quality: float,
    packet_loss: float,
    ber: float,
    args: argparse.Namespace,
    high_priority: bool,
    roi_fraction: float,
    clean_high: float,
    clean_mid: float,
    roi_heavy_threshold: float,
) -> str:
    if not high_priority and spectrum_quality < clean_mid:
        return "summary_only"

    base_action = "semantic_only" if high_priority or spectrum_quality >= clean_mid else "summary_only"
    if spectrum_quality < clean_high:
        return base_action

    candidates = [base_action]
    if high_priority:
        candidates.append("semantic_plus_roi" if roi_fraction <= roi_heavy_threshold else "lowres_plus_semantic")
        candidates.append("lowres_plus_semantic")

    # Practical rule: only upgrade payload granularity if the estimated utility
    # under the current spectrum/link state is better than semantic-only.
    scored = [
        (
            action_scores(
                frame,
                action,
                spectrum_quality,
                packet_loss,
                ber,
                args.packet_bits,
                args.lowres_fraction,
                args.bit_cost_per_mbit,
                args.priority_weight,
                args.detail_weight,
                args.semantic_weight,
            )["utility"],
            action,
        )
        for action in candidates
    ]
    return max(scored, key=lambda item: item[0])[1]


def delivered_probability(bits: float, clean_rate: float, packet_loss: float, ber: float, packet_bits: int) -> float:
    packets = max(1, int(np.ceil(bits / packet_bits)))
    # Clean resources reduce effective packet erasure; poor resources increase it.
    effective_packet_loss = np.clip(packet_loss * (1.0 - 0.85 * clean_rate), 0.0, 0.95)
    # A coarse packetized payload success model. Large images are naturally harder to deliver.
    packet_success = (1.0 - effective_packet_loss) ** packets
    # Assume ordinary payloads have basic coding/interleaving, so raw BER does
    # not invalidate the entire application payload bit-for-bit. Packet loss is
    # still the dominant impairment for large transfers.
    effective_ber = ber * 0.02
    bit_success = (1.0 - effective_ber) ** bits
    return float(np.clip(packet_success * bit_success, 0.0, 1.0))


def action_scores(
    frame: VisDroneFrame,
    action: str,
    clean_rate: float,
    packet_loss: float,
    ber: float,
    packet_bits: int,
    lowres_fraction: float,
    bit_cost_per_mbit: float,
    priority_weight: float,
    detail_weight: float,
    semantic_weight: float,
) -> dict[str, float]:
    bits = action_bits(frame, action, lowres_fraction)
    p_deliver = delivered_probability(bits, clean_rate, packet_loss, ber, packet_bits)
    high_priority = visual_priority(frame, min_priority_objects=1)
    priority_count = priority_object_count(frame)
    semantic_value = 0.0
    detail_value = 0.0
    if action in {"semantic_only", "semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        semantic_value = semantic_weight
    if action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        detail_value = detail_weight
    if action == "summary_only":
        semantic_value = 0.15 * semantic_weight
    task_value = semantic_value + detail_value
    if high_priority:
        task_value *= priority_weight
    # Dense visual scenes benefit more from detailed transmission, but only up to a cap.
    density_bonus = min(1.4, 1.0 + priority_count / 80.0)
    if action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        task_value *= density_bonus
    expected_task = p_deliver * task_value
    utility = expected_task - bit_cost_per_mbit * (bits / 1e6)
    return {
        "bits": bits,
        "delivery_probability": p_deliver,
        "expected_task": expected_task,
        "utility": utility,
        "detail_delivered_probability": p_deliver if action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"} else 0.0,
        "semantic_delivered_probability": p_deliver if action != "summary_only" else 0.15 * p_deliver,
    }


def choose_oracle_action(frame: VisDroneFrame, clean_rate: float, packet_loss: float, ber: float, args: argparse.Namespace) -> str:
    candidates = ["summary_only", "semantic_only", "semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"]
    scores = [
        (
            action_scores(
                frame,
                action,
                clean_rate,
                packet_loss,
                ber,
                args.packet_bits,
                args.lowres_fraction,
                args.bit_cost_per_mbit,
                args.priority_weight,
                args.detail_weight,
                args.semantic_weight,
            )["utility"],
            action,
        )
        for action in candidates
    ]
    return max(scores, key=lambda item: item[0])[1]


def policy_action(policy: str, frame: VisDroneFrame, clean_rate: float, packet_loss: float, ber: float, args: argparse.Namespace) -> str:
    high_priority = visual_priority(frame, args.min_priority_objects)
    roi_fraction = roi_area_fraction(frame)
    if policy == "summary_only":
        return "summary_only"
    if policy == "semantic_only":
        return "semantic_only"
    if policy == "roi_always":
        return "semantic_plus_roi"
    if policy == "lowres_always":
        return "lowres_plus_semantic"
    if policy == "jpeg_always":
        return "jpeg_full"
    if policy == "spectrum_rule":
        return choose_rule_action(
            frame,
            clean_rate,
            packet_loss,
            ber,
            args,
            high_priority,
            roi_fraction,
            args.clean_high,
            args.clean_mid,
            args.roi_heavy_threshold,
        )
    if policy == "oracle_policy":
        return choose_oracle_action(frame, clean_rate, packet_loss, ber, args)
    raise ValueError(policy)


def simulate_policy(frames: list[VisDroneFrame], spectrum_row: dict, policy: str, args: argparse.Namespace) -> dict[str, float | str]:
    packet_loss = float(spectrum_row["packet_loss"])
    clean_rate = float(spectrum_row["clean_channel_rate"])
    ber = float(spectrum_row["ber"])
    rng_offset = int(hashlib.sha256((policy + spectrum_row["scheme"] + str(packet_loss)).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.seed + rng_offset % 100000)

    action_counts = {action: 0 for action in ACTION_BITS}
    bits = []
    utilities = []
    expected_tasks = []
    semantic_probs = []
    detail_probs = []
    failed = []
    priority_detail_probs = []
    high_priority_flags = []

    for frame in frames:
        action = policy_action(policy, frame, clean_rate, packet_loss, ber, args)
        action_counts[action] += 1
        score = action_scores(
            frame,
            action,
            clean_rate,
            packet_loss,
            ber,
            args.packet_bits,
            args.lowres_fraction,
            args.bit_cost_per_mbit,
            args.priority_weight,
            args.detail_weight,
            args.semantic_weight,
        )
        delivered = rng.random() < score["delivery_probability"]
        bits.append(score["bits"])
        utilities.append(score["utility"])
        expected_tasks.append(score["expected_task"])
        semantic_probs.append(score["semantic_delivered_probability"])
        detail_probs.append(score["detail_delivered_probability"])
        failed.append(0.0 if delivered else 1.0)
        hp = visual_priority(frame, args.min_priority_objects)
        high_priority_flags.append(hp)
        priority_detail_probs.append(score["detail_delivered_probability"] if hp else 0.0)

    high_priority_count = max(1, sum(high_priority_flags))
    mean_bits = float(np.mean(bits))
    mean_detail = float(np.mean(detail_probs))
    mean_priority_detail = float(sum(priority_detail_probs) / high_priority_count)
    return {
        "spectrum_scheme": spectrum_row["scheme"],
        "packet_loss": packet_loss,
        "ber": ber,
        "clean_rate": clean_rate,
        "policy": policy,
        "mean_bits_per_frame": mean_bits,
        "mean_expected_task": float(np.mean(expected_tasks)),
        "mean_utility": float(np.mean(utilities)),
        "semantic_delivery_score": float(np.mean(semantic_probs)),
        "detail_delivery_score": mean_detail,
        "priority_detail_score": mean_priority_detail,
        "failed_transmission_rate": float(np.mean(failed)),
        "detail_per_mbit": mean_detail / max(mean_bits / 1e6, 1e-9),
        "priority_detail_per_mbit": mean_priority_detail / max(mean_bits / 1e6, 1e-9),
        **{f"action_{action}_ratio": action_counts[action] / len(frames) for action in ACTION_BITS},
    }


def plot_tradeoff(rows: list[dict], output_path: Path, spectrum_scheme: str, packet_loss: float) -> None:
    selected = [
        row
        for row in rows
        if row["spectrum_scheme"] == spectrum_scheme and abs(float(row["packet_loss"]) - packet_loss) < 1e-12
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for row in selected:
        ax.scatter(row["mean_bits_per_frame"] / 1000, row["mean_utility"], s=70)
        ax.annotate(row["policy"], (row["mean_bits_per_frame"] / 1000, row["mean_utility"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean payload (kbit/frame)")
    ax.set_ylabel("Mean utility")
    ax.set_title(f"Policy trade-off under {spectrum_scheme}, packet_loss={packet_loss:g}")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal multimodal decision policy simulation for spectrum-aware UAV visual semantic transmission.")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--spectrum-summary", type=Path, default=PROJECT_DIR / "results" / "phase1" / "resource_optimization_simulation.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--plot-packet-loss", type=float, default=0.2)
    parser.add_argument("--spectrum-schemes", type=str, default="random,semantic_hard,semantic_hard_rep3,spectrogram8_partial,oracle")
    parser.add_argument("--policies", type=str, default="summary_only,semantic_only,roi_always,lowres_always,jpeg_always,spectrum_rule,oracle_policy")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-priority-objects", type=int, default=10)
    parser.add_argument("--clean-high", type=float, default=0.90)
    parser.add_argument("--clean-mid", type=float, default=0.78)
    parser.add_argument("--roi-heavy-threshold", type=float, default=0.18)
    parser.add_argument("--lowres-fraction", type=float, default=0.04)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--bit-cost-per-mbit", type=float, default=0.12)
    parser.add_argument("--priority-weight", type=float, default=1.8)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--detail-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=831)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_visdrone_frames(args.visdrone_root, max_frames=args.max_frames)
    scheme_filter = {x.strip() for x in args.spectrum_schemes.split(",") if x.strip()}
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    spectrum_rows = load_spectrum_operating_points(args.spectrum_summary, args.ber, scheme_filter)

    rows = []
    for spectrum_row in spectrum_rows:
        for policy in policies:
            rows.append(simulate_policy(frames, spectrum_row, policy, args))

    csv_path = args.out_dir / "multimodal_decision_policy.csv"
    json_path = args.out_dir / "multimodal_decision_policy.json"
    md_path = args.out_dir / "multimodal_decision_policy.md"
    plot_path = args.out_dir / "multimodal_decision_policy_tradeoff.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "visual_dataset": "VisDrone2019-DET-val",
        "frames": len(frames),
        "spectrum_source": str(args.spectrum_summary),
        "ber": args.ber,
        "policies": policies,
        "state_definition": [
            "spectrum clean_rate, packet_loss, BER",
            "visual priority, ROI area, semantic/JPEG/ROI payload",
        ],
        "actions": ACTION_BITS,
        "utility": {
            "bit_cost_per_mbit": args.bit_cost_per_mbit,
            "priority_weight": args.priority_weight,
            "semantic_weight": args.semantic_weight,
            "detail_weight": args.detail_weight,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_tradeoff(rows, plot_path, "semantic_hard_rep3", args.plot_packet_loss)

    representative = [
        row
        for row in rows
        if row["spectrum_scheme"] in {"random", "semantic_hard_rep3", "spectrogram8_partial"}
        and abs(row["packet_loss"] - 0.2) < 1e-12
    ]
    sensitivity = [
        row
        for row in rows
        if row["spectrum_scheme"] == "semantic_hard_rep3"
        and row["policy"] in {"semantic_only", "spectrum_rule", "oracle_policy"}
    ]
    lines = [
        "# Multimodal Decision Policy Simulation",
        "",
        "## Setup",
        "",
        f"- Visual dataset: VisDrone2019-DET-val, {len(frames)} frames.",
        f"- Spectrum source: `{args.spectrum_summary}`",
        f"- BER: {args.ber:g}",
        f"- Policies: {', '.join(policies)}",
        "",
        "## Action space",
        "",
        "| Action | Meaning |",
        "|---|---|",
        "| summary_only | transmit only a compact state summary |",
        "| semantic_only | transmit visual semantic boxes only |",
        "| semantic_plus_roi | transmit visual semantics plus estimated ROI patches |",
        "| lowres_plus_semantic | transmit visual semantics plus low-resolution image |",
        "| jpeg_full | transmit full JPEG image |",
        "",
        "## Representative results at packet_loss=0.2",
        "",
        "| Spectrum | Policy | bits/frame | Utility | Semantic score | Priority detail | Fail rate | Main action ratios |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(representative, key=lambda r: (r["spectrum_scheme"], -r["mean_utility"])):
        actions = ", ".join(
            f"{action}:{row[f'action_{action}_ratio']:.2f}"
            for action in ACTION_BITS
            if row[f"action_{action}_ratio"] > 0.01
        )
        lines.append(
            f"| {row['spectrum_scheme']} | {row['policy']} | {row['mean_bits_per_frame']:.1f} | "
            f"{row['mean_utility']:.4f} | {row['semantic_delivery_score']:.4f} | "
            f"{row['priority_detail_score']:.4f} | {row['failed_transmission_rate']:.4f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Link-pressure sensitivity under semantic_hard_rep3 spectrum reporting",
            "",
            "| Packet loss | Policy | bits/frame | Utility | Priority detail | Fail rate | Action mix |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(sensitivity, key=lambda r: (r["packet_loss"], r["policy"])):
        actions = ", ".join(
            f"{action}:{row[f'action_{action}_ratio']:.2f}"
            for action in ACTION_BITS
            if row[f"action_{action}_ratio"] > 0.01
        )
        lines.append(
            f"| {row['packet_loss']:.2f} | {row['policy']} | {row['mean_bits_per_frame']:.1f} | "
            f"{row['mean_utility']:.4f} | {row['priority_detail_score']:.4f} | "
            f"{row['failed_transmission_rate']:.4f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This script formalizes the decision layer before reinforcement learning. "
            "The rule policy uses spectrum clean rate and visual priority to choose between semantic-only, ROI, low-resolution image, and full JPEG transmission. "
            "The sensitivity table shows that the rule policy transmits more image detail when the link is clean, but falls back to semantic-only as packet loss grows. "
            "The oracle policy is not deployable; it selects the action with the highest utility under the simulated channel and is used as an upper-bound reference.",
            "",
            f"Trade-off plot: `{plot_path}`",
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
