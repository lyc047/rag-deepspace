from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.visdrone import VisDroneFrame, iter_visdrone_frames
from spectrum_semcom.reproducibility import assert_disjoint_groups, grouped_split
from simulate_multimodal_decision_policy import (
    ACTION_BITS,
    action_scores,
    choose_oracle_action,
    load_spectrum_operating_points,
    policy_action,
)
from simulate_multimodal_policy import estimate_roi_bits, roi_area_fraction, visual_priority


ACTIONS = list(ACTION_BITS.keys())


@dataclass(frozen=True)
class DecisionSample:
    frame: VisDroneFrame
    spectrum_row: dict
    state: np.ndarray


class DQNPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_default_decision_args(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse the same utility and link parameters as the formal rule-based
    # decision-layer script so DQN, rule, and oracle are compared fairly.
    return argparse.Namespace(
        packet_bits=args.packet_bits,
        lowres_fraction=args.lowres_fraction,
        bit_cost_per_mbit=args.bit_cost_per_mbit,
        priority_weight=args.priority_weight,
        detail_weight=args.detail_weight,
        semantic_weight=args.semantic_weight,
        min_priority_objects=args.min_priority_objects,
        clean_high=args.clean_high,
        clean_mid=args.clean_mid,
        roi_heavy_threshold=args.roi_heavy_threshold,
        seed=args.seed,
    )


def state_features(frame: VisDroneFrame, spectrum_row: dict, args: argparse.Namespace) -> np.ndarray:
    clean_rate = float(spectrum_row["clean_channel_rate"])
    packet_loss = float(spectrum_row["packet_loss"])
    ber = float(spectrum_row["ber"])
    priority_count = sum(1 for box in frame.boxes if box.category in {4, 5, 6, 9})
    box_count = len(frame.boxes)
    semantic_mbit = frame.semantic_box_bits / 1e6
    roi_mbit = estimate_roi_bits(frame) / 1e6
    lowres_mbit = frame.jpeg_bits * args.lowres_fraction / 1e6
    jpeg_mbit = frame.jpeg_bits / 1e6
    return np.asarray(
        [
            clean_rate,
            packet_loss,
            min(1.0, -np.log10(max(ber, 1e-12)) / 8.0),
            1.0 if visual_priority(frame, args.min_priority_objects) else 0.0,
            min(1.0, priority_count / 80.0),
            min(1.0, box_count / 120.0),
            min(1.0, roi_area_fraction(frame)),
            min(1.0, semantic_mbit / 0.02),
            min(1.0, roi_mbit / 0.2),
            min(1.0, lowres_mbit / 0.08),
            min(1.0, jpeg_mbit / 2.5),
        ],
        dtype=np.float32,
    )


def build_samples(frames: list[VisDroneFrame], spectrum_rows: list[dict], args: argparse.Namespace) -> list[DecisionSample]:
    samples: list[DecisionSample] = []
    for spectrum_row in spectrum_rows:
        for frame in frames:
            samples.append(DecisionSample(frame=frame, spectrum_row=spectrum_row, state=state_features(frame, spectrum_row, args)))
    return samples


def reward_for_action(sample: DecisionSample, action: str, decision_args: argparse.Namespace) -> float:
    row = sample.spectrum_row
    return float(
        action_scores(
            sample.frame,
            action,
            float(row["clean_channel_rate"]),
            float(row["packet_loss"]),
            float(row["ber"]),
            decision_args.packet_bits,
            decision_args.lowres_fraction,
            decision_args.bit_cost_per_mbit,
            decision_args.priority_weight,
            decision_args.detail_weight,
            decision_args.semantic_weight,
        )["utility"]
    )


def visdrone_sequence_id(frame: VisDroneFrame) -> str:
    """Return the acquisition-sequence prefix used to prevent adjacent-frame leakage."""
    return frame.stem.split("_", 1)[0]


def split_samples(
    samples: list[DecisionSample],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[DecisionSample], list[DecisionSample], list[DecisionSample]]:
    """Split source sequences before treating channel variants as independent samples.

    Every channel-condition expansion of a frame, and every adjacent frame with
    the same VisDrone sequence prefix, remains in exactly one source split.
    """
    splits = grouped_split(
        samples,
        group_key=lambda sample: visdrone_sequence_id(sample.frame),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    assert_disjoint_groups(splits, group_key=lambda sample: visdrone_sequence_id(sample.frame))
    return splits["train"], splits["val"], splits["test"]


def describe_source_split(samples: list[DecisionSample]) -> dict[str, object]:
    frame_ids = sorted({sample.frame.stem for sample in samples})
    sequence_ids = sorted({visdrone_sequence_id(sample.frame) for sample in samples})
    return {
        "samples": len(samples),
        "unique_frames": len(frame_ids),
        "sequence_groups": len(sequence_ids),
        "frame_ids": frame_ids,
        "sequence_ids": sequence_ids,
    }


def train_dqn(
    train_samples: list[DecisionSample],
    val_samples: list[DecisionSample],
    decision_args: argparse.Namespace,
    args: argparse.Namespace,
) -> DQNPolicyNet:
    device = torch.device(args.device)
    state_dim = len(train_samples[0].state)
    model = DQNPolicyNet(state_dim, len(ACTIONS), args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay: deque[tuple[np.ndarray, int, float]] = deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)

    def greedy_action(sample: DecisionSample, epsilon: float) -> int:
        if rng.random() < epsilon:
            return rng.randrange(len(ACTIONS))
        with torch.no_grad():
            x = torch.tensor(sample.state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(model(x), dim=1).item())

    for step in range(1, args.steps + 1):
        epsilon = max(args.epsilon_final, args.epsilon_start - (args.epsilon_start - args.epsilon_final) * step / max(1, args.epsilon_decay_steps))
        sample = train_samples[rng.randrange(len(train_samples))]
        action_idx = greedy_action(sample, epsilon)
        reward = reward_for_action(sample, ACTIONS[action_idx], decision_args)
        replay.append((sample.state, action_idx, reward))

        if len(replay) < args.batch_size:
            continue
        batch = rng.sample(list(replay), args.batch_size)
        states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
        q_values = model(states).gather(1, actions[:, None]).squeeze(1)
        # One-step contextual decision: terminal target is the immediate utility.
        loss = F.smooth_l1_loss(q_values, rewards)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        if args.verbose and step % args.log_every == 0:
            val_rows = evaluate_policy("dqn", val_samples, decision_args, args, model)
            print(f"step={step} loss={loss.item():.4f} val_utility={val_rows['mean_utility']:.4f}")

    return model


def select_dqn_action(model: DQNPolicyNet, sample: DecisionSample, device: str) -> str:
    with torch.no_grad():
        x = torch.tensor(sample.state, dtype=torch.float32, device=torch.device(device)).unsqueeze(0)
        idx = int(torch.argmax(model(x), dim=1).item())
    return ACTIONS[idx]


def evaluate_policy(
    policy: str,
    samples: list[DecisionSample],
    decision_args: argparse.Namespace,
    args: argparse.Namespace,
    model: DQNPolicyNet | None = None,
) -> dict[str, float | str]:
    action_counts = {action: 0 for action in ACTIONS}
    bits: list[float] = []
    utilities: list[float] = []
    semantic_scores: list[float] = []
    detail_scores: list[float] = []
    priority_detail_scores: list[float] = []
    high_priority_count = 0

    for sample in samples:
        row = sample.spectrum_row
        clean_rate = float(row["clean_channel_rate"])
        packet_loss = float(row["packet_loss"])
        ber = float(row["ber"])
        if policy == "dqn":
            if model is None:
                raise ValueError("model is required for DQN evaluation")
            action = select_dqn_action(model, sample, args.device)
        elif policy == "oracle_policy":
            action = choose_oracle_action(sample.frame, clean_rate, packet_loss, ber, decision_args)
        else:
            action = policy_action(policy, sample.frame, clean_rate, packet_loss, ber, decision_args)
        action_counts[action] += 1
        score = action_scores(
            sample.frame,
            action,
            clean_rate,
            packet_loss,
            ber,
            decision_args.packet_bits,
            decision_args.lowres_fraction,
            decision_args.bit_cost_per_mbit,
            decision_args.priority_weight,
            decision_args.detail_weight,
            decision_args.semantic_weight,
        )
        bits.append(float(score["bits"]))
        utilities.append(float(score["utility"]))
        semantic_scores.append(float(score["semantic_delivered_probability"]))
        detail_scores.append(float(score["detail_delivered_probability"]))
        is_high = visual_priority(sample.frame, decision_args.min_priority_objects)
        if is_high:
            high_priority_count += 1
            priority_detail_scores.append(float(score["detail_delivered_probability"]))

    mean_bits = float(np.mean(bits))
    return {
        "policy": policy,
        "samples": len(samples),
        "mean_bits_per_frame": mean_bits,
        "mean_utility": float(np.mean(utilities)),
        "semantic_delivery_score": float(np.mean(semantic_scores)),
        "detail_delivery_score": float(np.mean(detail_scores)),
        "priority_detail_score": float(np.mean(priority_detail_scores)) if high_priority_count else 0.0,
        "detail_per_mbit": float(np.mean(detail_scores)) / max(mean_bits / 1e6, 1e-9),
        **{f"action_{action}_ratio": action_counts[action] / len(samples) for action in ACTIONS},
    }


def evaluate_by_packet_loss(
    samples: list[DecisionSample],
    policies: list[str],
    decision_args: argparse.Namespace,
    args: argparse.Namespace,
    model: DQNPolicyNet,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    packet_losses = sorted({float(sample.spectrum_row["packet_loss"]) for sample in samples})
    schemes = sorted({sample.spectrum_row["scheme"] for sample in samples})
    for scheme in schemes:
        for packet_loss in packet_losses:
            group = [
                sample
                for sample in samples
                if sample.spectrum_row["scheme"] == scheme and abs(float(sample.spectrum_row["packet_loss"]) - packet_loss) < 1e-12
            ]
            if not group:
                continue
            for policy in policies:
                row = evaluate_policy(policy, group, decision_args, args, model if policy == "dqn" else None)
                row["spectrum_scheme"] = scheme
                row["packet_loss"] = packet_loss
                row["ber"] = float(group[0].spectrum_row["ber"])
                row["clean_rate"] = float(group[0].spectrum_row["clean_channel_rate"])
                rows.append(row)
    return rows


def write_report(rows: list[dict[str, float | str]], out_path: Path, args: argparse.Namespace) -> None:
    representative = [
        row
        for row in rows
        if row["spectrum_scheme"] in {"random", "semantic_hard_rep3", "spectrogram8_partial"}
        and abs(float(row["packet_loss"]) - args.report_packet_loss) < 1e-12
    ]
    sensitivity = [
        row
        for row in rows
        if row["spectrum_scheme"] == "semantic_hard_rep3"
        and row["policy"] in {"semantic_only", "spectrum_rule", "dqn", "oracle_policy"}
    ]
    lines = [
        "# DQN Multimodal Semantic Transmission Policy",
        "",
        "## Purpose",
        "",
        "This experiment upgrades the previous rule-based decision layer to a lightweight trainable DQN/contextual-bandit policy. "
        "The state combines spectrum quality and visual semantic demand; the action selects how much visual information to transmit.",
        "",
        "## State and action design",
        "",
        "- State: clean-channel rate, packet loss, BER, high-priority visual flag, priority-object density, object density, ROI area, and normalized payload sizes.",
        "- Actions: summary_only, semantic_only, semantic_plus_roi, lowres_plus_semantic, jpeg_full.",
        "- Reward: the same task utility used by the formal decision-layer simulator, so DQN is directly comparable with spectrum_rule and oracle_policy.",
        "",
        "## Representative test results",
        "",
        f"Packet loss shown below: {args.report_packet_loss:g}.",
        "",
        "| Spectrum | Policy | bits/frame | Utility | Semantic score | Priority detail | Detail/Mbit | Action mix |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(representative, key=lambda r: (r["spectrum_scheme"], -float(r["mean_utility"]))):
        actions = ", ".join(
            f"{action}:{float(row[f'action_{action}_ratio']):.2f}"
            for action in ACTIONS
            if float(row[f"action_{action}_ratio"]) > 0.01
        )
        lines.append(
            f"| {row['spectrum_scheme']} | {row['policy']} | {float(row['mean_bits_per_frame']):.1f} | "
            f"{float(row['mean_utility']):.4f} | {float(row['semantic_delivery_score']):.4f} | "
            f"{float(row['priority_detail_score']):.4f} | {float(row['detail_per_mbit']):.2f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Link-pressure sensitivity under semantic_hard_rep3",
            "",
            "| Packet loss | Policy | bits/frame | Utility | Priority detail | Action mix |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(sensitivity, key=lambda r: (float(r["packet_loss"]), str(r["policy"]))):
        actions = ", ".join(
            f"{action}:{float(row[f'action_{action}_ratio']):.2f}"
            for action in ACTIONS
            if float(row[f"action_{action}_ratio"]) > 0.01
        )
        lines.append(
            f"| {float(row['packet_loss']):.2f} | {row['policy']} | {float(row['mean_bits_per_frame']):.1f} | "
            f"{float(row['mean_utility']):.4f} | {float(row['priority_detail_score']):.4f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "DQN is not used here to replace the spectrum detector. It optimizes the cross-layer decision: whether the UAV should send only visual semantics, add ROI/detail, or fall back to a compact summary under poor links. "
            "If DQN approaches the oracle and exceeds spectrum_rule on utility at similar or lower payload, it supports the next research claim: semantic communication should jointly decide content granularity and radio resource state, not only compress images or classify spectrum.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight DQN policy for spectrum-aware UAV visual semantic transmission.")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--spectrum-summary", type=Path, default=PROJECT_DIR / "results" / "phase1" / "resource_optimization_simulation.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--spectrum-schemes", type=str, default="random,semantic_hard_rep3,spectrogram8_partial,oracle")
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
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=4500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=944)
    parser.add_argument("--report-packet-loss", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = iter_visdrone_frames(args.visdrone_root, max_frames=args.max_frames)
    scheme_filter = {x.strip() for x in args.spectrum_schemes.split(",") if x.strip()}
    spectrum_rows = load_spectrum_operating_points(args.spectrum_summary, args.ber, scheme_filter)
    decision_args = make_default_decision_args(args)

    samples = build_samples(frames, spectrum_rows, args)
    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    model = train_dqn(train_samples, val_samples, decision_args, args)

    policies = ["summary_only", "semantic_only", "spectrum_rule", "dqn", "oracle_policy"]
    rows = evaluate_by_packet_loss(test_samples, policies, decision_args, args, model)

    csv_path = args.out_dir / "multimodal_dqn_policy.csv"
    json_path = args.out_dir / "multimodal_dqn_policy.json"
    md_path = args.out_dir / "multimodal_dqn_policy.md"
    model_path = args.out_dir / "multimodal_dqn_policy.pt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "visual_dataset": "VisDrone2019-DET-val",
        "frames": len(frames),
        "samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "split_policy": "VisDrone acquisition-sequence prefix split before channel-condition evaluation",
        "source_splits": {
            "train": describe_source_split(train_samples),
            "val": describe_source_split(val_samples),
            "test": describe_source_split(test_samples),
        },
        "spectrum_source": str(args.spectrum_summary),
        "actions": ACTIONS,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "actions": ACTIONS, "args": vars(args)}, model_path)
    write_report(rows, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
