"""Same-source real SigMF I/Q multi-node replay with adaptive semantic reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.baselines import connected_component_energy_detector, enhance_stft_for_detection
from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.frame_manifest import load_frame_manifest, load_iq_frame_from_manifest_row
from spectrum_semcom.iq_replay import ReplayChannel, frequency_channel_occupancy, reference_rf_energy_j, replay_iq_view
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport, fuse_occupancy
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv


def filter_truth(boxes):
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def evaluate_resource(estimates: np.ndarray, truths: np.ndarray, task: dict[str, Any]) -> dict[str, float]:
    demand = int(task["demand_channels"])
    estimated_blocks = np.stack([
        estimates[:, start : start + demand].mean(axis=1)
        for start in range(estimates.shape[1] - demand + 1)
    ], axis=1)
    truth_blocks = np.stack([
        truths[:, start : start + demand].mean(axis=1)
        for start in range(truths.shape[1] - demand + 1)
    ], axis=1)
    chosen = np.argmin(estimated_blocks, axis=1)
    oracle = np.argmin(truth_blocks, axis=1)
    rows = np.arange(len(chosen))
    selected = truth_blocks[rows, chosen]
    optimal = truth_blocks[rows, oracle]
    return {
        "clean_resource_rate": float(np.mean(selected <= float(task["clean_threshold"]))),
        "mean_occupancy_regret": float(np.mean(selected - optimal)),
        "oracle_equivalent_rate": float(np.mean(np.abs(selected - optimal) <= 1e-9)),
    }


def node_rank(config: dict[str, Any]) -> list[int]:
    scores = []
    for node, (settings, ebn0) in enumerate(zip(config["sensing_nodes"], config["reporting_ebn0_db_per_node"])):
        sensing = 1.0 / (1.0 + np.exp(-float(settings["sensing_snr_db"]) / 4.0))
        reliability = 1.0 / (1.0 + np.exp(-(float(ebn0) - 3.0) / 2.0))
        scores.append((float(sensing * reliability), node))
    return [node for _, node in sorted(scores, reverse=True)]


def policy_result(
    records: dict[int, dict[str, Any]],
    ranking: list[int],
    policy: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    adaptive = config["adaptive_policy"]
    if policy == "full_mean" or policy == "full_robust":
        attempted = list(ranking)
        feedback_rounds = 0
    elif policy == "fixed_two_mean":
        attempted = list(ranking[: int(adaptive["initial_nodes"])])
        feedback_rounds = 0
    elif policy == "adaptive_ack_robust":
        attempted = list(ranking[: int(adaptive["initial_nodes"])])
        delivered = sum(records[node]["report"] is not None for node in attempted)
        remaining = [node for node in ranking if node not in attempted]
        feedback_rounds = 0
        while delivered < int(adaptive["target_delivered_reports"]) and remaining:
            attempted.append(remaining.pop(0))
            feedback_rounds += 1
            delivered += int(records[attempted[-1]]["report"] is not None)
    else:
        raise ValueError(f"unknown policy: {policy}")

    reports = [records[node]["report"] for node in attempted if records[node]["report"] is not None]
    width = int(config["resource_task"]["n_channels"])
    if not reports:
        estimate = np.zeros(width, dtype=np.float32)
    elif policy in {"full_robust", "adaptive_ack_robust"} and len(reports) >= 3:
        estimate = fuse_occupancy(
            reports,
            str(adaptive["fusion_method"]),
            disagreement_scale=float(adaptive["disagreement_scale"]),
            max_weight_ratio=float(adaptive["max_weight_ratio"]),
        )
    else:
        estimate = fuse_occupancy(reports, "mean")

    initial = attempted[: int(adaptive["initial_nodes"])] if policy == "adaptive_ack_robust" else attempted
    link_latency = max([float(records[node]["duration_s"]) for node in initial] or [0.0])
    feedback_time = float(adaptive["scheduler_feedback_delay_s"]) + float(adaptive["ack_bits_per_attempted_node"]) / float(adaptive["scheduler_feedback_bitrate_bps"])
    if policy == "adaptive_ack_robust":
        for node in attempted[int(adaptive["initial_nodes"]) :]:
            link_latency += feedback_time + float(records[node]["duration_s"])
    control_bits = float(adaptive["ack_bits_per_attempted_node"] * len(attempted)) if policy == "adaptive_ack_robust" else 0.0
    return estimate, {
        "transmitted_bits": float(sum(records[node]["transmitted_bits"] for node in attempted)),
        "scheduler_feedback_bits": control_bits,
        "decision_link_latency_s": link_latency,
        "reference_rf_energy_j": float(sum(records[node]["reference_rf_energy_j"] for node in attempted)),
        "attempted_nodes": float(len(attempted)),
        "delivered_reports": float(len(reports)),
        "fallback_rounds": float(feedback_rounds),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "clean_resource_rate", "mean_occupancy_regret", "oracle_equivalent_rate",
        "mean_transmitted_bits", "mean_scheduler_feedback_bits", "mean_decision_link_latency_s",
        "mean_reference_rf_energy_j", "mean_attempted_nodes", "mean_delivered_reports", "mean_fallback_rounds",
    ]
    output = []
    for policy in sorted({row["policy"] for row in rows}):
        items = [row for row in rows if row["policy"] == policy]
        merged: dict[str, Any] = {"policy": policy, "repeat_count": len(items)}
        for metric in metrics:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def make_report(summary, node_detection, config, frame_count: int) -> str:
    lines = [
        "# Stage 3 Same-Source SigMF I/Q Record Replay",
        "",
        f"One real 5G NR SigMF recording is sliced into {frame_count} correlated frames and replayed through four independent controlled sensing channels.",
        "This validates the raw-I/Q-to-STFT-to-semantic-report pipeline. It is not independent multi-receiver capture and cannot establish spatial diversity.",
        "",
        "## Frozen detector and node sensing",
        "",
        f"Detector: sigma={config['frozen_detector']['threshold_sigma']}, min_cells={config['frozen_detector']['min_cells']}, enhancement={config['frozen_detector']['enhancement']}.",
        "",
        "| Node | Sensing SNR | Detection F1 |",
        "|---:|---:|---:|",
    ]
    for row in node_detection:
        lines.append(f"| {row['node']} | {row['sensing_snr_db']:.1f} dB | {row['f1']:.4f} |")
    lines += [
        "",
        "## Task and communication results",
        "",
        "Energy is a 1 W reference RF-airtime estimate, not measured hardware energy.",
        "",
        "| Policy | Clean | Regret | Oracle-equivalent | uplink bit | scheduler bit | link latency | RF energy | nodes | fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['clean_resource_rate_mean']:.4f} | {row['mean_occupancy_regret_mean']:.6f} | "
            f"{row['oracle_equivalent_rate_mean']:.4f} | {row['mean_transmitted_bits_mean']:.1f} | "
            f"{row['mean_scheduler_feedback_bits_mean']:.1f} | {1000*row['mean_decision_link_latency_s_mean']:.2f} ms | "
            f"{1000*row['mean_reference_rf_energy_j_mean']:.3f} mJ | {row['mean_attempted_nodes_mean']:.2f} | "
            f"{row['mean_fallback_rounds_mean']:.2f} |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "All frames originate from one short recording. Detector parameters were previously selected on the same recording, so detection and task numbers are diagnostic only.",
        "The contribution of this run is same-source raw-I/Q replay and explicit fallback latency/energy accounting, not a generalization or real-flight claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_iq_record_replay.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "frame_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "iq_record_replay")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    manifest = load_frame_manifest(args.manifest, PROJECT_DIR)
    frames = [load_iq_frame_from_manifest_row(row) for row in manifest]
    ranking = node_rank(config)
    n_nodes = len(config["sensing_nodes"])
    task = config["resource_task"]
    policies = ["full_mean", "full_robust", "fixed_two_mean", "adaptive_ack_robust"]
    trial_rows = []
    detection_counts = {node: {"tp": 0, "fp": 0, "fn": 0} for node in range(n_nodes)}

    truth_channels = np.stack([
        frequency_channel_occupancy(filter_truth(frame.boxes), frame.sample_rate_hz, frame.duration_s, int(task["n_channels"]))
        for frame in frames
    ])
    for repeat in range(int(config["repeat_count"])):
        policy_estimates = {policy: [] for policy in policies}
        policy_costs = {policy: [] for policy in policies}
        for frame_index, frame in enumerate(frames):
            records: dict[int, dict[str, Any]] = {}
            truth = filter_truth(frame.boxes)
            for node, settings in enumerate(config["sensing_nodes"]):
                taps = tuple(complex(float(real), float(imag)) for real, imag in settings["multipath_taps"])
                channel = ReplayChannel(
                    float(settings["sensing_snr_db"]), float(settings["gain_db"]),
                    float(settings["frequency_offset_hz"]), taps,
                )
                rng = np.random.default_rng(int(config["seed"]) + repeat * 1_000_000 + frame_index * 100 + node)
                view = replay_iq_view(frame.iq, frame.sample_rate_hz, channel, rng)
                stft = stft_power(view, frame.sample_rate_hz, int(config["stft"]["n_fft"]), int(config["stft"]["hop_length"]))
                enhanced = enhance_stft_for_detection(stft.power_db, str(config["frozen_detector"]["enhancement"]))
                boxes = connected_component_energy_detector(
                    stft,
                    threshold_sigma=float(config["frozen_detector"]["threshold_sigma"]),
                    enhanced_power=enhanced,
                    min_cells=int(config["frozen_detector"]["min_cells"]),
                    max_boxes=128,
                )
                metrics = evaluate_detections(boxes, truth, iou_threshold=0.05)
                if repeat == 0:
                    detection_counts[node]["tp"] += int(metrics.true_positive)
                    detection_counts[node]["fp"] += int(metrics.false_positive)
                    detection_counts[node]["fn"] += int(metrics.false_negative)
                occupancy = frequency_channel_occupancy(boxes, frame.sample_rate_hz, frame.duration_s, int(task["n_channels"]))
                ebn0 = float(config["reporting_ebn0_db_per_node"][node])
                link = build_link_config(stage2, ebn0)
                result = transmit_payload_analytic(
                    hard_box_packet_bits(len(boxes)), link,
                    int(config["seed"]) + 50_000_000 + repeat * 1_000_000 + frame_index * 100 + node,
                )
                confidence = float(np.mean([box.confidence for box in boxes])) if boxes else 0.5
                reliability = float(1.0 / (1.0 + np.exp(-(ebn0 - 3.0) / 2.0)))
                report = NodeOccupancyReport(occupancy, float(settings["sensing_snr_db"]), confidence, reliability, 0.0) if result.frame_success else None
                fixed_delay = float(link.propagation_delay_s + link.per_attempt_processing_s)
                energy = reference_rf_energy_j(
                    result.transmitted_bits, result.attempts, result.duration_s, fixed_delay,
                    float(config["energy_model"]["transmit_power_w_per_node"][node]),
                )
                records[node] = {
                    "report": report,
                    "transmitted_bits": float(result.transmitted_bits),
                    "duration_s": float(result.duration_s),
                    "reference_rf_energy_j": energy,
                }
            for policy in policies:
                estimate, cost = policy_result(records, ranking, policy, config)
                policy_estimates[policy].append(estimate)
                policy_costs[policy].append(cost)
        for policy in policies:
            costs = policy_costs[policy]
            trial_rows.append({
                "policy": policy,
                "repeat": repeat,
                **evaluate_resource(np.stack(policy_estimates[policy]), truth_channels, task),
                **{f"mean_{key}": float(np.mean([row[key] for row in costs])) for key in costs[0]},
            })

    node_detection = []
    for node, counts in detection_counts.items():
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        node_detection.append({"node": node, "sensing_snr_db": float(config["sensing_nodes"][node]["sensing_snr_db"]), "f1": float(f1), **counts})
    summary = aggregate(trial_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "iq_replay_trials.csv", trial_rows)
    write_csv(args.out_dir / "iq_replay_summary.csv", summary)
    write_csv(args.out_dir / "node_detection.csv", node_detection)
    (args.out_dir / "stage3_iq_record_replay_report.md").write_text(make_report(summary, node_detection, config, len(frames)), encoding="utf-8")
    result = {
        "config_sha256": sha256_file(args.config),
        "manifest_sha256": sha256_file(args.manifest),
        "frame_ids_sha256": sha256_strings([frame.frame_id for frame in frames]),
        "source_recording_count": 1,
        "node_ranking": ranking,
        "node_detection": node_detection,
        "summary": summary,
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": config["claim_boundary"],
    }
    (args.out_dir / "stage3_iq_record_replay_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_iq_record_replay_report.md")


if __name__ == "__main__":
    main()
