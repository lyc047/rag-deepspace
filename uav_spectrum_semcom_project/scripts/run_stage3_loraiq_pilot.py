"""Run a small real four-receiver LoRaIQ semantic-fusion experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.bit_budget import packet_header_bits
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.iq_replay import reference_rf_energy_j
from spectrum_semcom.loraiq import (
    channel_band_occupancy,
    excess_db_to_probability,
    load_cf32_sigmf,
    quantize_probability,
    welch_channel_excess_db,
)
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport, fuse_occupancy
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv


POLICIES = [
    "best_single",
    "full_mean",
    "full_snr_weighted",
    "full_robust_quality",
    "proposed_adaptive_value_bit",
]


def read_selected_rows(csv_path: Path, transmission_ids: list[int]) -> dict[int, list[dict[str, str]]]:
    selected: dict[int, list[dict[str, str]]] = defaultdict(list)
    wanted = set(int(value) for value in transmission_ids)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            transmission = int(row["transmission_idx"])
            if transmission in wanted:
                selected[transmission].append(row)
    if set(selected) != wanted:
        raise ValueError(f"missing transmissions: {sorted(wanted - set(selected))}")
    for transmission, rows in selected.items():
        rrhs = [int(row["rrh_idx"]) for row in rows]
        if sorted(rrhs) != [1, 2, 3, 4] or len(rows) != 4:
            raise ValueError(f"transmission {transmission} is not an exact four-RRH event")
        rows.sort(key=lambda row: int(row["rrh_idx"]))
    return dict(selected)


def annotation_band(metadata: dict[str, Any]) -> tuple[float, float]:
    annotations = metadata.get("annotations", [])
    if len(annotations) != 1:
        raise ValueError("pilot SigMF file must contain exactly one frame annotation")
    return float(annotations[0]["core:freq_lower_edge"]), float(annotations[0]["core:freq_upper_edge"])


def binary_metrics(estimates: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = np.asarray(estimates) >= float(threshold)
    targets = np.asarray(truth) >= 0.5
    tp = int(np.sum(predicted & targets))
    fp = int(np.sum(predicted & ~targets))
    fn = int(np.sum(~predicted & targets))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(np.mean(np.square(np.asarray(estimates) - np.asarray(truth)))),
    }


def resource_metrics(estimates: np.ndarray, truth: np.ndarray, clean_threshold: float) -> dict[str, float]:
    chosen = np.argmin(estimates, axis=1)
    oracle = np.argmin(truth, axis=1)
    rows = np.arange(estimates.shape[0])
    selected = truth[rows, chosen]
    optimal = truth[rows, oracle]
    return {
        "clean_resource_rate": float(np.mean(selected <= float(clean_threshold))),
        "mean_occupancy_regret": float(np.mean(selected - optimal)),
        "oracle_equivalent_rate": float(np.mean(np.abs(selected - optimal) <= 1e-9)),
    }


def link_reliability_score(ebn0_db: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(float(ebn0_db) - 3.0) / 2.0)))


def event_ranking(nodes: list[dict[str, Any]]) -> list[int]:
    """Value/bit ranking using observable SNR, link reliability, and fixed payload size."""

    ranked = []
    for node in nodes:
        sensing = 1.0 / (1.0 + np.exp(-float(node["snr_db"]) / 4.0))
        score = sensing * float(node["link_reliability"]) / max(float(node["application_bits"]), 1.0)
        ranked.append((score, int(node["node"])))
    return [node for _, node in sorted(ranked, reverse=True)]


def fuse_policy(
    policy: str,
    nodes: list[dict[str, Any]],
    ranking: list[int],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    adaptive = config["adaptive_policy"]
    by_node = {int(node["node"]): node for node in nodes}
    if policy == "best_single":
        attempted = ranking[:1]
        feedback_rounds = 0
    elif policy in {"full_mean", "full_snr_weighted", "full_robust_quality"}:
        attempted = list(ranking)
        feedback_rounds = 0
    elif policy == "proposed_adaptive_value_bit":
        attempted = list(ranking[: int(adaptive["initial_nodes"])])
        remaining = [node for node in ranking if node not in attempted]
        delivered = sum(by_node[node]["report"] is not None for node in attempted)
        feedback_rounds = 0
        while delivered < int(adaptive["target_delivered_reports"]) and remaining:
            candidate = remaining.pop(0)
            attempted.append(candidate)
            feedback_rounds += 1
            delivered += int(by_node[candidate]["report"] is not None)
    else:
        raise ValueError(f"unknown policy: {policy}")

    reports = [by_node[node]["report"] for node in attempted if by_node[node]["report"] is not None]
    width = int(config["detector"]["n_channels"])
    if not reports:
        estimate = np.zeros(width, dtype=np.float32)
    elif policy == "full_snr_weighted":
        estimate = fuse_occupancy(reports, "snr_weighted")
    elif policy in {"full_robust_quality", "proposed_adaptive_value_bit"} and len(reports) >= 3:
        estimate = fuse_occupancy(
            reports,
            str(adaptive["fusion_method"]),
            disagreement_scale=float(adaptive["disagreement_scale"]),
            max_weight_ratio=float(adaptive["max_weight_ratio"]),
        )
    else:
        estimate = fuse_occupancy(reports, "mean")

    initial_count = int(adaptive["initial_nodes"])
    initial = attempted[:initial_count] if policy == "proposed_adaptive_value_bit" else attempted
    latency = max([float(by_node[node]["duration_s"]) for node in initial] or [0.0])
    feedback_time = float(adaptive["scheduler_feedback_delay_s"]) + float(
        adaptive["ack_bits_per_attempted_node"]
    ) / float(adaptive["scheduler_feedback_bitrate_bps"])
    if policy == "proposed_adaptive_value_bit":
        for node in attempted[initial_count:]:
            latency += feedback_time + float(by_node[node]["duration_s"])
    feedback_bits = (
        float(adaptive["ack_bits_per_attempted_node"] * len(attempted))
        if policy == "proposed_adaptive_value_bit"
        else 0.0
    )
    return estimate, {
        "transmitted_bits": float(sum(by_node[node]["transmitted_bits"] for node in attempted)),
        "scheduler_feedback_bits": feedback_bits,
        "decision_link_latency_s": latency,
        "reference_rf_energy_j": float(sum(by_node[node]["reference_rf_energy_j"] for node in attempted)),
        "attempted_nodes": float(len(attempted)),
        "delivered_reports": float(len(reports)),
        "fallback_rounds": float(feedback_rounds),
    }


def aggregate_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "precision", "recall", "f1", "brier", "clean_resource_rate",
        "mean_occupancy_regret", "oracle_equivalent_rate", "mean_transmitted_bits",
        "mean_scheduler_feedback_bits", "mean_decision_link_latency_s",
        "mean_reference_rf_energy_j", "mean_attempted_nodes", "mean_delivered_reports",
        "mean_fallback_rounds",
    ]
    output = []
    for policy in POLICIES:
        items = [row for row in rows if row["policy"] == policy]
        merged: dict[str, Any] = {"policy": policy, "repeat_count": len(items)}
        for metric in metrics:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def make_report(
    summary: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
    config: dict[str, Any],
    application_bits: int,
) -> str:
    lines = [
        "# Stage 3 LoRaIQ Real Four-Receiver Pilot",
        "",
        "## Protocol",
        "",
        "Two UAV-transmitter LoRa frames are used: one LoS and one NLoS event. Each event has four aligned real RRH IQ recordings.",
        f"Each node sends one {application_bits}-bit application semantic containing four 8-bit occupancy probabilities and fixed metadata. The shared fair link adds header, CRC, FEC, padding, and retransmissions.",
        f"The off-frame Welch detector uses a fixed {config['detector']['excess_threshold_db']:.1f} dB excess threshold for this pilot.",
        "",
        "## Local receiver evidence",
        "",
        "| Transmission | Scene | RRH | annotated SNR | excess dB by channel | local F1 |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in local_rows:
        lines.append(
            f"| {row['transmission_idx']} | {row['area_type']} | {row['rrh_idx']} | {row['snr_db']:.2f} | "
            f"{row['excess_db']} | {row['local_f1']:.4f} |"
        )
    lines += [
        "",
        "## End-to-end results",
        "",
        "The confidence intervals below reflect reporting-link randomness only; two sensing events are not an independent statistical test set.",
        "",
        "| Policy | F1 | Brier | Clean | Regret | uplink bit | scheduler bit | latency | reports |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['f1_mean']:.4f} | {row['brier_mean']:.4f} | "
            f"{row['clean_resource_rate_mean']:.4f} | {row['mean_occupancy_regret_mean']:.6f} | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {row['mean_scheduler_feedback_bits_mean']:.1f} | "
            f"{1000.0 * row['mean_decision_link_latency_s_mean']:.2f} ms | {row['mean_delivered_reports_mean']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This run establishes that aligned real multi-receiver IQ can pass through local sensing, quantized semantic reporting, the fair digital link, fusion, and resource selection.",
        "The resource task is saturated because both LoRa frames occupy the same central half-band; clean rate and regret cannot distinguish the policies.",
        "The detector threshold was inspected on these same two pilot events, the event count is two, and the receiving nodes are fixed RRHs rather than UAVs. These results are diagnostic and do not establish H3.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_loraiq_pilot.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "loraiq_pilot")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    csv_path = PROJECT_DIR / config["dataset_csv"]
    iq_root = PROJECT_DIR / config["iq_root"]
    selected = read_selected_rows(csv_path, [int(value) for value in config["transmission_ids"]])
    detector = config["detector"]
    semantic = config["semantic_payload"]
    n_channels = int(detector["n_channels"])
    quantization_bits = int(semantic["probability_bits_per_channel"])
    application_bits = packet_header_bits(metadata_bits=int(semantic["metadata_bits"])) + n_channels * quantization_bits

    events: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for transmission in [int(value) for value in config["transmission_ids"]]:
        nodes = []
        truth_candidates = []
        for row in selected[transmission]:
            relative = str(row["sigmf_file"]).removeprefix("./")
            data_path = iq_root / f"{relative}.sigmf-data"
            meta_path = iq_root / f"{relative}.sigmf-meta"
            iq, metadata = load_cf32_sigmf(data_path, meta_path)
            source_files.extend([data_path, meta_path])
            sample_rate = float(row["rx_sample_rate"])
            excess = welch_channel_excess_db(
                iq,
                int(float(row["sigmf_file_offset"])),
                int(row["sigmf_file_n_samples"]),
                sample_rate,
                n_channels,
                int(detector["nperseg"]),
                int(detector["guard_samples"]),
            )
            probabilities = excess_db_to_probability(
                excess,
                float(detector["excess_threshold_db"]),
                float(detector["temperature_db"]),
            )
            low, high = annotation_band(metadata)
            truth_candidates.append(channel_band_occupancy(low, high, sample_rate, n_channels))
            rrh = int(row["rrh_idx"])
            snr = float(row["snr"])
            confidence = float(np.mean(2.0 * np.abs(probabilities - 0.5)))
            local = binary_metrics(probabilities, truth_candidates[-1], float(detector["binary_threshold"]))
            local_rows.append({
                "transmission_idx": transmission,
                "area_type": row["area_type"],
                "rrh_idx": rrh,
                "snr_db": snr,
                "excess_db": "[" + ", ".join(f"{value:.2f}" for value in excess) + "]",
                "local_f1": local["f1"],
            })
            nodes.append({
                "node": rrh - 1,
                "rrh_idx": rrh,
                "snr_db": snr,
                "confidence": confidence,
                "probabilities": probabilities,
                "application_bits": application_bits,
                "ebn0_db": float(config["reporting_ebn0_db_per_rrh"][rrh - 1]),
                "link_reliability": link_reliability_score(config["reporting_ebn0_db_per_rrh"][rrh - 1]),
            })
        events.append({
            "transmission_idx": transmission,
            "area_type": selected[transmission][0]["area_type"],
            "truth": np.mean(np.stack(truth_candidates), axis=0).astype(np.float32),
            "nodes": nodes,
        })

    trial_rows = []
    for repeat in range(int(config["repeat_count"])):
        estimates = {policy: [] for policy in POLICIES}
        costs = {policy: [] for policy in POLICIES}
        truths = []
        for event_index, event in enumerate(events):
            reporting_nodes = []
            for node in event["nodes"]:
                link = build_link_config(stage2, float(node["ebn0_db"]))
                result = transmit_payload_analytic(
                    application_bits,
                    link,
                    int(config["seed"]) + repeat * 100_000 + event_index * 100 + int(node["node"]),
                )
                report = None
                if result.frame_success:
                    report = NodeOccupancyReport(
                        quantize_probability(node["probabilities"], quantization_bits),
                        float(node["snr_db"]),
                        float(node["confidence"]),
                        float(node["link_reliability"]),
                        0.0,
                    )
                fixed_delay = float(link.propagation_delay_s + link.per_attempt_processing_s)
                reporting_nodes.append({
                    **node,
                    "report": report,
                    "transmitted_bits": float(result.transmitted_bits),
                    "duration_s": float(result.duration_s),
                    "reference_rf_energy_j": reference_rf_energy_j(
                        result.transmitted_bits,
                        result.attempts,
                        result.duration_s,
                        fixed_delay,
                        1.0,
                    ),
                })
            ranking = event_ranking(reporting_nodes)
            truths.append(event["truth"])
            for policy in POLICIES:
                estimate, cost = fuse_policy(policy, reporting_nodes, ranking, config)
                estimates[policy].append(estimate)
                costs[policy].append(cost)
        truth_array = np.stack(truths)
        for policy in POLICIES:
            estimate_array = np.stack(estimates[policy])
            sensing = binary_metrics(estimate_array, truth_array, float(detector["binary_threshold"]))
            resource = resource_metrics(
                estimate_array,
                truth_array,
                float(config["resource_task"]["clean_threshold"]),
            )
            trial_rows.append({
                "policy": policy,
                "repeat": repeat,
                **sensing,
                **resource,
                **{
                    f"mean_{key}": float(np.mean([event_cost[key] for event_cost in costs[policy]]))
                    for key in costs[policy][0]
                },
            })

    summary = aggregate_trials(trial_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "loraiq_local_sensing.csv", local_rows)
    write_csv(args.out_dir / "loraiq_trials.csv", trial_rows)
    write_csv(args.out_dir / "loraiq_summary.csv", summary)
    report = make_report(summary, local_rows, config, application_bits)
    (args.out_dir / "stage3_loraiq_pilot_report.md").write_text(report, encoding="utf-8")
    result = {
        "config_sha256": sha256_file(args.config),
        "dataset_csv_sha256": sha256_file(csv_path),
        "source_file_ids_sha256": sha256_strings([str(path.relative_to(PROJECT_DIR)) for path in source_files]),
        "source_files": [
            {"path": str(path.relative_to(PROJECT_DIR)), "sha256": sha256_file(path)} for path in source_files
        ],
        "transmission_ids": config["transmission_ids"],
        "event_count": len(events),
        "application_bits_per_report": application_bits,
        "summary": summary,
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": config["claim_boundary"],
    }
    (args.out_dir / "stage3_loraiq_pilot_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "stage3_loraiq_pilot_report.md")


if __name__ == "__main__":
    main()
