"""Analyze only the frozen AADM development flights for real-link fault structure."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
LORA_PREFIX = "AADM2025Dryad/LORA"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def runs_of_ones(sequence: list[int]) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in sequence:
        if value == 1:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def run_length_encode(sequence: list[int]) -> list[list[int]]:
    encoded: list[list[int]] = []
    for value in sequence:
        if encoded and encoded[-1][0] == value:
            encoded[-1][1] += 1
        else:
            encoded.append([value, 1])
    return encoded


def lag1_correlation(sequence: list[int]) -> float | None:
    if len(sequence) < 3:
        return None
    left = sequence[:-1]
    right = sequence[1:]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator_left = sum((a - left_mean) ** 2 for a in left)
    denominator_right = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(denominator_left * denominator_right)
    return numerator / denominator if denominator > 0 else None


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_optional_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_lora_flight(archive: ZipFile, flight_id: str) -> dict[str, Any]:
    member = f"{LORA_PREFIX}/{flight_id}/LORAlog.csv"
    rows = csv.DictReader(
        io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="")
    )
    packets: dict[str, dict[str, Any]] = {}
    malformed_rows = 0
    all_gateways: set[str] = set()
    row_count = 0
    for row in rows:
        row_count += 1
        try:
            frame_counter = int(row["f_cnt"])
            timestamp = parse_time(row["rx_info_time"])
        except (KeyError, TypeError, ValueError):
            malformed_rows += 1
            continue
        packet_id = row.get("deduplication_id", "").strip()
        if not packet_id:
            packet_id = f"{frame_counter}|{row.get('time', '')}"
        gateway = row.get("rx_info_gatewayId", "").strip()
        if gateway:
            all_gateways.add(gateway)
        packet = packets.setdefault(
            packet_id,
            {
                "frame_counter": frame_counter,
                "timestamp": timestamp,
                "gateways": set(),
                "snr": [],
                "rssi": [],
            },
        )
        if packet["frame_counter"] != frame_counter:
            raise ValueError(f"{flight_id}: one packet ID has multiple frame counters")
        packet["timestamp"] = min(packet["timestamp"], timestamp)
        if gateway:
            packet["gateways"].add(gateway)
        snr = parse_optional_float(row.get("rx_info_snr", ""))
        rssi = parse_optional_float(row.get("rx_info_rssi", ""))
        if snr is not None:
            packet["snr"].append(snr)
        if rssi is not None:
            packet["rssi"].append(rssi)

    ordered_packets = sorted(packets.values(), key=lambda item: item["timestamp"])
    segments: list[list[dict[str, Any]]] = []
    restart_events = 0
    for packet in ordered_packets:
        if segments and packet["frame_counter"] < segments[-1][-1]["frame_counter"]:
            segments.append([])
            restart_events += 1
        if not segments:
            segments.append([])
        segments[-1].append(packet)

    reconstructed_sequences: list[list[int]] = []
    segment_records = []
    duplicate_counter_packets = 0
    positive_interarrival = []
    best_snr = []
    best_rssi = []
    gateway_diversity = []
    for packet in ordered_packets:
        if packet["snr"]:
            best_snr.append(max(packet["snr"]))
        if packet["rssi"]:
            best_rssi.append(max(packet["rssi"]))
        gateway_diversity.append(len(packet["gateways"]))

    for segment in segments:
        observed_counters = [packet["frame_counter"] for packet in segment]
        observed = set(observed_counters)
        duplicate_counter_packets += len(observed_counters) - len(observed)
        start = min(observed)
        end = max(observed)
        reconstructed_sequences.append(
            [0 if counter in observed else 1 for counter in range(start, end + 1)]
        )
        sequence = reconstructed_sequences[-1]
        segment_runs = runs_of_ones(sequence)
        segment_records.append(
            {
                "start_counter": start,
                "end_counter": end,
                "received_packet_count": len(observed),
                "internal_frame_span": len(sequence),
                "internal_gap_count": sum(sequence),
                "internal_gap_fraction_lower_bound": sum(sequence) / len(sequence),
                "lag1_failure_correlation": lag1_correlation(sequence),
                "maximum_loss_run": max(segment_runs, default=0),
                "loss_trace_rle": run_length_encode(sequence),
            }
        )
        for left, right in zip(segment, segment[1:]):
            delta = right["timestamp"] - left["timestamp"]
            if delta > 0:
                positive_interarrival.append(delta)

    total_internal_frames = sum(len(sequence) for sequence in reconstructed_sequences)
    internal_gap_count = sum(sum(sequence) for sequence in reconstructed_sequences)
    transitions = Counter()
    loss_runs = []
    correlations = []
    for sequence in reconstructed_sequences:
        transitions.update(zip(sequence, sequence[1:]))
        loss_runs.extend(runs_of_ones(sequence))
        correlation = lag1_correlation(sequence)
        if correlation is not None:
            correlations.append(correlation)

    return {
        "flight_id": flight_id,
        "member": member,
        "row_count": row_count,
        "malformed_row_count": malformed_rows,
        "unique_packet_count": len(ordered_packets),
        "gateway_count": len(all_gateways),
        "segment_count": len(segments),
        "counter_restart_events": restart_events,
        "duplicate_counter_packets_within_segments": duplicate_counter_packets,
        "internal_frame_span": total_internal_frames,
        "internal_gap_count": internal_gap_count,
        "internal_gap_fraction_lower_bound": (
            internal_gap_count / total_internal_frames
            if total_internal_frames
            else None
        ),
        "transition_counts": {
            "success_to_success": transitions[(0, 0)],
            "success_to_failure": transitions[(0, 1)],
            "failure_to_success": transitions[(1, 0)],
            "failure_to_failure": transitions[(1, 1)],
        },
        "lag1_failure_correlation": (
            statistics.fmean(correlations) if correlations else None
        ),
        "loss_run_count": len(loss_runs),
        "maximum_loss_run": max(loss_runs, default=0),
        "mean_loss_run": statistics.fmean(loss_runs) if loss_runs else 0.0,
        "median_observed_interarrival_seconds": (
            statistics.median(positive_interarrival)
            if positive_interarrival
            else None
        ),
        "median_best_gateway_snr_db": (
            statistics.median(best_snr) if best_snr else None
        ),
        "median_best_gateway_rssi_dbm": (
            statistics.median(best_rssi) if best_rssi else None
        ),
        "mean_gateways_per_received_packet": (
            statistics.fmean(gateway_diversity) if gateway_diversity else None
        ),
        "segments": segment_records,
    }


def cluster_bootstrap_gap_fraction(
    flights: list[dict[str, Any]], replicates: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [generator.choice(flights) for _ in flights]
        gaps = sum(item["internal_gap_count"] for item in sampled)
        frames = sum(item["internal_frame_span"] for item in sampled)
        estimates.append(gaps / frames if frames else math.nan)
    return [value for value in estimates if math.isfinite(value)]


def analyze(
    split_path: Path,
    protocol_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development_ids = split["aadm"]["development_flight_ids"]
    confirmation_ids = set(split["aadm"]["confirmation_flight_ids"])
    if set(development_ids) & confirmation_ids:
        raise ValueError("development and confirmation flights overlap")

    with ZipFile(archive_path) as archive:
        flights = [parse_lora_flight(archive, flight_id) for flight_id in development_ids]

    transitions = Counter()
    for flight in flights:
        counts = flight["transition_counts"]
        transitions[(0, 0)] += counts["success_to_success"]
        transitions[(0, 1)] += counts["success_to_failure"]
        transitions[(1, 0)] += counts["failure_to_success"]
        transitions[(1, 1)] += counts["failure_to_failure"]
    success_origins = transitions[(0, 0)] + transitions[(0, 1)]
    failure_origins = transitions[(1, 0)] + transitions[(1, 1)]
    p_success_to_failure = (
        transitions[(0, 1)] / success_origins if success_origins else 0.0
    )
    p_failure_to_success = (
        transitions[(1, 0)] / failure_origins if failure_origins else 1.0
    )
    stationary_failure = (
        p_success_to_failure / (p_success_to_failure + p_failure_to_success)
        if p_success_to_failure + p_failure_to_success
        else 0.0
    )

    total_gaps = sum(item["internal_gap_count"] for item in flights)
    total_frames = sum(item["internal_frame_span"] for item in flights)
    pooled_gap_fraction = total_gaps / total_frames if total_frames else 0.0
    replicates = int(protocol["statistics"]["paired_cluster_bootstrap_replicates"])
    bootstrap = cluster_bootstrap_gap_fraction(
        flights, replicates, int(protocol["statistics"]["bootstrap_seed"])
    )
    valid_segment_correlations = [
        segment["lag1_failure_correlation"]
        for item in flights
        for segment in item["segments"]
        if segment["lag1_failure_correlation"] is not None
    ]
    threshold = float(
        protocol["fault_model_decision"]["burst_diagnostic_absolute_lag1_threshold"]
    )
    mean_segment_lag1 = (
        statistics.fmean(valid_segment_correlations)
        if valid_segment_correlations
        else None
    )
    short_range_burst_evidence = any(
        abs(value) >= threshold for value in valid_segment_correlations
    )
    flight_gap_fractions = [
        item["internal_gap_fraction_lower_bound"] for item in flights
    ]
    heterogeneous_rate_evidence = (
        max(flight_gap_fractions) - min(flight_gap_fractions) >= 0.20
    )
    result = {
        "version": "1.0",
        "status": "AADM_DEVELOPMENT_ANALYZED",
        "protocol_id": protocol["protocol_id"],
        "access_log": {
            "development_flights_opened": len(flights),
            "development_flight_ids": development_ids,
            "confirmation_flights_opened": 0,
            "confirmation_flight_ids_remain_unread": sorted(confirmation_ids),
        },
        "estimator": {
            "unit": "complete physical-testbed flight",
            "packet_identity": "LoRa deduplication_id",
            "loss_definition": "missing internal frame counter within each monotonic counter segment",
            "counter_restart_rule": "start a new segment whenever chronological frame counter decreases",
            "boundary_loss_observability": "leading and trailing losses are unobservable; estimate is a lower bound",
        },
        "aggregate": {
            "flight_count": len(flights),
            "received_unique_packets": sum(
                item["unique_packet_count"] for item in flights
            ),
            "internal_frame_span": total_frames,
            "internal_gap_count": total_gaps,
            "pooled_internal_gap_fraction_lower_bound": pooled_gap_fraction,
            "cluster_bootstrap_95_ci": [
                percentile(bootstrap, 0.025),
                percentile(bootstrap, 0.975),
            ],
            "flights_with_any_internal_gap": sum(
                item["internal_gap_count"] > 0 for item in flights
            ),
            "flights_with_counter_restart": sum(
                item["counter_restart_events"] > 0 for item in flights
            ),
            "counter_restart_events": sum(
                item["counter_restart_events"] for item in flights
            ),
            "mean_flight_gap_fraction": statistics.fmean(
                item["internal_gap_fraction_lower_bound"] for item in flights
            ),
            "median_flight_gap_fraction": statistics.median(
                item["internal_gap_fraction_lower_bound"] for item in flights
            ),
            "mean_flight_lag1_failure_correlation": statistics.fmean(
                item["lag1_failure_correlation"]
                for item in flights
                if item["lag1_failure_correlation"] is not None
            ),
            "mean_segment_lag1_failure_correlation": mean_segment_lag1,
            "maximum_absolute_segment_lag1_failure_correlation": max(
                (abs(value) for value in valid_segment_correlations),
                default=None,
            ),
            "maximum_observed_loss_run": max(
                item["maximum_loss_run"] for item in flights
            ),
            "gilbert_elliott_proxy": {
                "p_success_to_failure": p_success_to_failure,
                "p_failure_to_success": p_failure_to_success,
                "stationary_failure_probability": stationary_failure,
                "mean_failure_run_implied": (
                    1.0 / p_failure_to_success
                    if p_failure_to_success > 0
                    else None
                ),
                "transition_counts": {
                    "success_to_success": transitions[(0, 0)],
                    "success_to_failure": transitions[(0, 1)],
                    "failure_to_success": transitions[(1, 0)],
                    "failure_to_failure": transitions[(1, 1)],
                },
            },
            "short_range_burst_failure_evidence": short_range_burst_evidence,
            "between_flight_rate_heterogeneity_evidence": heterogeneous_rate_evidence,
        },
        "per_flight": flights,
        "development_decision": {
            "homogeneous_iid_loss_supported": (
                not short_range_burst_evidence
                and not heterogeneous_rate_evidence
            ),
            "within_segment_short_range_iid_diagnostic_supported": (
                not short_range_burst_evidence
            ),
            "empirical_hierarchical_trace_model_should_be_evaluated": (
                heterogeneous_rate_evidence
            ),
            "task_loss_10_percent_empirically_supported": (
                0.10
                >= percentile(bootstrap, 0.025)
                and 0.10 <= percentile(bootstrap, 0.975)
            ),
            "ack_loss_calibrated": False,
            "context_reset_calibrated": False,
        },
        "claim_boundary": [
            "This is an exploratory development estimate and not a confirmation result.",
            "Frame-counter gaps estimate only internal LoRa uplink losses.",
            "Pooled Markov transitions mix low-loss and high-loss flights; they are a proxy and are not interpreted as a stationary channel.",
            "The logs do not identify semantic ACK loss.",
            "A LoRa sender counter restart is not a semantic receiver-context reset.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "results/stage8/stage8_fault_split_freeze_v1/result.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/stage8_fault_calibration_protocol_v1.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            r"F:\uav_spectrum_stage8_fault_data\source_archives\aadm\AADM2025Dryad.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_aadm_development_v1/result.json",
    )
    args = parser.parse_args()
    result = analyze(args.split, args.protocol, args.archive, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate": result["aggregate"],
                "decision": result["development_decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
