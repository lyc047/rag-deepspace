"""Audit reset and continuity proxies in frozen ALFA development raw-flight groups."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def csv_rows(archive: ZipFile, member: str) -> Iterator[dict[str, str]]:
    stream = io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="")
    yield from csv.DictReader(stream)


def parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(int(start), int(end)) for start, end in merged]


def gap_summary(timestamps_ns: list[int], minimum_threshold_seconds: float) -> dict[str, Any]:
    ordered = sorted(set(timestamps_ns))
    gaps = [
        (right - left) / 1e9
        for left, right in zip(ordered, ordered[1:])
        if right > left
    ]
    median_gap = statistics.median(gaps) if gaps else None
    threshold = (
        max(minimum_threshold_seconds, 10.0 * median_gap)
        if median_gap is not None
        else minimum_threshold_seconds
    )
    large = [gap for gap in gaps if gap > threshold]
    return {
        "sample_count": len(ordered),
        "median_interval_seconds": median_gap,
        "large_gap_threshold_seconds": threshold,
        "large_gap_count": len(large),
        "maximum_gap_seconds": max(gaps, default=None),
    }


def find_member(names: set[str], sequence: str, suffix: str) -> str | None:
    member = f"processed/{sequence}/{sequence}-{suffix}"
    return member if member in names else None


def analyze_sequence(
    archive: ZipFile, names: set[str], raw_id: str, sequence: str
) -> dict[str, Any]:
    state_member = find_member(names, sequence, "mavros-state.csv")
    mavlink_member = find_member(names, sequence, "mavlink-from.csv")
    time_reference_member = find_member(
        names, sequence, "mavros-time_reference.csv"
    )
    result: dict[str, Any] = {
        "raw_flight_id": raw_id,
        "sequence_id": sequence,
        "members_present": {
            "mavros_state": state_member is not None,
            "mavlink_from": mavlink_member is not None,
            "time_reference": time_reference_member is not None,
        },
    }

    all_times = []
    if state_member is not None:
        state_times = []
        state_header_seq = []
        connected = []
        system_status = []
        for row in csv_rows(archive, state_member):
            time_ns = parse_int(row.get("%time"))
            seq = parse_int(row.get("field.header.seq"))
            connection = parse_int(row.get("field.connected"))
            status = parse_int(row.get("field.system_status"))
            if time_ns is not None:
                state_times.append(time_ns)
                all_times.append(time_ns)
            if seq is not None:
                state_header_seq.append(seq)
            if connection is not None:
                connected.append(connection)
            if status is not None:
                system_status.append(status)
        disconnect_samples = sum(value == 0 for value in connected)
        reconnect_transitions = sum(
            left == 0 and right == 1 for left, right in zip(connected, connected[1:])
        )
        result["mavros_state"] = {
            **gap_summary(state_times, 5.0),
            "disconnect_sample_count": disconnect_samples,
            "reconnect_transition_count": reconnect_transitions,
            "header_sequence_decrease_count": sum(
                right < left
                for left, right in zip(state_header_seq, state_header_seq[1:])
            ),
            "system_status_values": sorted(set(system_status)),
        }

    if mavlink_member is not None:
        mavlink_times = []
        header_seq = []
        component_sequence: dict[tuple[int, int], list[int]] = defaultdict(list)
        boot_samples: list[tuple[int, int]] = []
        malformed_rows = 0
        for row in csv_rows(archive, mavlink_member):
            time_ns = parse_int(row.get("%time"))
            ros_seq = parse_int(row.get("field.header.seq"))
            sequence_number = parse_int(row.get("field.seq"))
            system_id = parse_int(row.get("field.sysid"))
            component_id = parse_int(row.get("field.compid"))
            message_id = parse_int(row.get("field.msgid"))
            if time_ns is not None:
                mavlink_times.append(time_ns)
                all_times.append(time_ns)
            if ros_seq is not None:
                header_seq.append(ros_seq)
            if (
                sequence_number is not None
                and system_id is not None
                and component_id is not None
            ):
                component_sequence[(system_id, component_id)].append(sequence_number)
            if message_id == 2 and time_ns is not None:
                payload_word = parse_int(row.get("field.payload641"))
                if payload_word is not None:
                    boot_samples.append((time_ns, payload_word & 0xFFFFFFFF))
            if time_ns is None or message_id is None:
                malformed_rows += 1

        inferred_missing_messages = 0
        backward_or_reordered = 0
        duplicate_sequence = 0
        for values in component_sequence.values():
            for left, right in zip(values, values[1:]):
                delta = (right - left) % 256
                if delta == 0:
                    duplicate_sequence += 1
                elif 1 < delta <= 128:
                    inferred_missing_messages += delta - 1
                elif delta > 128:
                    backward_or_reordered += 1
        boot_samples.sort()
        boot_reset_events = sum(
            right_boot + 1000 < left_boot
            for (_, left_boot), (_, right_boot) in zip(
                boot_samples, boot_samples[1:]
            )
        )
        result["mavlink"] = {
            **gap_summary(mavlink_times, 1.0),
            "malformed_row_count": malformed_rows,
            "ros_header_sequence_decrease_count": sum(
                right < left for left, right in zip(header_seq, header_seq[1:])
            ),
            "component_stream_count": len(component_sequence),
            "mavlink_sequence_inferred_missing_messages": inferred_missing_messages,
            "mavlink_sequence_duplicate_count": duplicate_sequence,
            "mavlink_sequence_backward_or_reordered_count": backward_or_reordered,
            "system_time_boot_sample_count": len(boot_samples),
            "system_time_boot_reset_event_count": boot_reset_events,
            "system_time_boot_observable": bool(boot_samples),
        }

    if time_reference_member is not None:
        reference_times = []
        source_times = []
        for row in csv_rows(archive, time_reference_member):
            time_ns = parse_int(row.get("%time"))
            reference_ns = parse_int(row.get("field.time_ref"))
            if time_ns is not None:
                reference_times.append(time_ns)
                all_times.append(time_ns)
            if reference_ns is not None:
                source_times.append(reference_ns)
        result["time_reference"] = {
            **gap_summary(reference_times, 2.0),
            "source_time_backward_count": sum(
                right < left for left, right in zip(source_times, source_times[1:])
            ),
        }

    if all_times:
        result["observed_interval_ns"] = [min(all_times), max(all_times)]
        result["observed_duration_seconds"] = (max(all_times) - min(all_times)) / 1e9
    else:
        result["observed_interval_ns"] = None
        result["observed_duration_seconds"] = 0.0
    return result


def analyze(
    split_path: Path,
    protocol_path: Path,
    archive_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development_ids = split["alfa"]["development_raw_flight_ids"]
    confirmation_ids = set(split["alfa"]["confirmation_raw_flight_ids"])
    if set(development_ids) & confirmation_ids:
        raise ValueError("development and confirmation raw flights overlap")

    inventory = split["alfa"]["raw_flight_inventory"]
    sequence_records = []
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for raw_id in development_ids:
            for sequence in inventory[raw_id]["processed_sequence_directories"]:
                sequence_records.append(
                    analyze_sequence(archive, names, raw_id, sequence)
                )

    by_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequence_records:
        by_raw[sequence["raw_flight_id"]].append(sequence)
    raw_records = []
    for raw_id in development_ids:
        sequences = by_raw.get(raw_id, [])
        intervals = [
            tuple(sequence["observed_interval_ns"])
            for sequence in sequences
            if sequence["observed_interval_ns"] is not None
        ]
        merged = merge_intervals(intervals)
        observed_seconds = sum((end - start) / 1e9 for start, end in merged)
        raw_records.append(
            {
                "raw_flight_id": raw_id,
                "processed_sequence_count": len(sequences),
                "processed_sequence_ids": [
                    sequence["sequence_id"] for sequence in sequences
                ],
                "merged_observed_duration_seconds": observed_seconds,
                "system_time_boot_sample_count": sum(
                    sequence.get("mavlink", {}).get(
                        "system_time_boot_sample_count", 0
                    )
                    for sequence in sequences
                ),
                "system_time_boot_reset_event_count": sum(
                    sequence.get("mavlink", {}).get(
                        "system_time_boot_reset_event_count", 0
                    )
                    for sequence in sequences
                ),
                "mavros_reconnect_transition_count": sum(
                    sequence.get("mavros_state", {}).get(
                        "reconnect_transition_count", 0
                    )
                    for sequence in sequences
                ),
                "mavros_disconnect_sample_count": sum(
                    sequence.get("mavros_state", {}).get(
                        "disconnect_sample_count", 0
                    )
                    for sequence in sequences
                ),
                "mavlink_large_gap_count": sum(
                    sequence.get("mavlink", {}).get("large_gap_count", 0)
                    for sequence in sequences
                ),
                "mavlink_inferred_missing_messages": sum(
                    sequence.get("mavlink", {}).get(
                        "mavlink_sequence_inferred_missing_messages", 0
                    )
                    for sequence in sequences
                ),
            }
        )

    covered_raw = [record for record in raw_records if record["processed_sequence_count"]]
    total_hours = sum(
        record["merged_observed_duration_seconds"] for record in raw_records
    ) / 3600.0
    boot_samples = sum(
        record["system_time_boot_sample_count"] for record in raw_records
    )
    boot_events = sum(
        record["system_time_boot_reset_event_count"] for record in raw_records
    )
    poisson_upper_per_hour = (
        -math.log(0.05) / total_hours
        if boot_events == 0 and total_hours > 0
        else None
    )
    result = {
        "version": "1.0",
        "status": "ALFA_DEVELOPMENT_AUDITED",
        "protocol_id": protocol["protocol_id"],
        "access_log": {
            "development_raw_flights_registered": len(development_ids),
            "development_raw_flights_with_processed_values_opened": len(covered_raw),
            "development_processed_sequences_opened": len(sequence_records),
            "confirmation_raw_flights_opened": 0,
            "confirmation_raw_flight_ids_remain_unread": sorted(confirmation_ids),
            "raw_bag_payloads_opened": 0,
            "telemetry_payloads_opened": 0,
            "dataflash_payloads_opened": 0,
        },
        "detectors": {
            "fcu_boot_reset": (
                "MAVLink SYSTEM_TIME (message 2) time_boot_ms decreases by more "
                "than 1000 ms within a processed sequence"
            ),
            "transport_reconnect": (
                "MAVROS state field.connected transitions from 0 to 1"
            ),
            "large_gap": (
                "timestamp gap exceeds max(fixed minimum, 10 times the "
                "within-sequence median interval)"
            ),
            "segment_start_not_counted_as_reset": True,
        },
        "aggregate": {
            "raw_flights_registered": len(development_ids),
            "raw_flights_with_processed_coverage": len(covered_raw),
            "coverage_fraction": len(covered_raw) / len(development_ids),
            "processed_sequence_count": len(sequence_records),
            "merged_observed_duration_hours": total_hours,
            "system_time_boot_sample_count": boot_samples,
            "fcu_boot_reset_event_count": boot_events,
            "mavros_reconnect_transition_count": sum(
                record["mavros_reconnect_transition_count"]
                for record in raw_records
            ),
            "mavros_disconnect_sample_count": sum(
                record["mavros_disconnect_sample_count"] for record in raw_records
            ),
            "mavlink_large_gap_count": sum(
                record["mavlink_large_gap_count"] for record in raw_records
            ),
            "mavlink_sequence_inferred_missing_messages": sum(
                record["mavlink_inferred_missing_messages"] for record in raw_records
            ),
            "descriptive_zero_event_poisson_95_upper_rate_per_hour": (
                poisson_upper_per_hour
            ),
        },
        "per_raw_flight": raw_records,
        "per_processed_sequence": sequence_records,
        "development_decision": {
            "fcu_boot_marker_observable": boot_samples > 0,
            "fcu_boot_reset_observed": boot_events > 0,
            "semantic_receiver_context_reset_probability_calibratable": False,
            "reason": (
                "ALFA observes the Pixhawk/MAVLink source and selected ROS "
                "processed clips, not the semantic receiver process, its codebook "
                "catalog, or its volatile action cache."
            ),
            "retain_two_percent_context_reset_as_empirical_value": False,
            "retain_two_percent_context_reset_as_stress_only": True,
        },
        "claim_boundary": [
            "Processed clips from the same raw flight were not treated as independent.",
            "No confirmation processed value or raw bag payload was opened.",
            "A detected FCU boot reset would still not equal a semantic receiver-context reset.",
            "A zero event count cannot be generalized beyond the detector coverage and selected clips.",
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
            r"F:\uav_spectrum_stage8_fault_data\source_archives\alfa\processed.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_alfa_development_v1/result.json",
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
