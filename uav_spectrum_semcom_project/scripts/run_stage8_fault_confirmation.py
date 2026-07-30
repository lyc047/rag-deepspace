#!/usr/bin/env python
"""Single-use confirmation of Stage-8 public fault-model conclusions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage8_aadm_development import (  # noqa: E402
    cluster_bootstrap_gap_fraction,
    parse_lora_flight,
    percentile,
)
from analyze_stage8_alfa_development import (  # noqa: E402
    analyze_sequence,
    merge_intervals,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/stage8_fault_confirmation_v1.json"
DEFAULT_OUTPUT = ROOT / "results/stage8/stage8_fault_confirmation_v1/result.json"


def dispersion_statistic(rows: list[dict[str, Any]], pooled_probability: float) -> float:
    if not 0.0 < pooled_probability < 1.0:
        return 0.0
    statistic = 0.0
    for row in rows:
        total = int(row["internal_frame_span"])
        observed = int(row["internal_gap_count"])
        expected = total * pooled_probability
        statistic += (observed - expected) ** 2 / (
            total * pooled_probability * (1.0 - pooled_probability)
        )
    return statistic


def overdispersion_monte_carlo(
    rows: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    gaps = sum(int(row["internal_gap_count"]) for row in rows)
    frames = sum(int(row["internal_frame_span"]) for row in rows)
    pooled = gaps / frames
    observed = dispersion_statistic(rows, pooled)
    generator = np.random.default_rng(seed)
    totals = np.asarray(
        [int(row["internal_frame_span"]) for row in rows], dtype=np.int64
    )
    exceed = 0
    for _ in range(replicates):
        simulated = generator.binomial(totals, pooled)
        simulated_rows = [
            {
                "internal_gap_count": int(value),
                "internal_frame_span": int(total),
            }
            for value, total in zip(simulated, totals)
        ]
        if dispersion_statistic(simulated_rows, pooled) >= observed:
            exceed += 1
    return {
        "pooled_probability": pooled,
        "pearson_statistic": observed,
        "degrees_of_freedom": len(rows) - 1,
        "monte_carlo_replicates": replicates,
        "monte_carlo_p_value": (exceed + 1) / (replicates + 1),
    }


def bootstrap_difference(
    confirmation: list[dict[str, Any]],
    development: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    generator = random.Random(seed)
    values = []
    for _ in range(replicates):
        confirm_sample = [
            generator.choice(confirmation) for _ in confirmation
        ]
        develop_sample = [generator.choice(development) for _ in development]
        confirm_rate = sum(row["internal_gap_count"] for row in confirm_sample) / sum(
            row["internal_frame_span"] for row in confirm_sample
        )
        develop_rate = sum(row["internal_gap_count"] for row in develop_sample) / sum(
            row["internal_frame_span"] for row in develop_sample
        )
        values.append(confirm_rate - develop_rate)
    return values


def mark_access_started(
    state_path: Path, aadm_ids: list[str], alfa_ids: list[str]
) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["execution_started"]:
        raise RuntimeError("Stage-8 confirmation access was already started")
    if any(role["access_count"] != 0 for role in state["roles"].values()):
        raise RuntimeError("Stage-8 confirmation role was already consumed")
    timestamp = datetime.now(timezone.utc).isoformat()
    state["execution_started"] = True
    for role_name, unit_ids in (
        ("aadm_confirmation", aadm_ids),
        ("alfa_confirmation", alfa_ids),
    ):
        role = state["roles"][role_name]
        role["access_count"] = 1
        role["signal_values_accessed"] = True
        role["first_access_at_utc"] = timestamp
        role["unit_ids"] = unit_ids
    atomic_write_json(state_path, state)
    return state


def aggregate_alfa(
    archive: ZipFile,
    names: set[str],
    raw_ids: list[str],
    inventory: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = []
    for raw_id in raw_ids:
        for sequence in inventory[raw_id]["processed_sequence_directories"]:
            sequences.append(analyze_sequence(archive, names, raw_id, sequence))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        grouped[sequence["raw_flight_id"]].append(sequence)
    raw_rows = []
    for raw_id in raw_ids:
        local = grouped.get(raw_id, [])
        intervals = [
            tuple(sequence["observed_interval_ns"])
            for sequence in local
            if sequence["observed_interval_ns"] is not None
        ]
        merged = merge_intervals(intervals)
        raw_rows.append(
            {
                "raw_flight_id": raw_id,
                "processed_sequence_count": len(local),
                "processed_sequence_ids": [
                    sequence["sequence_id"] for sequence in local
                ],
                "merged_observed_duration_seconds": sum(
                    (end - start) / 1e9 for start, end in merged
                ),
                "system_time_boot_sample_count": sum(
                    sequence.get("mavlink", {}).get(
                        "system_time_boot_sample_count", 0
                    )
                    for sequence in local
                ),
                "system_time_boot_reset_event_count": sum(
                    sequence.get("mavlink", {}).get(
                        "system_time_boot_reset_event_count", 0
                    )
                    for sequence in local
                ),
                "mavros_reconnect_transition_count": sum(
                    sequence.get("mavros_state", {}).get(
                        "reconnect_transition_count", 0
                    )
                    for sequence in local
                ),
                "mavros_disconnect_sample_count": sum(
                    sequence.get("mavros_state", {}).get(
                        "disconnect_sample_count", 0
                    )
                    for sequence in local
                ),
            }
        )
    return raw_rows, sequences


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite Stage-8 confirmation")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {}
    for name, value in config["inputs"].items():
        candidate = Path(value)
        paths[name] = candidate if candidate.is_absolute() else ROOT / candidate
    for name, expected in config["frozen_input_hashes"].items():
        if sha256_file(paths[name]) != expected:
            raise ValueError(f"frozen {name} hash changed")
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    development_aadm = json.loads(
        paths["aadm_development"].read_text(encoding="utf-8")
    )
    replay = json.loads(paths["development_replay"].read_text(encoding="utf-8"))
    if replay["selection"]["selected_development_candidate"] != "NO_CANDIDATE":
        raise ValueError("confirmation protocol assumes no development candidate")
    aadm_ids = split["aadm"]["confirmation_flight_ids"]
    alfa_ids = split["alfa"]["confirmation_raw_flight_ids"]
    if len(aadm_ids) != 20 or len(alfa_ids) != 20:
        raise ValueError("confirmation unit count changed")

    access = mark_access_started(paths["access_state"], aadm_ids, alfa_ids)

    with ZipFile(paths["aadm_archive"]) as archive:
        aadm_rows = [parse_lora_flight(archive, flight_id) for flight_id in aadm_ids]
    settings = config["statistics"]
    replicates = int(settings["cluster_bootstrap_replicates"])
    bootstrap = cluster_bootstrap_gap_fraction(
        aadm_rows, replicates, int(settings["cluster_bootstrap_seed"])
    )
    gaps = sum(row["internal_gap_count"] for row in aadm_rows)
    frames = sum(row["internal_frame_span"] for row in aadm_rows)
    pooled = gaps / frames
    interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
    segment_correlations = [
        segment["lag1_failure_correlation"]
        for row in aadm_rows
        for segment in row["segments"]
        if segment["lag1_failure_correlation"] is not None
    ]
    dispersion = overdispersion_monte_carlo(
        aadm_rows,
        int(settings["overdispersion_monte_carlo_replicates"]),
        int(settings["overdispersion_seed"]),
    )
    differences = bootstrap_difference(
        aadm_rows,
        development_aadm["per_flight"],
        replicates=replicates,
        seed=int(settings["cluster_bootstrap_seed"]) + 1,
    )

    with ZipFile(paths["alfa_processed_archive"]) as archive:
        names = set(archive.namelist())
        alfa_raw_rows, alfa_sequences = aggregate_alfa(
            archive, names, alfa_ids, split["alfa"]["raw_flight_inventory"]
        )
    covered_alfa = [
        row for row in alfa_raw_rows if row["processed_sequence_count"] > 0
    ]
    alfa_hours = sum(
        row["merged_observed_duration_seconds"] for row in alfa_raw_rows
    ) / 3600.0
    alfa_boot_events = sum(
        row["system_time_boot_reset_event_count"] for row in alfa_raw_rows
    )
    alpha = float(settings["alpha"])
    result = {
        "version": "1.0",
        "status": "STAGE8_FAULT_CONFIRMATION_COMPLETE",
        "verification_status": "SINGLE_USE_CONFIRMATION",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "frozen_input_hashes": {
            name: sha256_file(paths[name])
            for name in config["frozen_input_hashes"]
        },
        "access_state_after_start": access,
        "aadm_confirmation": {
            "independent_flight_count": len(aadm_rows),
            "received_unique_packets": sum(
                row["unique_packet_count"] for row in aadm_rows
            ),
            "internal_frame_span": frames,
            "internal_gap_count": gaps,
            "pooled_internal_gap_fraction_lower_bound": pooled,
            "cluster_bootstrap_95_ci": interval,
            "fixed_10_percent_inside_interval": interval[0] <= 0.10 <= interval[1],
            "flights_with_any_internal_gap": sum(
                row["internal_gap_count"] > 0 for row in aadm_rows
            ),
            "flight_gap_fraction_minimum": min(
                row["internal_gap_fraction_lower_bound"] for row in aadm_rows
            ),
            "flight_gap_fraction_maximum": max(
                row["internal_gap_fraction_lower_bound"] for row in aadm_rows
            ),
            "flights_with_counter_restart": sum(
                row["counter_restart_events"] > 0 for row in aadm_rows
            ),
            "maximum_loss_run": max(row["maximum_loss_run"] for row in aadm_rows),
            "maximum_absolute_segment_lag1": max(
                (abs(value) for value in segment_correlations), default=None
            ),
            "short_range_lag1_threshold": float(
                settings["lag1_absolute_threshold"]
            ),
            "homogeneous_binomial_overdispersion_test": dispersion,
            "homogeneous_model_rejected": (
                dispersion["monte_carlo_p_value"] < alpha
            ),
            "confirmation_minus_development_gap_fraction": {
                "mean": sum(differences) / len(differences),
                "ci_lower": percentile(differences, 0.025),
                "ci_upper": percentile(differences, 0.975),
            },
            "per_flight": aadm_rows,
        },
        "alfa_confirmation": {
            "registered_raw_flight_count": len(alfa_ids),
            "raw_flights_with_processed_coverage": len(covered_alfa),
            "coverage_fraction": len(covered_alfa) / len(alfa_ids),
            "processed_sequence_count": len(alfa_sequences),
            "merged_observed_duration_hours": alfa_hours,
            "system_time_boot_sample_count": sum(
                row["system_time_boot_sample_count"] for row in alfa_raw_rows
            ),
            "fcu_boot_reset_event_count": alfa_boot_events,
            "mavros_reconnect_transition_count": sum(
                row["mavros_reconnect_transition_count"] for row in alfa_raw_rows
            ),
            "mavros_disconnect_sample_count": sum(
                row["mavros_disconnect_sample_count"] for row in alfa_raw_rows
            ),
            "descriptive_zero_event_poisson_95_upper_rate_per_hour": (
                -math.log(0.05) / alfa_hours
                if alfa_boot_events == 0 and alfa_hours > 0
                else None
            ),
            "semantic_receiver_context_reset_probability_calibratable": False,
            "per_raw_flight": alfa_raw_rows,
            "per_processed_sequence": alfa_sequences,
        },
        "decisions": {
            "flight_hierarchical_task_loss_model_confirmed": (
                dispersion["monte_carlo_p_value"] < alpha
            ),
            "exact_10_percent_loss_rejected_by_cluster_interval": not (
                interval[0] <= 0.10 <= interval[1]
            ),
            "semantic_context_reset_calibrated": False,
            "new_heartbeat_candidate": False,
            "retain_fixed10_role": "stress_model_engineering_boundary",
            "board_validation_still_required": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(output_path, result)

    final_state = json.loads(paths["access_state"].read_text(encoding="utf-8"))
    final_state["evidence"] = {
        "output_path": str(output_path),
        "config_sha256": sha256_file(config_path),
        "result_sha256": sha256_file(output_path),
    }
    atomic_write_json(paths["access_state"], final_state)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "aadm": {
                    key: result["aadm_confirmation"][key]
                    for key in (
                        "pooled_internal_gap_fraction_lower_bound",
                        "cluster_bootstrap_95_ci",
                        "homogeneous_model_rejected",
                    )
                },
                "alfa": {
                    key: result["alfa_confirmation"][key]
                    for key in (
                        "merged_observed_duration_hours",
                        "fcu_boot_reset_event_count",
                        "semantic_receiver_context_reset_probability_calibratable",
                    )
                },
                "decisions": result["decisions"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
