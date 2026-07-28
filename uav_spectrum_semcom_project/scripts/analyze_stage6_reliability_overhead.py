#!/usr/bin/env python
"""Attribute the frozen grouped negative result to communication components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_reliability_protocol_diagnostic_v1.json"
)


def _verified_json(entry: dict, label: str) -> dict:
    path = PROJECT_DIR / entry["path"]
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluated_groups(grouped: dict, n_key: str) -> list[dict]:
    value = grouped["results"][n_key]
    return [
        value["outer_groups"][key]
        for key in value["configured_outer_group_ids"]
        if value["outer_groups"][key].get("status", "evaluated")
        == "evaluated"
    ]


def _semantic_rows(
    groups: list[dict],
    *,
    candidate_id: str,
    deployment_mode: str,
    scene_count: int,
) -> list[dict]:
    rows = []
    for group in groups:
        matches = [
            row
            for row in group["semantic_workpoints"]
            if row["candidate_id"] == candidate_id
            and row["deployment_mode"] == deployment_mode
            and int(row["requested_session_scene_count"]) == int(scene_count)
        ]
        if len(matches) != 1:
            raise ValueError("semantic workpoint identity is not unique")
        rows.append(matches[0])
    return rows


def _exact_rows(
    groups: list[dict],
    *,
    attempts: int,
    scene_count: int,
) -> list[dict]:
    rows = []
    for group in groups:
        matches = [
            row
            for row in group["exact_workpoints"]
            if int(row["task_packet_open_loop_attempts"]) == int(attempts)
            and int(row["requested_session_scene_count"]) == int(scene_count)
        ]
        if len(matches) != 1:
            raise ValueError("exact workpoint identity is not unique")
        rows.append(matches[0])
    return rows


def _weighted_metric(rows: list[dict], metric: str) -> float:
    weights = np.asarray(
        [float(row["evaluated_scene_count"]) for row in rows],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(row["summary"][metric]) for row in rows],
        dtype=np.float64,
    )
    return float(np.sum(values * weights) / np.sum(weights))


def _count_rate(rows: list[dict], metric: str) -> float:
    total_count = sum(float(row["summary"][metric]) for row in rows)
    total_scenes = sum(int(row["evaluated_scene_count"]) for row in rows)
    return float(total_count / total_scenes)


def _savings(exact_bits: float, semantic_bits: float) -> float:
    return float(100.0 * (exact_bits - semantic_bits) / exact_bits)


def analyze(protocol: dict, grouped: dict, grouped_analysis: dict) -> dict:
    candidate_protocol = _verified_json(
        protocol["frozen_inputs"]["grouped_candidate_protocol"],
        "grouped candidate protocol",
    )
    rows_by_n = {}
    hypotheses = {}
    for n_channels in protocol["evaluation_scope"]["n_channels"]:
        n_key = str(int(n_channels))
        candidate = candidate_protocol["controller_candidates"][n_key][0]
        candidate_id = candidate["candidate_id"]
        groups = _evaluated_groups(grouped, n_key)
        semantic = _semantic_rows(
            groups,
            candidate_id=candidate_id,
            deployment_mode="preconfigured_codebook_empty_action_state",
            scene_count=500,
        )
        exact = _exact_rows(groups, attempts=1, scene_count=500)
        semantic_bits = _weighted_metric(
            semantic, "actual_bits_per_scene"
        )
        exact_bits = _weighted_metric(exact, "actual_bits_per_scene")
        components = {
            name: _weighted_metric(semantic, name)
            for name in (
                "forward_bits_per_scene",
                "feedback_bits_per_scene",
                "task_frame_bits_per_scene",
                "heartbeat_bits_per_scene",
                "initial_install_bits_per_scene",
                "recovery_install_bits_per_scene",
                "escape_bits_per_scene",
            )
        }
        oracle_bits = (
            semantic_bits - components["recovery_install_bits_per_scene"]
        )
        row = {
            "n_channels": int(n_channels),
            "candidate_id": candidate_id,
            "deployment_mode": (
                "preconfigured_codebook_empty_action_state"
            ),
            "requested_session_scene_count": 500,
            "evaluated_scene_count": int(
                sum(item["evaluated_scene_count"] for item in semantic)
            ),
            "clean_rate": _weighted_metric(semantic, "clean_rate"),
            "semantic_actual_bits_per_scene": semantic_bits,
            "exact_one_attempt_bits_per_scene": exact_bits,
            "observed_savings_percentage": _savings(
                exact_bits, semantic_bits
            ),
            "component_bits_per_scene": components,
            "component_share_of_semantic_total_percentage": {
                name: float(100.0 * value / semantic_bits)
                for name, value in components.items()
            },
            "event_rates_per_scene": {
                "compact_update": _count_rate(
                    semantic, "compact_update_count"
                ),
                "context_install": _count_rate(
                    semantic, "context_install_count"
                ),
                "heartbeat_probe": _count_rate(
                    semantic, "heartbeat_probe_count"
                ),
            },
            "non_deployable_no_recovery_install_upper_bound": {
                "bits_per_scene": oracle_bits,
                "savings_percentage": _savings(exact_bits, oracle_bits),
                "safety_claim_allowed": False,
            },
        }
        rows_by_n[n_key] = row
        hypotheses[n_key] = {
            "h1_recovery_share_exceeds_35_percent": (
                row["component_share_of_semantic_total_percentage"][
                    "recovery_install_bits_per_scene"
                ]
                > 35.0
            ),
            "h2_no_recovery_upper_bound_has_positive_savings": (
                row[
                    "non_deployable_no_recovery_install_upper_bound"
                ]["savings_percentage"]
                > 0.0
            ),
        }

    primary_supported = int(
        grouped_analysis["decision_summary"][
            "supported_primary_candidate_comparison_count"
        ]
    )
    return {
        "version": "1.0",
        "status": "reliability_overhead_diagnostic_complete",
        "protocol_id": protocol["protocol_id"],
        "negative_result_retained": {
            "supported_primary_candidate_comparison_count": (
                primary_supported
            ),
            "primary_development_success": bool(
                grouped_analysis["decision_summary"][
                    "primary_development_success"
                ]
            ),
        },
        "primary_l500_preconfigured_by_n": rows_by_n,
        "diagnostic_hypothesis_checks": hypotheses,
        "aggregate_checks": {
            "h1_passes_all_n": all(
                item["h1_recovery_share_exceeds_35_percent"]
                for item in hypotheses.values()
            ),
            "h2_passes_all_n": all(
                item[
                    "h2_no_recovery_upper_bound_has_positive_savings"
                ]
                for item in hypotheses.values()
            ),
            "wrong_codebook_actions_remain_zero": grouped["checks"][
                "wrong_codebook_actions_must_be_zero"
            ],
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "interpretation_boundary": (
            "The no-recovery calculation is an accounting upper bound, not "
            "a deployable protocol result. It motivates R1 but cannot support "
            "a reliability or savings claim by itself."
        ),
        "environment": environment_snapshot(["numpy"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    grouped = _verified_json(
        protocol["frozen_inputs"]["grouped_result"], "grouped result"
    )
    grouped_analysis = _verified_json(
        protocol["frozen_inputs"]["grouped_analysis"], "grouped analysis"
    )
    if (
        protocol["governance"]["external_final_signal_values_may_be_loaded"]
        or not protocol["governance"][
            "external_final_access_count_must_remain_zero"
        ]
    ):
        raise ValueError("external Final boundary changed")
    result = analyze(protocol, grouped, grouped_analysis)
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["overhead_diagnostic"]
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite diagnostic result")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
