#!/usr/bin/env python
"""Run preregistered Stage-9.5 causal RESET-NACK experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectrum_semcom.stage9_1_hybrid_recovery import generate_hybrid_fault_schedule  # noqa: E402
from spectrum_semcom.stage9_5_causal_feedback import (  # noqa: E402
    POLICIES,
    simulate_causal_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import SilProtocolBits, generate_task_trace  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(int(replicates), data.size))
    means = np.mean(data[draws], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {"trajectory_count": len(rows)}
    for index, metric in enumerate(("bits_per_scene", "clean_rate", "availability")):
        values = np.asarray([row[metric] for row in rows])
        output[metric] = {
            "mean": float(np.mean(values)),
            "ci": _ci(values, replicates, seed + index),
        }
    latencies = [value for row in rows for value in row["recovery_latencies"]]
    output["recovery_latency_p95_scenes"] = (
        float(np.quantile(latencies, 0.95)) if latencies else None
    )
    for metric in (
        "restore_frame_count",
        "reset_nack_frame_count",
        "feedback_transition_count",
        "sender_internal_state_read_count",
        "uncharged_feedback_transition_count",
        "wrong_context_execution_count",
        "stale_restore_acceptance_count",
    ):
        output[metric] = int(sum(row[metric] for row in rows))
    output["component_bits_per_scene"] = {
        name: float(np.mean([row[name] / row["scene_count"] for row in rows]))
        for name in (
            "heartbeat_bits",
            "boot_status_bits",
            "reset_nack_bits",
            "install_bits",
            "update_bits",
            "ack_bits",
        )
    }
    return output


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    clean = np.asarray([100.0 * (x["clean_rate"] - y["clean_rate"]) for x, y in zip(a, b)])
    availability = np.asarray([100.0 * (x["availability"] - y["availability"]) for x, y in zip(a, b)])
    bit_change = np.asarray([100.0 * (x["total_bits"] - y["total_bits"]) / y["total_bits"] for x, y in zip(a, b)])
    return {
        "clean_delta_percentage_points_mean": float(np.mean(clean)),
        "clean_delta_percentage_points_ci": _ci(clean, replicates, seed),
        "availability_delta_percentage_points_mean": float(np.mean(availability)),
        "availability_delta_percentage_points_ci": _ci(availability, replicates, seed + 1),
        "relative_bit_change_percent_mean": float(np.mean(bit_change)),
        "relative_bit_change_percent_ci": _ci(bit_change, replicates, seed + 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", default="configs/stage9_5_causal_reset_nack_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze", default="configs/stage9_5_causal_reset_nack_freeze_v1.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_fault_config"]
    if _sha256(parent_path) != config["parent_fault_config_sha256"]:
        raise RuntimeError("Stage-9.1 fault configuration changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    input_paths = {
        "config": config_path,
        "parent_config": parent_path,
        "stage9_3_runtime": ROOT / "src/spectrum_semcom/stage9_3_tagged_recovery.py",
        "stage9_5_runtime": ROOT / "src/spectrum_semcom/stage9_5_causal_feedback.py",
        "runner": Path(__file__).resolve(),
        "tests": ROOT / "tests/test_stage9_5_causal_feedback.py",
    }
    hashes = {name: _sha256(path) for name, path in input_paths.items()}
    if args.mode == "confirmation":
        freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
        if freeze["status"] != "stage9_5_candidate_frozen_before_confirmation":
            raise RuntimeError("Stage-9.5 freeze is not active")
        if freeze["input_hashes"] != hashes:
            raise RuntimeError("Stage-9.5 inputs differ from freeze")

    p = parent["protocol_bits"]
    bits = SilProtocolBits(
        int(p["compact_update_bits"]), int(p["cumulative_ack_bits"]),
        int(p["heartbeat_request_bits"]), int(p["heartbeat_response_bits"]),
        int(p["boot_status_bits"]), int(p["preinstalled_activation_bits"]),
        int(p["exact_action_bits"]), int(p["full_install_bits"]),
    )
    mode = config[args.mode]
    trajectories = int(mode["trajectories_per_profile"])
    replicates = int(mode["bootstrap_replicates"])
    seed = int(mode["seed"])
    task = parent["task_trace"]
    profiles: dict[str, Any] = {}

    for profile_index, (name, profile) in enumerate(parent["fault_profiles"].items()):
        rows = {policy: [] for policy in POLICIES}
        for trajectory in range(trajectories):
            sequence = np.random.SeedSequence([seed, profile_index, trajectory])
            trace_rng, fault_rng = [np.random.default_rng(item) for item in sequence.spawn(2)]
            trace = generate_task_trace(
                scene_count=int(task["scene_count"]),
                symbol_count=int(task["semantic_symbol_count"]),
                change_probability=float(task["symbol_change_probability"]),
                equivalence_probability=float(task["temporary_equivalence_probability"]),
                equivalence_scenes=int(task["temporary_equivalence_scenes"]),
                rng=trace_rng,
            )
            values = dict(profile)
            emitter_failure = float(values.pop("boot_emitter_failure_probability_per_boot"))
            faults = generate_hybrid_fault_schedule(
                scene_count=int(task["scene_count"]),
                boot_emitter_failure_probability_per_boot=emitter_failure,
                rng=fault_rng,
                **{key: float(value) for key, value in values.items()},
            )
            for policy in POLICIES:
                rows[policy].append(simulate_causal_trajectory(trace, faults, policy=policy, bits=bits).to_dict())

        summaries = {
            policy: _summary(values, replicates, seed + 100 * profile_index + index)
            for index, (policy, values) in enumerate(rows.items())
        }
        candidate = rows["causal_hbw30_reset_nack"]
        vs_no_nack = _paired(candidate, rows["causal_hbw30_no_nack"], replicates, seed + 1000 + profile_index)
        vs_fixed = _paired(candidate, rows["causal_fixed10_reset_nack"], replicates, seed + 2000 + profile_index)
        vs_event = _paired(candidate, rows["causal_boot_event_reset_nack"], replicates, seed + 3000 + profile_index)
        global_pass = all(
            summaries[policy][metric] == 0
            for policy in POLICIES
            for metric in (
                "sender_internal_state_read_count",
                "uncharged_feedback_transition_count",
                "wrong_context_execution_count",
                "stale_restore_acceptance_count",
            )
        )
        if name.startswith("normal_"):
            nack_gate = config["gates"]["normal_vs_no_nack"]
            nack_pass = bool(
                vs_no_nack["clean_delta_percentage_points_ci"][0] >= float(nack_gate["minimum_clean_delta_ci_lower_percentage_points"])
                and vs_no_nack["relative_bit_change_percent_mean"] <= float(nack_gate["maximum_mean_bit_increase_percent"])
            )
            value_gate = config["gates"]["normal_vs_causal_fixed10"]
            value_pass = bool(
                vs_fixed["clean_delta_percentage_points_ci"][0] >= float(value_gate["minimum_clean_delta_ci_lower_percentage_points"])
                and -vs_fixed["relative_bit_change_percent_ci"][1] > float(value_gate["minimum_relative_bit_saving_ci_lower_percent"])
            )
            gate_type = "normal"
        else:
            nack_gate = config["gates"]["emitter_failure_vs_no_nack"]
            nack_pass = bool(
                vs_no_nack["clean_delta_percentage_points_ci"][0] >= float(nack_gate["minimum_clean_delta_ci_lower_percentage_points"])
                and vs_no_nack["relative_bit_change_percent_mean"] <= float(nack_gate["maximum_mean_bit_increase_percent"])
            )
            value_gate = config["gates"]["emitter_failure_vs_causal_boot_event"]
            value_pass = bool(
                vs_event["clean_delta_percentage_points_ci"][0] >= float(value_gate["minimum_clean_delta_ci_lower_percentage_points"])
                and vs_event["relative_bit_change_percent_mean"] <= float(value_gate["maximum_mean_bit_increase_percent"])
            )
            gate_type = "emitter_failure"
        profiles[name] = {
            "policies": summaries,
            "candidate_minus_no_nack": vs_no_nack,
            "candidate_minus_causal_fixed10": vs_fixed,
            "candidate_minus_causal_boot_event": vs_event,
            "gate_type": gate_type,
            "causal_safety_passed": global_pass,
            "reset_nack_gate_passed": nack_pass,
            "value_gate_passed": value_pass,
            "profile_gate_passed": bool(global_pass and nack_pass and value_pass),
        }

    normal_nack = sum(v["reset_nack_gate_passed"] for n, v in profiles.items() if n.startswith("normal_"))
    failure_nack = sum(v["reset_nack_gate_passed"] for n, v in profiles.items() if not n.startswith("normal_"))
    normal_value = sum(v["value_gate_passed"] for n, v in profiles.items() if n.startswith("normal_"))
    failure_value = sum(v["value_gate_passed"] for n, v in profiles.items() if not n.startswith("normal_"))
    primary_gate = {
        "normal_nack_passing": int(normal_nack), "normal_nack_required": 2,
        "emitter_failure_nack_passing": int(failure_nack), "emitter_failure_nack_required": 2,
        "normal_value_passing": int(normal_value), "normal_value_required": 2,
        "emitter_failure_value_passing": int(failure_value), "emitter_failure_value_required": 2,
        "causal_safety_all_profiles": bool(all(v["causal_safety_passed"] for v in profiles.values())),
    }
    primary_gate["passed"] = bool(
        primary_gate["normal_nack_passing"] >= 2
        and primary_gate["emitter_failure_nack_passing"] >= 2
        and primary_gate["normal_value_passing"] >= 2
        and primary_gate["emitter_failure_value_passing"] >= 2
        and primary_gate["causal_safety_all_profiles"]
    )
    result = {
        "version": "1.0", "experiment_id": config["experiment_id"],
        "candidate": config["candidate"], "mode": args.mode,
        "status": f"stage9_5_{args.mode}_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes, "protected_external_data_read": False,
        "paired_common_random_numbers": True, "profiles": profiles,
        "primary_gate": primary_gate, "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(primary_gate, ensure_ascii=False))
    if not primary_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
