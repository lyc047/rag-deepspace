#!/usr/bin/env python
"""Run preregistered Stage-9.1 hybrid watchdog experiments."""

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

from spectrum_semcom.stage9_1_hybrid_recovery import (  # noqa: E402
    POLICIES,
    generate_hybrid_fault_schedule,
    simulate_hybrid_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import (  # noqa: E402
    SilProtocolBits,
    generate_task_trace,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(replicates, data.size))
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
    output["wrong_context_execution_count"] = int(
        sum(row["wrong_context_execution_count"] for row in rows)
    )
    output["component_bits_per_scene"] = {
        name: float(np.mean([row[name] / row["scene_count"] for row in rows]))
        for name in ("heartbeat_bits", "boot_status_bits", "install_bits", "update_bits", "ack_bits")
    }
    return output


def _paired(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    clean = np.asarray(
        [100.0 * (a["clean_rate"] - b["clean_rate"]) for a, b in zip(rows_a, rows_b)]
    )
    bit_change = np.asarray(
        [100.0 * (a["total_bits"] - b["total_bits"]) / b["total_bits"] for a, b in zip(rows_a, rows_b)]
    )
    return {
        "clean_delta_percentage_points_mean": float(np.mean(clean)),
        "clean_delta_percentage_points_ci": _ci(clean, replicates, seed),
        "relative_bit_change_percent_mean": float(np.mean(bit_change)),
        "relative_bit_change_percent_ci": _ci(bit_change, replicates, seed + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "--config", default="configs/stage9_1_hybrid_watchdog_protocol_v1.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freeze", default="results/stage9_1/stage9_1_candidate_freeze_v1.json"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    process_result_path = ROOT / "results/stage9_1/process_fault_conformance_v1/result.json"
    process_result = json.loads(process_result_path.read_text(encoding="utf-8"))
    if process_result["status"] != "PASS":
        raise RuntimeError("process fault conformance gate failed")
    paths = {
        "config": config_path,
        "runtime": ROOT / "src/spectrum_semcom/stage9_1_hybrid_recovery.py",
        "runner": Path(__file__).resolve(),
        "worker": ROOT / "scripts/stage9_1_receiver_worker.py",
        "tests": ROOT / "tests/test_stage9_1_hybrid_recovery.py",
        "process_conformance": process_result_path,
    }
    hashes = {name: _hash(path) for name, path in paths.items()}
    if args.mode == "confirmation":
        freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
        if freeze["status"] != "stage9_1_candidate_frozen_before_confirmation":
            raise RuntimeError("Stage-9.1 freeze is not active")
        if freeze["input_hashes"] != hashes:
            raise RuntimeError("Stage-9.1 inputs differ from freeze")

    p = config["protocol_bits"]
    bits = SilProtocolBits(
        int(p["compact_update_bits"]),
        int(p["cumulative_ack_bits"]),
        int(p["heartbeat_request_bits"]),
        int(p["heartbeat_response_bits"]),
        int(p["boot_status_bits"]),
        int(p["preinstalled_activation_bits"]),
        int(p["exact_action_bits"]),
        int(p["full_install_bits"]),
    )
    mode = config[args.mode]
    trajectories = int(mode["trajectories_per_profile"])
    replicates = int(mode["bootstrap_replicates"])
    seed = int(mode["seed"])
    task = config["task_trace"]
    profiles: dict[str, Any] = {}

    for profile_index, (name, profile) in enumerate(config["fault_profiles"].items()):
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
            profile_values = dict(profile)
            emitter_failure = float(profile_values.pop("boot_emitter_failure_probability_per_boot"))
            faults = generate_hybrid_fault_schedule(
                scene_count=int(task["scene_count"]),
                boot_emitter_failure_probability_per_boot=emitter_failure,
                rng=fault_rng,
                **{key: float(value) for key, value in profile_values.items()},
            )
            for policy in POLICIES:
                rows[policy].append(
                    simulate_hybrid_trajectory(trace, faults, policy=policy, bits=bits).to_dict()
                )

        summaries = {
            policy: _summary(values, replicates, seed + 100 * profile_index + index)
            for index, (policy, values) in enumerate(rows.items())
        }
        versus_fixed = _paired(
            rows["boot_event_watchdog30"], rows["fixed10"], replicates, seed + 1000 + profile_index
        )
        versus_event = _paired(
            rows["boot_event_watchdog30"], rows["boot_event"], replicates, seed + 2000 + profile_index
        )
        normal = name.startswith("normal_")
        if normal:
            gate = config["gates"]["normal_profiles"]
            passed = bool(
                versus_fixed["clean_delta_percentage_points_ci"][0]
                >= float(gate["clean_delta_ci_lower_percentage_points"])
                and -versus_fixed["relative_bit_change_percent_ci"][1]
                > float(gate["relative_bit_saving_ci_lower_percent"])
            )
            gate_type = "normal_versus_fixed10"
        else:
            gate = config["gates"]["emitter_failure_profiles"]
            passed = bool(
                versus_event["clean_delta_percentage_points_ci"][0]
                > float(gate["clean_delta_ci_lower_percentage_points"])
                and versus_event["relative_bit_change_percent_mean"]
                <= float(gate["maximum_mean_bit_increase_percent"])
            )
            gate_type = "emitter_failure_versus_boot_event"
        latency_ok = bool(
            summaries["boot_event_watchdog30"]["recovery_latency_p95_scenes"]
            <= summaries["boot_event_watchdog60"]["recovery_latency_p95_scenes"]
        )
        safety_ok = all(
            summary["wrong_context_execution_count"] == 0 for summary in summaries.values()
        )
        profiles[name] = {
            "policies": summaries,
            "watchdog30_minus_fixed10": versus_fixed,
            "watchdog30_minus_boot_event": versus_event,
            "gate_type": gate_type,
            "performance_gate_passed": passed,
            "latency_gate_passed": latency_ok,
            "safety_gate_passed": safety_ok,
            "profile_gate_passed": bool(passed and latency_ok and safety_ok),
        }

    normal_pass = sum(
        value["profile_gate_passed"] for name, value in profiles.items() if name.startswith("normal_")
    )
    failure_pass = sum(
        value["profile_gate_passed"] for name, value in profiles.items() if not name.startswith("normal_")
    )
    normal_required = int(config["gates"]["normal_profiles"]["required_profiles"])
    failure_required = int(config["gates"]["emitter_failure_profiles"]["required_profiles"])
    result = {
        "version": "1.0",
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "status": f"stage9_1_{args.mode}_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "protected_external_data_read": False,
        "paired_common_random_numbers": True,
        "profiles": profiles,
        "primary_gate": {
            "normal_passing": int(normal_pass),
            "normal_required": normal_required,
            "emitter_failure_passing": int(failure_pass),
            "emitter_failure_required": failure_required,
            "passed": bool(normal_pass >= normal_required and failure_pass >= failure_required),
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["primary_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
