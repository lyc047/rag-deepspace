#!/usr/bin/env python
"""Run the preregistered Stage-9 software-in-loop protocol experiment."""

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

from spectrum_semcom.stage9_sil_runtime import (  # noqa: E402
    POLICIES,
    SilProtocolBits,
    generate_fault_schedule,
    generate_task_trace,
    simulate_sil_trajectory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, samples.size, size=(int(replicates), samples.size))
    means = np.mean(samples[draws], axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def _summary(
    values: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {"trajectory_count": len(values)}
    for metric in ("bits_per_scene", "clean_rate", "availability"):
        array = np.asarray([row[metric] for row in values], dtype=np.float64)
        lower, upper = _bootstrap_ci(
            array, replicates=replicates, seed=seed + len(output)
        )
        output[metric] = {
            "mean": float(np.mean(array)),
            "ci_lower": lower,
            "ci_upper": upper,
        }
    output["wrong_context_execution_count"] = int(
        sum(row["wrong_context_execution_count"] for row in values)
    )
    output["mean_component_bits_per_scene"] = {
        name: float(np.mean([row[name] / row["scene_count"] for row in values]))
        for name in (
            "heartbeat_bits",
            "boot_status_bits",
            "install_bits",
            "update_bits",
            "ack_bits",
        )
    }
    latencies = [
        latency for row in values for latency in row["recovery_latencies"]
    ]
    output["recovery_event_count"] = len(latencies)
    output["recovery_latency_p95_scenes"] = (
        float(np.quantile(latencies, 0.95)) if latencies else None
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "--config",
        default="configs/stage9_software_in_loop_protocol_v1.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freeze",
        default="results/stage9/stage9_candidate_freeze_v1.json",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode_config = config[args.mode]
    source_paths = {
        "config": config_path,
        "runtime": ROOT / "src/spectrum_semcom/stage9_sil_runtime.py",
        "runner": Path(__file__).resolve(),
        "worker": ROOT / "scripts/stage9_receiver_worker.py",
        "tests": ROOT / "tests/test_stage9_sil_runtime.py",
    }
    hashes = {name: _sha256(path) for name, path in source_paths.items()}
    if args.mode == "confirmation":
        freeze_path = ROOT / args.freeze
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["status"] != "stage9_candidate_frozen_before_confirmation":
            raise RuntimeError("Stage-9 confirmation freeze is not active")
        if freeze["input_hashes"] != hashes:
            raise RuntimeError("Stage-9 confirmation inputs differ from freeze")

    protocol = config["protocol_bits"]
    bits = SilProtocolBits(
        compact_update_bits=int(protocol["compact_update_bits"]),
        cumulative_ack_bits=int(protocol["cumulative_ack_bits"]),
        heartbeat_request_bits=int(protocol["heartbeat_request_bits"]),
        heartbeat_response_bits=int(protocol["heartbeat_response_bits"]),
        boot_status_bits=int(protocol["boot_status_bits"]),
        preinstalled_activation_bits=int(protocol["preinstalled_activation_bits"]),
        exact_action_bits=int(
            protocol["exact_action_header_bits"]
            + protocol["exact_action_payload_bits"]
        ),
        full_install_bits=int(protocol["full_install_bits"]),
    )
    task = config["task_trace"]
    profile_results: dict[str, Any] = {}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seed = int(mode_config["seed"])
    trajectory_count = int(mode_config["trajectories_per_profile"])
    replicates = int(mode_config["bootstrap_replicates"])

    for profile_index, (profile_name, profile) in enumerate(
        config["fault_profiles"].items()
    ):
        rows = {policy: [] for policy in POLICIES}
        for trajectory in range(trajectory_count):
            sequence = np.random.SeedSequence(
                [seed, profile_index, trajectory]
            )
            trace_rng, fault_rng = [
                np.random.default_rng(item) for item in sequence.spawn(2)
            ]
            trace = generate_task_trace(
                scene_count=int(task["scene_count"]),
                symbol_count=int(task["semantic_symbol_count"]),
                change_probability=float(task["symbol_change_probability"]),
                equivalence_probability=float(
                    task["temporary_equivalence_probability"]
                ),
                equivalence_scenes=int(task["temporary_equivalence_scenes"]),
                rng=trace_rng,
            )
            faults = generate_fault_schedule(
                scene_count=int(task["scene_count"]),
                rng=fault_rng,
                **{name: float(value) for name, value in profile.items()},
            )
            for policy in POLICIES:
                rows[policy].append(
                    simulate_sil_trajectory(
                        trace,
                        faults,
                        policy=policy,
                        bits=bits,
                    ).to_dict()
                )
        all_rows[profile_name] = rows
        summaries = {
            policy: _summary(
                values,
                replicates=replicates,
                seed=seed + 1000 * profile_index + 10 * index,
            )
            for index, (policy, values) in enumerate(rows.items())
        }
        fixed = rows["fixed10"]
        event = rows["boot_event"]
        clean_delta_pp = np.asarray(
            [
                100.0 * (event_row["clean_rate"] - fixed_row["clean_rate"])
                for fixed_row, event_row in zip(fixed, event)
            ]
        )
        bit_saving = np.asarray(
            [
                100.0
                * (fixed_row["total_bits"] - event_row["total_bits"])
                / fixed_row["total_bits"]
                for fixed_row, event_row in zip(fixed, event)
            ]
        )
        clean_ci = _bootstrap_ci(
            clean_delta_pp, replicates=replicates, seed=seed + 5000 + profile_index
        )
        saving_ci = _bootstrap_ci(
            bit_saving, replicates=replicates, seed=seed + 6000 + profile_index
        )
        fixed_latency = summaries["fixed10"]["recovery_latency_p95_scenes"]
        event_latency = summaries["boot_event"]["recovery_latency_p95_scenes"]
        gate = config["primary_gate"]
        profile_pass = bool(
            clean_ci[0] >= float(gate["clean_rate_margin_percentage_points"])
            and saving_ci[0] > float(gate["minimum_relative_bit_saving_percent"])
            and summaries["boot_event"]["wrong_context_execution_count"]
            == int(gate["wrong_context_execution_count"])
            and event_latency is not None
            and fixed_latency is not None
            and event_latency <= fixed_latency
        )
        profile_results[profile_name] = {
            "policies": summaries,
            "paired_boot_event_minus_fixed10": {
                "clean_delta_percentage_points_mean": float(
                    np.mean(clean_delta_pp)
                ),
                "clean_delta_percentage_points_ci": list(clean_ci),
                "relative_bit_saving_percent_mean": float(np.mean(bit_saving)),
                "relative_bit_saving_percent_ci": list(saving_ci),
                "profile_gate_pass": profile_pass,
            },
        }

    passing = sum(
        result["paired_boot_event_minus_fixed10"]["profile_gate_pass"]
        for result in profile_results.values()
    )
    result = {
        "version": "1.0",
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "status": f"stage9_{args.mode}_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "protected_external_data_read": False,
        "paired_common_random_numbers": True,
        "profiles": profile_results,
        "primary_gate": {
            "passing_profile_count": int(passing),
            "required_profile_count": int(
                config["primary_gate"]["minimum_profiles_passing"]
            ),
            "passed": bool(
                passing
                >= int(config["primary_gate"]["minimum_profiles_passing"])
            ),
        },
        "claim_boundary": config["protected_data_policy"]["claim_boundary"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["primary_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
