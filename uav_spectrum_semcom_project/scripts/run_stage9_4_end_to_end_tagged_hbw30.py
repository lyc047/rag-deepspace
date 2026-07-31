#!/usr/bin/env python
"""Run preregistered Stage-9.4 tagged HBW30 integration experiments."""

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
    generate_hybrid_fault_schedule,
    simulate_hybrid_trajectory,
)
from spectrum_semcom.stage9_4_tagged_hybrid_integration import (  # noqa: E402
    simulate_tagged_hbw30_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import (  # noqa: E402
    SilProtocolBits,
    generate_task_trace,
)


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
        "wrong_context_execution_count",
        "stale_restore_acceptance_count",
        "restore_frame_count",
        "identity_sync_wait_scenes",
        "recovery_tag_bits",
    ):
        if metric in rows[0]:
            output[metric] = int(sum(row[metric] for row in rows))
    output["component_bits_per_scene"] = {
        name: float(np.mean([row[name] / row["scene_count"] for row in rows]))
        for name in (
            "heartbeat_bits",
            "boot_status_bits",
            "install_bits",
            "update_bits",
            "ack_bits",
        )
    }
    return output


def _paired(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clean = np.asarray(
        [100.0 * (a["clean_rate"] - b["clean_rate"]) for a, b in zip(rows_a, rows_b)]
    )
    availability = np.asarray(
        [100.0 * (a["availability"] - b["availability"]) for a, b in zip(rows_a, rows_b)]
    )
    bit_change = np.asarray(
        [
            100.0 * (a["total_bits"] - b["total_bits"]) / b["total_bits"]
            for a, b in zip(rows_a, rows_b)
        ]
    )
    return {
        "clean_delta_percentage_points_mean": float(np.mean(clean)),
        "clean_delta_percentage_points_ci": _ci(clean, replicates, seed),
        "availability_delta_percentage_points_mean": float(np.mean(availability)),
        "availability_delta_percentage_points_ci": _ci(
            availability, replicates, seed + 1
        ),
        "relative_bit_change_percent_mean": float(np.mean(bit_change)),
        "relative_bit_change_percent_ci": _ci(bit_change, replicates, seed + 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "--config",
        default="configs/stage9_4_end_to_end_tagged_hbw30_v1.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freeze",
        default="configs/stage9_4_end_to_end_tagged_hbw30_freeze_v1.json",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_protocol_config"]
    if _sha256(parent_path) != config["parent_protocol_config_sha256"]:
        raise RuntimeError("frozen Stage-9.1 parent configuration changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    frozen_paths = {
        "config": config_path,
        "parent_config": parent_path,
        "stage9_1_runtime": ROOT / "src/spectrum_semcom/stage9_1_hybrid_recovery.py",
        "stage9_3_runtime": ROOT / "src/spectrum_semcom/stage9_3_tagged_recovery.py",
        "integration_runtime": ROOT / "src/spectrum_semcom/stage9_4_tagged_hybrid_integration.py",
        "runner": Path(__file__).resolve(),
        "tests": ROOT / "tests/test_stage9_4_tagged_hybrid_integration.py",
    }
    hashes = {name: _sha256(path) for name, path in frozen_paths.items()}
    if args.mode == "confirmation":
        freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
        if freeze["status"] != "stage9_4_candidate_frozen_before_confirmation":
            raise RuntimeError("Stage-9.4 freeze is not active")
        if freeze["input_hashes"] != hashes:
            raise RuntimeError("Stage-9.4 inputs differ from freeze")

    p = parent["protocol_bits"]
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
    task = parent["task_trace"]
    profiles: dict[str, Any] = {}
    all_accounting_exact = True
    all_shadow_outcomes_exact = True

    for profile_index, (name, profile) in enumerate(parent["fault_profiles"].items()):
        rows: dict[str, list[dict[str, Any]]] = {
            "tagged_hbw30": [],
            "zero_tag_shadow": [],
            "legacy_hbw30": [],
            "legacy_fixed10": [],
            "legacy_boot_event": [],
        }
        accounting_checks: list[bool] = []
        shadow_outcome_checks: list[bool] = []
        for trajectory in range(trajectories):
            sequence = np.random.SeedSequence([seed, profile_index, trajectory])
            trace_rng, fault_rng = [
                np.random.default_rng(item) for item in sequence.spawn(2)
            ]
            trace = generate_task_trace(
                scene_count=int(task["scene_count"]),
                symbol_count=int(task["semantic_symbol_count"]),
                change_probability=float(task["symbol_change_probability"]),
                equivalence_probability=float(task["temporary_equivalence_probability"]),
                equivalence_scenes=int(task["temporary_equivalence_scenes"]),
                rng=trace_rng,
            )
            values = dict(profile)
            emitter_failure = float(
                values.pop("boot_emitter_failure_probability_per_boot")
            )
            faults = generate_hybrid_fault_schedule(
                scene_count=int(task["scene_count"]),
                boot_emitter_failure_probability_per_boot=emitter_failure,
                rng=fault_rng,
                **{key: float(value) for key, value in values.items()},
            )
            tagged = simulate_tagged_hbw30_trajectory(
                trace, faults, bits=bits, restore_tag_bits=16
            ).to_dict()
            shadow = simulate_tagged_hbw30_trajectory(
                trace, faults, bits=bits, restore_tag_bits=0
            ).to_dict()
            legacy_hbw30 = simulate_hybrid_trajectory(
                trace, faults, policy="boot_event_watchdog30", bits=bits
            ).to_dict()
            fixed10 = simulate_hybrid_trajectory(
                trace, faults, policy="fixed10", bits=bits
            ).to_dict()
            boot_event = simulate_hybrid_trajectory(
                trace, faults, policy="boot_event", bits=bits
            ).to_dict()
            rows["tagged_hbw30"].append(tagged)
            rows["zero_tag_shadow"].append(shadow)
            rows["legacy_hbw30"].append(legacy_hbw30)
            rows["legacy_fixed10"].append(fixed10)
            rows["legacy_boot_event"].append(boot_event)
            accounting_checks.append(
                tagged["total_bits"] - shadow["total_bits"]
                == 16 * tagged["restore_frame_count"]
                and tagged["recovery_tag_bits"]
                == 16 * tagged["restore_frame_count"]
                and tagged["update_bits"] == shadow["update_bits"]
                and tagged["ack_bits"] == shadow["ack_bits"]
            )
            shadow_outcome_checks.append(
                tagged["clean_count"] == shadow["clean_count"]
                and tagged["available_count"] == shadow["available_count"]
                and tagged["recovery_latencies"] == shadow["recovery_latencies"]
            )

        summaries = {
            policy: _summary(
                policy_rows,
                replicates,
                seed + 100 * profile_index + policy_index,
            )
            for policy_index, (policy, policy_rows) in enumerate(rows.items())
        }
        versus_legacy = _paired(
            rows["tagged_hbw30"],
            rows["legacy_hbw30"],
            replicates,
            seed + 1000 + profile_index,
        )
        versus_fixed = _paired(
            rows["tagged_hbw30"],
            rows["legacy_fixed10"],
            replicates,
            seed + 2000 + profile_index,
        )
        versus_event = _paired(
            rows["tagged_hbw30"],
            rows["legacy_boot_event"],
            replicates,
            seed + 3000 + profile_index,
        )
        integration_gate = config["gates"]["integration_noninferiority"]
        integration_passed = bool(
            versus_legacy["clean_delta_percentage_points_mean"]
            >= float(integration_gate["minimum_clean_delta_percentage_points"])
            and versus_legacy["availability_delta_percentage_points_mean"]
            >= float(integration_gate["minimum_availability_delta_percentage_points"])
        )
        value_gate = config["gates"]["preserve_stage9_1_value"]
        if name.startswith("normal_"):
            value_passed = bool(
                versus_fixed["clean_delta_percentage_points_ci"][0]
                >= float(value_gate["normal_profiles_clean_delta_ci_lower_percentage_points"])
                and -versus_fixed["relative_bit_change_percent_ci"][1]
                > float(value_gate["normal_profiles_relative_bit_saving_ci_lower_percent"])
            )
            value_gate_type = "normal_versus_legacy_fixed10"
        else:
            value_passed = bool(
                versus_event["clean_delta_percentage_points_ci"][0]
                > float(value_gate["emitter_failure_clean_delta_ci_lower_percentage_points"])
                and versus_event["relative_bit_change_percent_mean"]
                <= float(value_gate["emitter_failure_maximum_mean_bit_increase_percent"])
            )
            value_gate_type = "emitter_failure_versus_legacy_boot_event"
        safety_passed = bool(
            summaries["tagged_hbw30"]["wrong_context_execution_count"] == 0
            and summaries["tagged_hbw30"]["stale_restore_acceptance_count"] == 0
        )
        accounting_exact = all(accounting_checks)
        shadow_outcomes_exact = all(shadow_outcome_checks)
        all_accounting_exact &= accounting_exact
        all_shadow_outcomes_exact &= shadow_outcomes_exact
        profiles[name] = {
            "policies": summaries,
            "tagged_minus_legacy_hbw30": versus_legacy,
            "tagged_minus_legacy_fixed10": versus_fixed,
            "tagged_minus_legacy_boot_event": versus_event,
            "accounting_exact_every_trajectory": accounting_exact,
            "shadow_outcomes_exact_every_trajectory": shadow_outcomes_exact,
            "integration_noninferiority_passed": integration_passed,
            "value_gate_type": value_gate_type,
            "stage9_1_value_preserved": value_passed,
            "safety_passed": safety_passed,
            "profile_gate_passed": bool(
                accounting_exact
                and shadow_outcomes_exact
                and integration_passed
                and value_passed
                and safety_passed
            ),
        }

    integration_passing = sum(
        profile["integration_noninferiority_passed"] for profile in profiles.values()
    )
    normal_value_passing = sum(
        profile["stage9_1_value_preserved"]
        for name, profile in profiles.items()
        if name.startswith("normal_")
    )
    failure_value_passing = sum(
        profile["stage9_1_value_preserved"]
        for name, profile in profiles.items()
        if not name.startswith("normal_")
    )
    value_gate = config["gates"]["preserve_stage9_1_value"]
    primary_gate = {
        "accounting_exact": bool(all_accounting_exact),
        "shadow_outcomes_exact": bool(all_shadow_outcomes_exact),
        "integration_noninferiority_passing": int(integration_passing),
        "integration_noninferiority_required": int(
            config["gates"]["integration_noninferiority"]["required_profiles"]
        ),
        "normal_value_passing": int(normal_value_passing),
        "normal_value_required": int(value_gate["normal_profiles_required"]),
        "emitter_failure_value_passing": int(failure_value_passing),
        "emitter_failure_value_required": int(
            value_gate["emitter_failure_profiles_required"]
        ),
    }
    primary_gate["passed"] = bool(
        primary_gate["accounting_exact"]
        and primary_gate["shadow_outcomes_exact"]
        and primary_gate["integration_noninferiority_passing"]
        >= primary_gate["integration_noninferiority_required"]
        and primary_gate["normal_value_passing"]
        >= primary_gate["normal_value_required"]
        and primary_gate["emitter_failure_value_passing"]
        >= primary_gate["emitter_failure_value_required"]
        and all(profile["safety_passed"] for profile in profiles.values())
    )
    result = {
        "version": "1.0",
        "experiment_id": config["experiment_id"],
        "candidate": config["candidate"],
        "mode": args.mode,
        "status": f"stage9_4_{args.mode}_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "protected_external_data_read": False,
        "paired_common_random_numbers": True,
        "profiles": profiles,
        "primary_gate": primary_gate,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(primary_gate, ensure_ascii=False))
    if not primary_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

