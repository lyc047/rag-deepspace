#!/usr/bin/env python
"""Run paired Stage-6 silence-heartbeat sensitivity experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_context_recovery_development import (  # noqa: E402
    simulate_trajectory,
    summarize_trajectories,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    decode_context_install,
    encode_context_install,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


_REPRODUCTION_KEYS = (
    "mean_application_bits_per_scene",
    "mean_forward_bits_per_scene",
    "mean_ack_bits_per_scene",
    "mean_context_install_count",
    "mean_compact_update_count",
    "mean_ack_frame_count",
    "mean_stale_ack_rejection_count",
    "mean_rejected_compact_count",
    "wrong_codebook_decode_count",
    "maximum_belief_size",
    "availability_rate",
    "clean_rate",
    "risk_violation_rate",
    "conditional_mean_regret_db",
    "conditional_maximum_regret_db",
    "effective_mean_regret_db",
    "effective_cvar_0_9_regret_db",
)


def _comparison(candidate: dict, baseline: dict) -> dict:
    baseline_bits = float(baseline["mean_application_bits_per_scene"])
    return {
        "clean_rate_improvement_percentage_points": float(
            100.0 * (candidate["clean_rate"] - baseline["clean_rate"])
        ),
        "availability_improvement_percentage_points": float(
            100.0
            * (
                candidate["availability_rate"]
                - baseline["availability_rate"]
            )
        ),
        "application_bit_increase_pct": float(
            100.0
            * (
                candidate["mean_application_bits_per_scene"]
                - baseline_bits
            )
            / baseline_bits
        ),
        "effective_cvar_change_db": float(
            candidate["effective_cvar_0_9_regret_db"]
            - baseline["effective_cvar_0_9_regret_db"]
        ),
        "context_install_count_change": float(
            candidate["mean_context_install_count"]
            - baseline["mean_context_install_count"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_context_heartbeat_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/context_heartbeat_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "context_heartbeat_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 heartbeat result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        not governance["faults_are_controlled_injections_not_measurements"]
        or governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("Stage-6 heartbeat governance is invalid")

    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("Stage-6 heartbeat cache hash mismatch")
    predecessor_spec = protocol["predecessor_result"]
    predecessor_path = PROJECT_DIR / predecessor_spec["path"]
    if sha256_file(predecessor_path) != predecessor_spec["sha256"]:
        raise ValueError("Stage-6 predecessor result hash mismatch")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))

    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train, evaluation, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    trajectory_count = int(protocol["monte_carlo"]["trajectories"])
    base_seed = int(protocol["monte_carlo"]["seed"])
    seed_offset = int(
        protocol["monte_carlo"]["predecessor_condition_seed_offset"]
    )
    ack_bits = cumulative_ack_payload_bits()
    request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=int(grid["codebook_epoch"]),
            expected_update_epoch=0,
        ).size
    )
    response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=int(grid["codebook_epoch"]),
            current_update_epoch=0,
        ).size
    )
    if (
        ack_bits != int(grid["ack_application_bits"])
        or request_bits
        != int(grid["heartbeat_request_application_bits"])
        or response_bits
        != int(grid["heartbeat_response_application_bits"])
    ):
        raise ValueError("declared application bits do not match real codecs")

    all_results = {}
    reproduction_checks = {}
    for n_position, n_value in enumerate(grid["n_channels"]):
        n_channels = int(n_value)
        demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
        queries = tuple(SpectrumTaskQuery(demand) for demand in demands)
        states = [
            build_task_state(
                values,
                queries,
                epsilon_db=float(grid["epsilon_db"]),
            )
            for values in cache[f"channel_power_n{n_channels}"]
        ]
        codebook = fit_greedy_task_codebook(
            [states[int(index)] for index in train]
        )
        sender_session = install_codebook(
            codebook, epoch=int(grid["codebook_epoch"])
        )
        install_packet = encode_context_install(sender_session, node_id=1)
        receiver_session = decode_context_install(install_packet).session

        rng = np.random.default_rng(
            base_seed + n_position * 100_000 + seed_offset
        )
        predecessor_random = rng.random(
            (trajectory_count, len(evaluation), 6),
            dtype=np.float64,
        )
        heartbeat_random = rng.random(
            (trajectory_count, len(evaluation), 2),
            dtype=np.float64,
        )
        common_random = np.concatenate(
            [predecessor_random, heartbeat_random], axis=2
        )

        policy_results = {}
        for policy in protocol["heartbeat_policies"]:
            runs = [
                simulate_trajectory(
                    states,
                    cache["timestamps_local"],
                    evaluation,
                    sender_session=sender_session,
                    receiver_session=receiver_session,
                    install_packet=install_packet,
                    method="belief_risk_recovery",
                    condition=protocol["fault_condition"],
                    random_values=common_random[trajectory],
                    epsilon_db=float(grid["epsilon_db"]),
                    max_age_minutes=float(
                        grid["max_state_age_minutes"]
                    ),
                    ack_bits=ack_bits,
                    outage_penalty_db=float(grid["outage_penalty_db"]),
                    heartbeat_interval_scenes=policy[
                        "silence_interval_scenes"
                    ],
                    heartbeat_request_bits=request_bits,
                    heartbeat_response_bits=response_bits,
                )
                for trajectory in range(trajectory_count)
            ]
            policy_results[policy["name"]] = summarize_trajectories(
                runs, scene_count=len(evaluation)
            )

        baseline = policy_results["no_heartbeat"]
        comparisons = {
            name: _comparison(value, baseline)
            for name, value in policy_results.items()
            if name != "no_heartbeat"
        }
        old = predecessor["results"][str(n_channels)]["conditions"][
            predecessor_spec["reproduced_condition"]
        ]["methods"][predecessor_spec["reproduced_method"]]
        reproduction_checks[str(n_channels)] = all(
            baseline[key] == old[key] for key in _REPRODUCTION_KEYS
        )
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "context_install_bits": int(install_packet.size),
            "policies": policy_results,
            "comparisons_vs_no_heartbeat": comparisons,
        }

    checks = {
        "no_heartbeat_reproduces_predecessor": all(
            reproduction_checks.values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            policy["wrong_codebook_decode_count"] == 0
            for n_result in all_results.values()
            for policy in n_result["policies"].values()
        ),
        "at_least_one_interval_improves_clean_rate_for_every_n": all(
            any(
                comparison[
                    "clean_rate_improvement_percentage_points"
                ]
                > 0.0
                for comparison in n_result[
                    "comparisons_vs_no_heartbeat"
                ].values()
            )
            for n_result in all_results.values()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Stage-6 heartbeat gate failed: {checks}")
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks["final_access_state_unchanged"] = access_before == access_after
    checks["final_access_count_remains_zero"] = (
        access_after["access_count"] == 0
        and not access_after["final_signal_values_accessed"]
        and not access_after["final_method_outputs_accessed"]
    )
    if not all(checks.values()):
        raise AssertionError(f"Stage-6 heartbeat governance failed: {checks}")

    result = {
        "version": "1.0",
        "status": "stage6_context_heartbeat_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_sha256": source["cache_sha256"],
            "predecessor_result_sha256": predecessor_spec["sha256"],
            "train_scene_count": int(train.size),
            "evaluation_scene_count": int(evaluation.size),
            "train_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
        },
        "results": all_results,
        "checks": checks,
        "reproduction_checks_by_n": reproduction_checks,
        "governance": {
            "development_only": True,
            "faults_are_controlled_injections_not_measurements": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "checks": checks,
                "comparisons": {
                    n: value["comparisons_vs_no_heartbeat"]
                    for n, value in all_results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
