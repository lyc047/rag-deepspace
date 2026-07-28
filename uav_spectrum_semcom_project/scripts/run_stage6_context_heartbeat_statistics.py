#!/usr/bin/env python
"""Compute paired trajectory-bootstrap intervals for Stage-6 heartbeats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_context_recovery_development import simulate_trajectory  # noqa: E402
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
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


def _trajectory_metrics(run, scene_count: int) -> dict[str, float]:
    return {
        "clean_rate_percentage_points": float(
            100.0 * np.mean(run.clean)
        ),
        "availability_rate_percentage_points": float(
            100.0 * np.mean(run.available)
        ),
        "application_bits_per_scene": float(
            run.total_application_bits / scene_count
        ),
    }


def _paired_interval(
    differences: np.ndarray,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict:
    trajectory_count = int(differences.size)
    indices = rng.integers(
        0,
        trajectory_count,
        size=(bootstrap_replicates, trajectory_count),
    )
    bootstrap = np.mean(differences[indices], axis=1)
    alpha = 1.0 - confidence_level
    return {
        "mean_paired_difference": float(np.mean(differences)),
        "confidence_interval_lower": float(
            np.quantile(bootstrap, alpha / 2.0)
        ),
        "confidence_interval_upper": float(
            np.quantile(bootstrap, 1.0 - alpha / 2.0)
        ),
        "trajectory_standard_deviation": float(
            np.std(differences, ddof=1)
        ),
        "trajectory_count": trajectory_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_context_heartbeat_statistics_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/context_heartbeat_statistics_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "context_heartbeat_statistics.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite heartbeat statistics")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    protocol_spec = analysis["heartbeat_protocol"]
    aggregate_spec = analysis["frozen_aggregate_result"]
    protocol_path = PROJECT_DIR / protocol_spec["path"]
    aggregate_path = PROJECT_DIR / aggregate_spec["path"]
    if (
        sha256_file(protocol_path) != protocol_spec["sha256"]
        or sha256_file(aggregate_path) != aggregate_spec["sha256"]
    ):
        raise ValueError("heartbeat statistical inputs are not hash locked")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))

    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("heartbeat statistics cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train, evaluation, _, _ = grouped_train_evaluation_indices(
        cache[split["group_field"]],
        seed=int(split["seed"]),
        train_fraction=float(split["train_group_fraction"]),
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    trajectory_count = int(protocol["monte_carlo"]["trajectories"])
    base_seed = int(protocol["monte_carlo"]["seed"])
    seed_offset = int(
        protocol["monte_carlo"]["predecessor_condition_seed_offset"]
    )
    ack_bits = cumulative_ack_payload_bits()
    request_bits = int(grid["heartbeat_request_application_bits"])
    response_bits = int(grid["heartbeat_response_application_bits"])
    resampling = analysis["resampling"]
    bootstrap_replicates = int(resampling["bootstrap_replicates"])
    confidence_level = float(resampling["confidence_level"])

    results = {}
    aggregate_reproduced = {}
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
        rng_fault = np.random.default_rng(
            base_seed + n_position * 100_000 + seed_offset
        )
        old_random = rng_fault.random(
            (trajectory_count, len(evaluation), 6)
        )
        probe_random = rng_fault.random(
            (trajectory_count, len(evaluation), 2)
        )
        common_random = np.concatenate(
            [old_random, probe_random], axis=2
        )
        policy_runs = {}
        for policy in protocol["heartbeat_policies"]:
            policy_runs[policy["name"]] = [
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

        baseline_metrics = [
            _trajectory_metrics(run, len(evaluation))
            for run in policy_runs["no_heartbeat"]
        ]
        n_results = {}
        for policy_position, policy in enumerate(
            protocol["heartbeat_policies"][1:], start=1
        ):
            name = policy["name"]
            candidate_metrics = [
                _trajectory_metrics(run, len(evaluation))
                for run in policy_runs[name]
            ]
            metric_results = {}
            for metric_position, metric in enumerate(
                analysis["reported_paired_differences"]
            ):
                differences = np.asarray(
                    [
                        candidate[metric] - baseline[metric]
                        for candidate, baseline in zip(
                            candidate_metrics, baseline_metrics
                        )
                    ],
                    dtype=np.float64,
                )
                rng_bootstrap = np.random.default_rng(
                    int(resampling["seed"])
                    + n_position * 1000
                    + policy_position * 100
                    + metric_position
                )
                metric_results[metric] = _paired_interval(
                    differences,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                    rng=rng_bootstrap,
                )
            n_results[name] = metric_results
        results[str(n_channels)] = n_results

        frozen_baseline = aggregate["results"][str(n_channels)][
            "policies"
        ]["no_heartbeat"]
        computed_clean = float(
            np.mean(
                [
                    value
                    for run in policy_runs["no_heartbeat"]
                    for value in run.clean
                ]
            )
        )
        computed_bits = float(
            np.mean(
                [
                    run.total_application_bits
                    for run in policy_runs["no_heartbeat"]
                ]
            )
            / len(evaluation)
        )
        aggregate_reproduced[str(n_channels)] = (
            computed_clean == frozen_baseline["clean_rate"]
            and computed_bits
            == frozen_baseline["mean_application_bits_per_scene"]
        )

    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "frozen_aggregate_reproduced": all(
            aggregate_reproduced.values()
        ),
        "all_clean_rate_interval_means_positive": all(
            policy["clean_rate_percentage_points"][
                "mean_paired_difference"
            ]
            > 0.0
            for n_result in results.values()
            for policy in n_result.values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"heartbeat statistics check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_context_heartbeat_statistics_complete",
        "analysis_sha256": sha256_file(args.analysis),
        "results": results,
        "checks": checks,
        "aggregate_reproduction_by_n": aggregate_reproduced,
        "resampling": resampling,
        "governance": {
            "development_only": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": analysis["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "checks": checks,
                "clean_rate_intervals": {
                    n: {
                        policy: metrics[
                            "clean_rate_percentage_points"
                        ]
                        for policy, metrics in values.items()
                    }
                    for n, values in results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
