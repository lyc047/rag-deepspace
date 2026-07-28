#!/usr/bin/env python
"""Run Stage-6 prediction, heartbeat, and context-recovery jointly."""

from __future__ import annotations

import argparse
import json
import math
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
from run_stage6_predictive_repetition_development import (  # noqa: E402
    _periodic_candidates,
    _score_candidates,
    _upper_quantile_threshold,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from run_stage6_temporal_hazard_development import (  # noqa: E402
    _fit_oof,
    _model,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    decode_context_install,
    encode_context_install,
    maximum_compact_update_bits,
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
from spectrum_semcom.stage6_temporal_hazard import (  # noqa: E402
    build_temporal_hazard_dataset,
)


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
    "mean_heartbeat_probe_count",
    "mean_heartbeat_response_count",
    "mean_heartbeat_failure_count",
    "availability_rate",
    "clean_rate",
    "risk_violation_rate",
    "conditional_mean_regret_db",
    "conditional_maximum_regret_db",
    "effective_mean_regret_db",
    "effective_cvar_0_9_regret_db",
)


def _paired_clean_interval(
    candidate_runs,
    baseline_runs,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict:
    differences = np.asarray(
        [
            100.0
            * (np.mean(candidate.clean) - np.mean(baseline.clean))
            for candidate, baseline in zip(candidate_runs, baseline_runs)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(replicates, differences.size),
    )
    bootstrap = np.mean(differences[indices], axis=1)
    alpha = 1.0 - confidence_level
    return {
        "mean_clean_rate_gain_percentage_points": float(
            np.mean(differences)
        ),
        "confidence_interval_lower": float(
            np.quantile(bootstrap, alpha / 2.0)
        ),
        "confidence_interval_upper": float(
            np.quantile(bootstrap, 1.0 - alpha / 2.0)
        ),
    }


def _common_random_with_frozen_prefix(
    *,
    trajectories: int,
    scene_count: int,
    seed: int,
) -> np.ndarray:
    if trajectories < 30:
        raise ValueError("joint experiment must preserve 30 frozen trajectories")
    frozen_rng = np.random.default_rng(seed)
    frozen_six = frozen_rng.random((30, scene_count, 6))
    frozen_heartbeat = frozen_rng.random((30, scene_count, 2))
    remaining = trajectories - 30
    if remaining:
        extension_rng = np.random.default_rng(seed + 7_777_777)
        extension_six = extension_rng.random(
            (remaining, scene_count, 6)
        )
        extension_heartbeat = extension_rng.random(
            (remaining, scene_count, 2)
        )
        first_eight = np.concatenate(
            [
                np.concatenate([frozen_six, extension_six], axis=0),
                np.concatenate(
                    [frozen_heartbeat, extension_heartbeat], axis=0
                ),
            ],
            axis=2,
        )
    else:
        first_eight = np.concatenate(
            [frozen_six, frozen_heartbeat], axis=2
        )
    repetition_rng = np.random.default_rng(seed + 8_888_888)
    repetition = repetition_rng.random(
        (trajectories, scene_count, 1)
    )
    return np.concatenate([first_eight, repetition], axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_joint_predictive_recovery_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/joint_predictive_recovery_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "joint_predictive_recovery_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite joint recovery result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    safety = protocol["safety_boundary"]
    if (
        not governance["faults_are_controlled_injections_not_measurements"]
        or governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not all(safety.values())
    ):
        raise ValueError("joint predictive recovery governance is invalid")
    predecessor_data = {}
    for name, specification in protocol["frozen_predecessors"].items():
        path = PROJECT_DIR / specification["path"]
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"frozen predecessor changed: {name}")
        predecessor_data[name] = json.loads(
            path.read_text(encoding="utf-8")
        )
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("joint recovery cache hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
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
        raise ValueError("joint recovery real codec declaration mismatch")
    controller = protocol["controller"]
    reservation_fraction = float(
        controller[
            "maximum_update_reservation_fraction_of_evaluation_scenes"
        ]
    )
    maximum_reservations = int(
        math.ceil(reservation_fraction * len(evaluation_indices))
    )
    monte_carlo = protocol["monte_carlo"]
    trajectory_count = int(monte_carlo["trajectories"])
    all_results = {}
    reproduction_by_n = {}

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
            [states[int(index)] for index in train_indices]
        )
        sender_session = install_codebook(
            codebook, epoch=int(grid["codebook_epoch"])
        )
        install_packet = encode_context_install(
            sender_session, node_id=1
        )
        receiver_session = decode_context_install(install_packet).session
        reservation_bits = maximum_compact_update_bits(sender_session)
        dataset = build_temporal_hazard_dataset(
            cache[f"channel_power_n{n_channels}"],
            states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            codebook,
        )
        features = dataset["features"]
        labels = dataset["labels"].astype(np.uint8)
        groups = dataset["groups"].astype(str)
        training = np.isin(groups, np.asarray(train_groups))
        evaluation = np.isin(groups, np.asarray(evaluation_groups))
        selected_c = float(
            predecessor_data["temporal_hazard_result"]["results"][
                str(n_channels)
            ]["selected_c"]
        )
        oof, _ = _fit_oof(
            features[training],
            labels[training],
            groups[training],
            c_value=selected_c,
            maximum_iterations=5000,
        )
        estimator = _model(selected_c, maximum_iterations=5000)
        estimator.fit(features[training], labels[training])
        learned_scores = np.full(labels.size, np.nan)
        learned_scores[training] = oof
        learned_scores[evaluation] = estimator.predict_proba(
            features[evaluation]
        )[:, 1]
        learned_threshold = _upper_quantile_threshold(
            learned_scores[training], reservation_fraction
        )
        learned_candidates = _score_candidates(
            dataset["source_indices"],
            learned_scores,
            evaluation,
            threshold=learned_threshold,
            state_count=len(states),
        )
        name_to_index = {
            name: index
            for index, name in enumerate(dataset["feature_names"])
        }
        previous_scores = features[
            :, name_to_index["previous_action_violation"]
        ]
        previous_threshold = _upper_quantile_threshold(
            previous_scores[training], reservation_fraction
        )
        previous_candidates = _score_candidates(
            dataset["source_indices"],
            previous_scores,
            evaluation,
            threshold=previous_threshold,
            state_count=len(states),
        )
        periodic_candidates = _periodic_candidates(
            evaluation_indices,
            cache["cluster_ids"],
            fraction=reservation_fraction,
            state_count=len(states),
        )
        false_candidates = np.zeros(len(states), dtype=bool)
        fixed_intervals = np.full(
            len(states),
            int(controller["base_heartbeat_silence_scenes"]),
            dtype=np.int64,
        )
        adaptive_intervals = fixed_intervals.copy()
        adaptive_intervals[learned_candidates] = int(
            controller[
                "learned_high_risk_heartbeat_silence_scenes"
            ]
        )
        candidates_by_method = {
            "fixed_heartbeat_no_repetition": false_candidates,
            "periodic_repetition": periodic_candidates,
            "previous_transition_violation_repetition": (
                previous_candidates
            ),
            "learned_repetition": learned_candidates,
            "learned_repetition_adaptive_heartbeat": (
                learned_candidates
            ),
        }
        intervals_by_method = {
            method: (
                adaptive_intervals
                if method
                == "learned_repetition_adaptive_heartbeat"
                else fixed_intervals
            )
            for method in protocol["methods"]
        }
        seed = (
            int(monte_carlo["seed"])
            + n_position * 100_000
            + int(
                monte_carlo[
                    "heartbeat_predecessor_condition_seed_offset"
                ]
            )
        )
        common_random = _common_random_with_frozen_prefix(
            trajectories=trajectory_count,
            scene_count=len(evaluation_indices),
            seed=seed,
        )
        method_runs = {}
        method_summaries = {}
        for method in protocol["methods"]:
            cap = (
                0
                if method == "fixed_heartbeat_no_repetition"
                else maximum_reservations
            )
            runs = [
                simulate_trajectory(
                    states,
                    cache["timestamps_local"],
                    evaluation_indices,
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
                    heartbeat_request_bits=request_bits,
                    heartbeat_response_bits=response_bits,
                    heartbeat_intervals_by_state=intervals_by_method[
                        method
                    ],
                    update_protection_candidates=candidates_by_method[
                        method
                    ],
                    maximum_update_reservations=cap,
                    reservation_equivalent_bits=reservation_bits,
                )
                for trajectory in range(trajectory_count)
            ]
            method_runs[method] = runs
            method_summaries[method] = summarize_trajectories(
                runs, scene_count=len(evaluation_indices)
            )
        baseline_runs = method_runs["fixed_heartbeat_no_repetition"]
        baseline = method_summaries[
            "fixed_heartbeat_no_repetition"
        ]
        comparisons = {}
        for method_position, method in enumerate(protocol["methods"][1:]):
            summary = method_summaries[method]
            interval = _paired_clean_interval(
                method_runs[method],
                baseline_runs,
                replicates=int(
                    monte_carlo["paired_bootstrap_replicates"]
                ),
                confidence_level=float(
                    monte_carlo["confidence_level"]
                ),
                seed=seed + 90_000 + method_position,
            )
            reserved = summary[
                "mean_reserved_capacity_bits_per_scene"
            ]
            comparisons[method] = {
                **interval,
                "application_bit_change_per_scene": float(
                    summary["mean_application_bits_per_scene"]
                    - baseline["mean_application_bits_per_scene"]
                ),
                "reserved_capacity_bits_per_scene": reserved,
                "clean_gain_per_reserved_bit": float(
                    interval[
                        "mean_clean_rate_gain_percentage_points"
                    ]
                    / max(reserved, 1e-12)
                ),
                "effective_cvar_change_db": float(
                    summary["effective_cvar_0_9_regret_db"]
                    - baseline["effective_cvar_0_9_regret_db"]
                ),
            }
        frozen_prefix_summary = summarize_trajectories(
            baseline_runs[:30], scene_count=len(evaluation_indices)
        )
        frozen = predecessor_data["heartbeat_result"]["results"][
            str(n_channels)
        ]["policies"]["silence_20_scenes"]
        reproduction_by_n[str(n_channels)] = all(
            frozen_prefix_summary[key] == frozen[key]
            for key in _REPRODUCTION_KEYS
        )
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "maximum_update_reservations": maximum_reservations,
            "reservation_equivalent_bits": reservation_bits,
            "learned_training_threshold": learned_threshold,
            "previous_transition_threshold": previous_threshold,
            "learned_candidate_count": int(
                np.sum(learned_candidates[evaluation_indices])
            ),
            "methods": method_summaries,
            "comparisons_vs_fixed_heartbeat": comparisons,
        }

    learned_positive = 0
    learned_beats_periodic = 0
    adaptive_passes = 0
    for result in all_results.values():
        comparison = result["comparisons_vs_fixed_heartbeat"]
        learned = comparison["learned_repetition"]
        periodic = comparison["periodic_repetition"]
        adaptive = result["methods"][
            "learned_repetition_adaptive_heartbeat"
        ]
        fixed_learned = result["methods"]["learned_repetition"]
        learned_positive += (
            learned["mean_clean_rate_gain_percentage_points"] > 0.0
        )
        learned_beats_periodic += (
            learned["clean_gain_per_reserved_bit"]
            > periodic["clean_gain_per_reserved_bit"]
        )
        adaptive_passes += (
            adaptive["mean_application_bits_per_scene"]
            < fixed_learned["mean_application_bits_per_scene"]
            and 100.0
            * (adaptive["clean_rate"] - fixed_learned["clean_rate"])
            >= -0.5
        )
    retention = {
        "learned_positive_clean_gain_n_count": int(learned_positive),
        "learned_efficiency_beats_periodic_n_count": int(
            learned_beats_periodic
        ),
        "adaptive_heartbeat_pass_n_count": int(adaptive_passes),
        "required_n_count_each_gate": 3,
    }
    retention["learned_repetition_retained"] = (
        learned_positive >= 3 and learned_beats_periodic >= 3
    )
    retention["adaptive_heartbeat_retained"] = adaptive_passes >= 3
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "frozen_heartbeat_prefix_reproduced": all(
            reproduction_by_n.values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            method["wrong_codebook_decode_count"] == 0
            for result in all_results.values()
            for method in result["methods"].values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"joint predictive recovery failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_joint_predictive_recovery_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_scene_count": int(len(evaluation_indices)),
        },
        "results": all_results,
        "retention_decision": retention,
        "reproduction_by_n": reproduction_by_n,
        "checks": checks,
        "safety_boundary": safety,
        "governance": {
            "development_only": True,
            "faults_are_controlled_injections_not_measurements": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "retention_decision": retention,
                "checks": checks,
                "comparisons": {
                    n: value["comparisons_vs_fixed_heartbeat"]
                    for n, value in all_results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
