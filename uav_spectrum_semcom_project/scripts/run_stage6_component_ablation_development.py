#!/usr/bin/env python
"""Run the preregistered Stage-6 frozen-architecture component ablation."""

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
    TrajectoryResult,
    simulate_trajectory,
    summarize_trajectories,
)
from run_stage6_joint_predictive_recovery_development import (  # noqa: E402
    _common_random_with_frozen_prefix,
)
from run_stage6_predictive_repetition_development import (  # noqa: E402
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
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
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


METHODS = (
    "full_candidate",
    "minus_learned_update_protection",
    "minus_fixed_heartbeat",
    "minus_context_belief_recovery",
    "ideal_feedback_upper_bound",
    "exact_query_bundle_reference",
)


def _exact_bundle_bits(
    n_channels: int,
    queries: tuple[SpectrumTaskQuery, ...],
    *,
    header_bits: int,
) -> int:
    payload = 0
    for query in queries:
        alphabet_size = n_channels - query.demand_channels + 1
        if alphabet_size < 1:
            raise ValueError("query exceeds channel grid")
        payload += int(math.ceil(math.log2(alphabet_size)))
    return int(header_bits + payload)


def _simulate_exact_query_bundle(
    states,
    evaluation_indices: np.ndarray,
    *,
    random_values: np.ndarray,
    packet_loss_probability: float,
    receiver_reset_probability: float,
    packet_bits: int,
    epsilon_db: float,
    outage_penalty_db: float,
) -> TrajectoryResult:
    receiver_actions: tuple[int, ...] | None = None
    available_rows: list[bool] = []
    clean_rows: list[bool] = []
    effective_regret: list[float] = []
    available_regret: list[float] = []
    for local_position, index in enumerate(evaluation_indices):
        random = random_values[local_position]
        if random[0] < receiver_reset_probability:
            receiver_actions = None
        state = states[int(index)]
        if random[3] >= packet_loss_probability:
            receiver_actions = state.optimal_actions
        available = receiver_actions is not None
        available_rows.append(available)
        if available:
            regret = state.max_regret_db(receiver_actions)
            available_regret.append(regret)
            effective_regret.append(regret)
            clean_rows.append(regret <= epsilon_db + 1e-12)
        else:
            effective_regret.append(outage_penalty_db)
            clean_rows.append(False)
    scene_count = len(evaluation_indices)
    return TrajectoryResult(
        total_application_bits=int(packet_bits * scene_count),
        forward_application_bits=int(packet_bits * scene_count),
        ack_application_bits=0,
        reserved_capacity_bits=0,
        context_install_count=0,
        compact_update_count=scene_count,
        update_reservation_count=0,
        protected_update_count=0,
        ack_frame_count=0,
        stale_ack_rejection_count=0,
        rejected_compact_count=0,
        wrong_codebook_decode_count=0,
        maximum_belief_size=0,
        heartbeat_probe_count=0,
        heartbeat_response_count=0,
        heartbeat_failure_count=0,
        available=available_rows,
        clean=clean_rows,
        effective_regret_db=effective_regret,
        available_regret_db=available_regret,
    )


def _trajectory_metrics(
    trajectory: TrajectoryResult,
    scene_count: int,
) -> dict[str, float]:
    effective = np.asarray(
        trajectory.effective_regret_db, dtype=np.float64
    )
    return {
        "clean_rate_percentage_points": float(
            100.0 * np.mean(trajectory.clean)
        ),
        "availability_rate_percentage_points": float(
            100.0 * np.mean(trajectory.available)
        ),
        "application_bits_per_scene": float(
            trajectory.total_application_bits / scene_count
        ),
        "reserved_capacity_bits_per_scene": float(
            trajectory.reserved_capacity_bits / scene_count
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": empirical_cvar_numpy(
            effective, 0.9
        ),
    }


def _paired_intervals(
    candidate_runs: list[TrajectoryResult],
    reference_runs: list[TrajectoryResult],
    *,
    scene_count: int,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict:
    candidate = [
        _trajectory_metrics(run, scene_count) for run in candidate_runs
    ]
    reference = [
        _trajectory_metrics(run, scene_count) for run in reference_runs
    ]
    names = tuple(candidate[0])
    differences = {
        name: np.asarray(
            [
                candidate_row[name] - reference_row[name]
                for candidate_row, reference_row in zip(
                    candidate, reference
                )
            ],
            dtype=np.float64,
        )
        for name in names
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(candidate_runs),
        size=(replicates, len(candidate_runs)),
    )
    alpha = 1.0 - confidence_level
    output = {}
    for name, values in differences.items():
        bootstrap = np.mean(values[indices], axis=1)
        output[name] = {
            "mean_difference": float(np.mean(values)),
            "confidence_interval_lower": float(
                np.quantile(bootstrap, alpha / 2.0)
            ),
            "confidence_interval_upper": float(
                np.quantile(bootstrap, 1.0 - alpha / 2.0)
            ),
        }
    return output


def _load_and_validate(protocol_path: Path) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not governance["faults_are_controlled_injections_not_measurements"]
        or not all(protocol["safety_boundary"].values())
    ):
        raise ValueError("invalid ablation governance")
    freeze = protocol["architecture_freeze"]
    for key in ("manifest", "result"):
        path = PROJECT_DIR / freeze[key]
        if sha256_file(path) != freeze[f"{key}_sha256"]:
            raise ValueError(f"architecture freeze {key} changed")
    if (
        freeze["algorithm_modules_may_be_added"]
        or freeze["frozen_parameters_may_be_retuned"]
    ):
        raise ValueError("ablation may not alter the frozen architecture")
    predecessors = {}
    for name, specification in protocol["frozen_predecessors"].items():
        path = PROJECT_DIR / specification["path"]
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"frozen predecessor changed: {name}")
        predecessors[name] = json.loads(path.read_text(encoding="utf-8"))
    if set(protocol["ablation_cells"]) != set(METHODS):
        raise ValueError("ablation cells do not match the frozen matrix")
    return protocol, predecessors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_component_ablation_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/component_ablation_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "component_ablation_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite component ablation result")
    protocol, predecessors = _load_and_validate(args.protocol)
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("ablation development cache changed")
    access_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
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
        raise ValueError("real protocol bit declaration changed")
    controller = protocol["frozen_controller"]
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
        receiver_session = decode_context_install(
            install_packet
        ).session
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
            predecessors["temporal_hazard_result"]["results"][
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
        fixed_intervals = np.full(
            len(states),
            int(controller["heartbeat_silence_scenes"]),
            dtype=np.int64,
        )
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
        exact_bits = _exact_bundle_bits(
            n_channels,
            queries,
            header_bits=int(grid["task_header_bits"]),
        )
        method_runs: dict[str, list[TrajectoryResult]] = {}
        for method in METHODS:
            runs = []
            for trajectory in range(trajectory_count):
                random_values = common_random[trajectory]
                if method == "exact_query_bundle_reference":
                    run = _simulate_exact_query_bundle(
                        states,
                        evaluation_indices,
                        random_values=random_values,
                        packet_loss_probability=float(
                            protocol["fault_condition"][
                                "task_loss_probability"
                            ]
                        ),
                        receiver_reset_probability=float(
                            protocol["fault_condition"][
                                "receiver_context_reset_probability"
                            ]
                        ),
                        packet_bits=exact_bits,
                        epsilon_db=float(grid["epsilon_db"]),
                        outage_penalty_db=float(
                            grid["outage_penalty_db"]
                        ),
                    )
                else:
                    belief = method != "minus_context_belief_recovery"
                    heartbeat = method in (
                        "full_candidate",
                        "minus_learned_update_protection",
                    )
                    learned = method in (
                        "full_candidate",
                        "minus_fixed_heartbeat",
                        "minus_context_belief_recovery",
                    )
                    simulation_method = (
                        "ideal_feedback"
                        if method == "ideal_feedback_upper_bound"
                        else (
                            "belief_risk_recovery"
                            if belief
                            else "naive_assumption"
                        )
                    )
                    run = simulate_trajectory(
                        states,
                        cache["timestamps_local"],
                        evaluation_indices,
                        sender_session=sender_session,
                        receiver_session=receiver_session,
                        install_packet=install_packet,
                        method=simulation_method,
                        condition=protocol["fault_condition"],
                        random_values=random_values,
                        epsilon_db=float(grid["epsilon_db"]),
                        max_age_minutes=float(
                            grid["max_state_age_minutes"]
                        ),
                        ack_bits=ack_bits,
                        outage_penalty_db=float(
                            grid["outage_penalty_db"]
                        ),
                        heartbeat_request_bits=request_bits,
                        heartbeat_response_bits=response_bits,
                        heartbeat_intervals_by_state=(
                            fixed_intervals if heartbeat else None
                        ),
                        update_protection_candidates=(
                            learned_candidates if learned else None
                        ),
                        maximum_update_reservations=(
                            maximum_reservations if learned else 0
                        ),
                        reservation_equivalent_bits=reservation_bits,
                    )
                runs.append(run)
            method_runs[method] = runs
        summaries = {
            method: summarize_trajectories(
                runs, scene_count=len(evaluation_indices)
            )
            for method, runs in method_runs.items()
        }
        predecessor_methods = predecessors["joint_result"]["results"][
            str(n_channels)
        ]["methods"]
        reproduction_by_n[str(n_channels)] = {
            "full_candidate_reproduces_learned_repetition": (
                summaries["full_candidate"]
                == predecessor_methods["learned_repetition"]
            ),
            "minus_learned_reproduces_no_repetition": (
                summaries["minus_learned_update_protection"]
                == predecessor_methods["fixed_heartbeat_no_repetition"]
            ),
        }
        comparisons = {}
        for method_position, method in enumerate(METHODS[1:]):
            comparisons[f"full_candidate_minus_{method}"] = (
                _paired_intervals(
                    method_runs["full_candidate"],
                    method_runs[method],
                    scene_count=len(evaluation_indices),
                    replicates=int(
                        monte_carlo["paired_bootstrap_replicates"]
                    ),
                    confidence_level=float(
                        monte_carlo["confidence_level"]
                    ),
                    seed=seed + 90_000 + method_position,
                )
            )
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "learned_training_threshold": learned_threshold,
            "learned_candidate_count": int(
                np.sum(learned_candidates[evaluation_indices])
            ),
            "maximum_update_reservations": maximum_reservations,
            "reservation_equivalent_bits": reservation_bits,
            "exact_query_bundle_bits": exact_bits,
            "methods": summaries,
            "paired_differences": comparisons,
        }

    rules = protocol["analysis_rules"]
    support = {}
    for component, ablation in (
        (
            "learned_update_protection",
            "minus_learned_update_protection",
        ),
        ("fixed_heartbeat", "minus_fixed_heartbeat"),
        (
            "context_belief_recovery_dependency_group",
            "minus_context_belief_recovery",
        ),
    ):
        supported_n = []
        point_positive_n = []
        for n_channels, result in all_results.items():
            comparison = result["paired_differences"][
                f"full_candidate_minus_{ablation}"
            ]["clean_rate_percentage_points"]
            if comparison["mean_difference"] > 0.0:
                point_positive_n.append(int(n_channels))
            if comparison["confidence_interval_lower"] > 0.0:
                supported_n.append(int(n_channels))
        support[component] = {
            "point_positive_n": point_positive_n,
            "paired_interval_above_zero_n": supported_n,
            "support_count": len(supported_n),
            "cross_n_support_threshold": int(
                rules["cross_n_component_support_count"]
            ),
            "cross_n_support_met": len(supported_n)
            >= int(rules["cross_n_component_support_count"]),
            "architecture_change_authorized": False,
        }

    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_frozen_predecessor_cells_reproduced": all(
            all(value.values()) for value in reproduction_by_n.values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            summary["wrong_codebook_decode_count"] == 0
            for result in all_results.values()
            for summary in result["methods"].values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
        "architecture_remains_frozen": (
            not rules["ablation_results_may_change_frozen_architecture"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"component ablation failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_component_ablation_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_scene_count": int(len(evaluation_indices)),
        },
        "comparison_sign": (
            "Every paired difference is full_candidate minus named method."
        ),
        "results": all_results,
        "component_support": support,
        "reproduction_by_n": reproduction_by_n,
        "checks": checks,
        "governance": {
            "development_only": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
            "architecture_change_authorized": False,
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
                "component_support": support,
                "checks": checks,
                "summary": {
                    n: {
                        method: {
                            "clean_rate": values["clean_rate"],
                            "application_bits_per_scene": values[
                                "mean_application_bits_per_scene"
                            ],
                            "effective_mean_regret_db": values[
                                "effective_mean_regret_db"
                            ],
                            "effective_cvar_0_9_regret_db": values[
                                "effective_cvar_0_9_regret_db"
                            ],
                        }
                        for method, values in result_by_n[
                            "methods"
                        ].items()
                    }
                    for n, result_by_n in all_results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
