#!/usr/bin/env python
"""Evaluate Markov task-codeword probability schedulers in the recovery loop."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_context_recovery_development import (  # noqa: E402
    simulate_trajectory,
    summarize_trajectories,
)
from run_stage6_joint_predictive_recovery_development import (  # noqa: E402
    _common_random_with_frozen_prefix,
    _paired_clean_interval,
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
from spectrum_semcom.stage6_markov_scheduler import (  # noqa: E402
    fit_markov_scheduler,
    online_token_bucket_candidates,
    score_markov_scheduler,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6_temporal_hazard import (  # noqa: E402
    build_temporal_hazard_dataset,
)


def _prediction_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    target = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    return {
        "sample_count": int(target.size),
        "positive_count": int(np.sum(target)),
        "positive_prevalence": float(np.mean(target)),
        "roc_auc": (
            float(roc_auc_score(target, values))
            if np.unique(target).size == 2
            else None
        ),
        "average_precision": float(
            average_precision_score(target, values)
        ),
    }


def _codeword_labels(states, codebook) -> np.ndarray:
    fallback = len(codebook.codewords)
    return np.asarray(
        [
            (
                fallback
                if decision.uses_fallback
                else int(decision.symbol_id)
            )
            for decision in (
                encode_task_state(codebook, state) for state in states
            )
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_markov_probability_scheduler_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/markov_probability_scheduler_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "markov_probability_scheduler_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Markov scheduler result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    safety = protocol["safety_boundary"]
    if (
        not governance["faults_are_controlled_injections_not_measurements"]
        or any(
            governance[key]
            for key in (
                "external_final_archives_may_be_opened",
                "external_final_signal_values_may_be_loaded",
                "external_final_access_may_be_consumed",
                "output_is_confirmatory_final",
            )
        )
        or not all(safety.values())
    ):
        raise ValueError("invalid Markov scheduler governance")
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
        raise ValueError("Markov scheduler cache hash mismatch")
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
    controller = protocol["controller"]
    reservation_fraction = float(
        controller[
            "maximum_update_reservation_fraction_of_evaluation_scenes"
        ]
    )
    maximum_reservations = int(
        math.ceil(reservation_fraction * len(evaluation_indices))
    )
    markov_config = protocol["markov_model"]
    monte_carlo = protocol["monte_carlo"]
    trajectory_count = int(monte_carlo["trajectories"])
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
        raise ValueError("Markov scheduler real codec declaration mismatch")

    all_results = {}
    reproduction_by_n = {}
    for n_position, n_value in enumerate(grid["n_channels"]):
        n_channels = int(n_value)
        demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
        queries = tuple(SpectrumTaskQuery(value) for value in demands)
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
        labels_all = _codeword_labels(states, codebook)
        markov_model = fit_markov_scheduler(
            labels_all,
            cache["cluster_ids"],
            training_groups=train_groups,
            class_count=len(codebook.codewords) + 1,
            duration_bin_lower_bounds=markov_config[
                "duration_bin_lower_bounds_scenes"
            ],
            laplace_alpha=float(markov_config["laplace_alpha"]),
            duration_shrinkage_strength=float(
                markov_config["duration_shrinkage_strength"]
            ),
            entropy_weight=float(markov_config["entropy_weight"]),
        )
        markov_scores_all = score_markov_scheduler(
            markov_model, labels_all, cache["cluster_ids"]
        )
        dataset = build_temporal_hazard_dataset(
            cache[f"channel_power_n{n_channels}"],
            states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            codebook,
        )
        features = dataset["features"]
        hazard_labels = dataset["labels"].astype(np.uint8)
        groups = dataset["groups"].astype(str)
        training = np.isin(groups, np.asarray(train_groups))
        evaluation = np.isin(groups, np.asarray(evaluation_groups))
        selected_c = float(
            predecessor_data["temporal_hazard"]["results"][
                str(n_channels)
            ]["selected_c"]
        )
        oof, _ = _fit_oof(
            features[training],
            hazard_labels[training],
            groups[training],
            c_value=selected_c,
            maximum_iterations=5000,
        )
        estimator = _model(selected_c, maximum_iterations=5000)
        estimator.fit(features[training], hazard_labels[training])
        learned_scores = np.full(hazard_labels.size, np.nan)
        learned_scores[training] = oof
        learned_scores[evaluation] = estimator.predict_proba(
            features[evaluation]
        )[:, 1]
        name_to_index = {
            name: index
            for index, name in enumerate(dataset["feature_names"])
        }
        previous_scores = features[
            :, name_to_index["previous_action_violation"]
        ]
        score_vectors = {
            "learned_repetition": learned_scores,
            "previous_transition_violation_repetition": previous_scores,
            "first_order_markov_repetition": markov_scores_all[
                "first_order_change_probability"
            ][dataset["source_indices"]],
            "duration_markov_repetition": markov_scores_all[
                "duration_change_probability"
            ][dataset["source_indices"]],
            "duration_entropy_markov_repetition": markov_scores_all[
                "duration_entropy_priority"
            ][dataset["source_indices"]],
        }
        thresholds = {}
        candidates_by_method = {
            "fixed_heartbeat_no_repetition": np.zeros(
                len(states), dtype=bool
            ),
            "periodic_repetition": _periodic_candidates(
                evaluation_indices,
                cache["cluster_ids"],
                fraction=reservation_fraction,
                state_count=len(states),
            ),
        }
        for method, scores in score_vectors.items():
            threshold = _upper_quantile_threshold(
                scores[training], reservation_fraction
            )
            thresholds[method] = threshold
            candidates_by_method[method] = _score_candidates(
                dataset["source_indices"],
                scores,
                evaluation,
                threshold=threshold,
                state_count=len(states),
            )
        token_bucket = protocol.get("token_bucket")
        token_method = (
            "token_bucket_duration_entropy_markov_repetition"
        )
        if token_bucket is not None:
            duration_entropy_scores_by_state = markov_scores_all[
                "duration_entropy_priority"
            ]
            candidates_by_method[token_method] = (
                online_token_bucket_candidates(
                    duration_entropy_scores_by_state,
                    evaluation_indices,
                    cache["cluster_ids"],
                    threshold=thresholds[
                        "duration_entropy_markov_repetition"
                    ],
                    state_count=len(states),
                    token_accrual_per_scene=float(
                        token_bucket["token_accrual_per_scene"]
                    ),
                    bucket_capacity=float(
                        token_bucket["bucket_capacity"]
                    ),
                    initial_tokens=float(
                        token_bucket["initial_tokens"]
                    ),
                    reset_at_group_boundary=bool(
                        token_bucket["reset_at_group_boundary"]
                    ),
                )
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
                    heartbeat_intervals_by_state=fixed_intervals,
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
        baseline_runs = method_runs[
            "fixed_heartbeat_no_repetition"
        ]
        baseline_summary = method_summaries[
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
                    - baseline_summary[
                        "mean_application_bits_per_scene"
                    ]
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
                    - baseline_summary[
                        "effective_cvar_0_9_regret_db"
                    ]
                ),
            }
        predecessor_methods = predecessor_data["joint_recovery"][
            "results"
        ][str(n_channels)]["methods"]
        reproduction_by_n[str(n_channels)] = {
            method: method_summaries[method]
            == predecessor_methods[method]
            for method in (
                "fixed_heartbeat_no_repetition",
                "periodic_repetition",
                "previous_transition_violation_repetition",
                "learned_repetition",
            )
        }
        transition_labels = (
            labels_all[dataset["source_indices"] + 1]
            != labels_all[dataset["source_indices"]]
        ).astype(np.uint8)
        predictive_metrics = {
            method: _prediction_metrics(
                transition_labels[evaluation], scores[evaluation]
            )
            for method, scores in score_vectors.items()
        }
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "maximum_update_reservations": maximum_reservations,
            "reservation_equivalent_bits": reservation_bits,
            "training_score_thresholds": thresholds,
            "candidate_counts": {
                method: int(
                    np.sum(values[evaluation_indices])
                )
                for method, values in candidates_by_method.items()
            },
            "prediction_metrics_on_evaluation_transitions": (
                predictive_metrics
            ),
            "methods": method_summaries,
            "comparisons_vs_fixed_heartbeat": comparisons,
        }

    markov_methods = (
        "first_order_markov_repetition",
        "duration_markov_repetition",
        "duration_entropy_markov_repetition",
    )
    rule = protocol["retention_rule"]
    if protocol.get("token_bucket") is None:
        variant_gates = {}
        for method in markov_methods:
            positive = 0
            beats_periodic = 0
            matches_learned = 0
            efficiencies = []
            for value in all_results.values():
                comparisons = value["comparisons_vs_fixed_heartbeat"]
                current = comparisons[method]
                efficiencies.append(
                    current["clean_gain_per_reserved_bit"]
                )
                positive += (
                    current[
                        "mean_clean_rate_gain_percentage_points"
                    ]
                    > 0.0
                )
                beats_periodic += (
                    current["clean_gain_per_reserved_bit"]
                    > comparisons["periodic_repetition"][
                        "clean_gain_per_reserved_bit"
                    ]
                )
                matches_learned += (
                    current["clean_gain_per_reserved_bit"]
                    >= comparisons["learned_repetition"][
                        "clean_gain_per_reserved_bit"
                    ]
                )
            variant_gates[method] = {
                "positive_clean_gain_n_count": int(positive),
                "efficiency_beats_periodic_n_count": int(
                    beats_periodic
                ),
                "efficiency_matches_or_beats_learned_n_count": int(
                    matches_learned
                ),
                "median_clean_gain_per_reserved_bit": float(
                    np.median(efficiencies)
                ),
                "eligible": (
                    positive
                    >= int(rule["minimum_positive_clean_gain_n_count"])
                    and beats_periodic
                    >= int(
                        rule[
                            "minimum_efficiency_beats_periodic_n_count"
                        ]
                    )
                    and matches_learned
                    >= int(
                        rule[
                            "minimum_efficiency_matches_or_beats_learned_n_count"
                        ]
                    )
                ),
            }
        complexity_order = {
            name: index for index, name in enumerate(markov_methods)
        }
        eligible = [
            method
            for method in markov_methods
            if variant_gates[method]["eligible"]
        ]
        selected_markov = (
            sorted(
                eligible,
                key=lambda method: (
                    -variant_gates[method][
                        "median_clean_gain_per_reserved_bit"
                    ],
                    complexity_order[method],
                ),
            )[0]
            if eligible
            else None
        )
        duration_beats_first = sum(
            value["comparisons_vs_fixed_heartbeat"][
                "duration_entropy_markov_repetition"
            ]["clean_gain_per_reserved_bit"]
            > value["comparisons_vs_fixed_heartbeat"][
                "first_order_markov_repetition"
            ]["clean_gain_per_reserved_bit"]
            for value in all_results.values()
        )
        retention = {
            "variant_gates": variant_gates,
            "selected_markov_scheduler": selected_markov,
            "markov_scheduler_retained": selected_markov is not None,
            "duration_entropy_beats_first_order_n_count": int(
                duration_beats_first
            ),
            "duration_feature_retained": duration_beats_first
            >= int(
                rule[
                    "duration_feature_requires_efficiency_beats_first_order_n_count"
                ]
            ),
        }
    else:
        token_method = (
            "token_bucket_duration_entropy_markov_repetition"
        )
        positive = 0
        beats_unshaped = 0
        matches_learned = 0
        for value in all_results.values():
            comparisons = value["comparisons_vs_fixed_heartbeat"]
            current = comparisons[token_method]
            positive += (
                current["mean_clean_rate_gain_percentage_points"] > 0.0
            )
            beats_unshaped += (
                current["clean_gain_per_reserved_bit"]
                > comparisons[
                    "duration_entropy_markov_repetition"
                ]["clean_gain_per_reserved_bit"]
            )
            matches_learned += (
                current["clean_gain_per_reserved_bit"]
                >= comparisons["learned_repetition"][
                    "clean_gain_per_reserved_bit"
                ]
            )
        required = int(rule["required_n_count_each_gate"])
        retained = (
            positive >= required
            and beats_unshaped >= required
            and matches_learned >= required
        )
        retention = {
            "positive_clean_gain_n_count": int(positive),
            "efficiency_beats_unshaped_markov_n_count": int(
                beats_unshaped
            ),
            "efficiency_matches_or_beats_learned_n_count": int(
                matches_learned
            ),
            "required_n_count_each_gate": required,
            "token_bucket_markov_scheduler_retained": retained,
            "terminate_markov_controller_research": not retained,
        }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "s6_4c_shared_methods_exactly_reproduced": all(
            all(methods.values())
            for methods in reproduction_by_n.values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            method["wrong_codebook_decode_count"] == 0
            for value in all_results.values()
            for method in value["methods"].values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Markov scheduler check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_markov_probability_scheduler_development_complete",
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
                "comparisons": {
                    n: {
                        method: value[
                            "comparisons_vs_fixed_heartbeat"
                        ][method]
                        for method in (
                            markov_methods
                            + (
                                (
                                    "token_bucket_duration_entropy_markov_repetition",
                                )
                                if protocol.get("token_bucket")
                                is not None
                                else ()
                            )
                        )
                    }
                    for n, value in all_results.items()
                },
                "checks": checks,
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
