#!/usr/bin/env python
"""Run budgeted score-guided repetition experiments for Stage-6 updates."""

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

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from run_stage6_temporal_hazard_development import (  # noqa: E402
    _fit_oof,
    _model,
)
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_predictive_repetition import (  # noqa: E402
    ideal_hard_update_candidates,
    simulate_predictive_repetition,
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


def _upper_quantile_threshold(
    scores: np.ndarray, fraction: float
) -> float:
    return float(
        np.quantile(
            np.asarray(scores, dtype=np.float64),
            1.0 - float(fraction),
            method="higher",
        )
    )


def _score_candidates(
    source_indices: np.ndarray,
    scores: np.ndarray,
    selected_rows: np.ndarray,
    *,
    threshold: float,
    state_count: int,
) -> np.ndarray:
    candidates = np.zeros(state_count, dtype=bool)
    for source, score, selected in zip(
        source_indices, scores, selected_rows
    ):
        if selected and score >= threshold:
            candidates[int(source) + 1] = True
    return candidates


def _periodic_candidates(
    evaluation_indices: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    fraction: float,
    state_count: int,
) -> np.ndarray:
    candidates = np.zeros(state_count, dtype=bool)
    interval = max(1, int(round(1.0 / float(fraction))))
    position_in_group = 0
    previous_group = None
    for index in np.asarray(evaluation_indices, dtype=np.int64):
        group = str(cluster_ids[int(index)])
        if group != previous_group:
            position_in_group = 0
            previous_group = group
        elif position_in_group % interval == 0:
            candidates[int(index)] = True
        position_in_group += 1
    return candidates


def _remove_group_first_scenes(
    candidates: np.ndarray,
    evaluation_indices: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    result = np.asarray(candidates, dtype=bool).copy()
    previous = None
    for index in np.asarray(evaluation_indices, dtype=np.int64):
        group = str(cluster_ids[int(index)])
        if group != previous:
            result[int(index)] = False
            previous = group
    return result


def _summarize(runs, scene_count: int) -> dict:
    effective = np.asarray(
        [
            regret
            for run in runs
            for regret in run.effective_regret_db
        ],
        dtype=np.float64,
    )
    total = len(runs) * scene_count
    return {
        "trajectory_count": len(runs),
        "scene_evaluations": total,
        "mean_application_bits_per_scene": float(
            np.mean([run.total_application_bits for run in runs])
            / scene_count
        ),
        "mean_extra_repetition_bits_per_scene": float(
            np.mean([run.extra_repetition_bits for run in runs])
            / scene_count
        ),
        "mean_reserved_capacity_bits_per_scene": float(
            np.mean([run.reserved_capacity_bits for run in runs])
            / scene_count
        ),
        "mean_update_count": float(
            np.mean([run.update_count for run in runs])
        ),
        "mean_reservation_count": float(
            np.mean([run.reservation_count for run in runs])
        ),
        "mean_protected_update_count": float(
            np.mean([run.protected_update_count for run in runs])
        ),
        "mean_failed_update_count": float(
            np.mean([run.failed_update_count for run in runs])
        ),
        "reservation_utilization": float(
            np.sum([run.protected_update_count for run in runs])
            / max(
                np.sum([run.reservation_count for run in runs]),
                1,
            )
        ),
        "clean_rate": float(
            np.sum([np.sum(run.clean) for run in runs]) / total
        ),
        "availability_rate": float(
            np.sum([np.sum(run.available) for run in runs]) / total
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": empirical_cvar_numpy(
            effective, 0.9
        ),
    }


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
            * (
                np.mean(candidate.clean)
                - np.mean(baseline.clean)
            )
            for candidate, baseline in zip(
                candidate_runs, baseline_runs
            )
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(replicates, differences.size),
    )
    samples = np.mean(differences[indices], axis=1)
    alpha = 1.0 - confidence_level
    return {
        "mean_clean_rate_gain_percentage_points": float(
            np.mean(differences)
        ),
        "confidence_interval_lower": float(
            np.quantile(samples, alpha / 2.0)
        ),
        "confidence_interval_upper": float(
            np.quantile(samples, 1.0 - alpha / 2.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_predictive_repetition_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/predictive_repetition_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "predictive_repetition_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite predictive repetition result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    safety = protocol["safety_boundary"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not all(safety.values())
    ):
        raise ValueError("predictive repetition governance is invalid")
    predecessor = protocol["predecessor"]
    predecessor_path = PROJECT_DIR / predecessor["result"]
    if sha256_file(predecessor_path) != predecessor["result_sha256"]:
        raise ValueError("temporal hazard predecessor hash mismatch")
    predecessor_result = json.loads(
        predecessor_path.read_text(encoding="utf-8")
    )
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("predictive repetition cache hash mismatch")
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
    monte_carlo = protocol["monte_carlo"]
    trajectory_count = int(monte_carlo["trajectories"])
    all_results = {}

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
        session = install_codebook(
            codebook, epoch=int(grid["codebook_epoch"])
        )
        reservation_bits = maximum_compact_update_bits(session)
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
            predecessor_result["results"][str(n_channels)]["selected_c"]
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
        name_to_index = {
            name: index
            for index, name in enumerate(dataset["feature_names"])
        }
        score_vectors = {
            "learned_logistic": learned_scores,
            "previous_transition_violation": features[
                :, name_to_index["previous_action_violation"]
            ],
            "past_mean_absolute_power_change": features[
                :, name_to_index["past_change_mean_db"]
            ],
        }
        ideal = ideal_hard_update_candidates(
            states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            evaluation_indices,
            session=session,
            max_age_minutes=float(grid["max_state_age_minutes"]),
        )
        ideal = _remove_group_first_scenes(
            ideal, evaluation_indices, cache["cluster_ids"]
        )
        n_results = {}
        for loss_position, loss_value in enumerate(
            protocol["link_conditions"][
                "task_packet_loss_probabilities"
            ]
        ):
            loss = float(loss_value)
            rng = np.random.default_rng(
                int(monte_carlo["seed"])
                + n_position * 100_000
                + loss_position * 10_000
            )
            common_random = rng.random(
                (trajectory_count, len(evaluation_indices), 2)
            )
            budget_results = {}
            for budget_position, fraction_value in enumerate(
                protocol["protection"][
                    "maximum_reservation_fractions_of_evaluation_scenes"
                ]
            ):
                fraction = float(fraction_value)
                maximum_reservations = int(
                    math.ceil(fraction * len(evaluation_indices))
                )
                method_candidates = {
                    "no_protection": np.zeros(len(states), dtype=bool),
                    "periodic_schedule": _periodic_candidates(
                        evaluation_indices,
                        cache["cluster_ids"],
                        fraction=fraction,
                        state_count=len(states),
                    ),
                    "future_hard_trigger_oracle": ideal,
                }
                thresholds = {}
                for method, scores in score_vectors.items():
                    threshold = _upper_quantile_threshold(
                        scores[training], fraction
                    )
                    thresholds[method] = threshold
                    method_candidates[method] = _score_candidates(
                        dataset["source_indices"],
                        scores,
                        evaluation,
                        threshold=threshold,
                        state_count=len(states),
                    )
                method_runs = {}
                method_summaries = {}
                for method in protocol["methods"]:
                    cap = (
                        0
                        if method == "no_protection"
                        else maximum_reservations
                    )
                    runs = [
                        simulate_predictive_repetition(
                            states,
                            cache["timestamps_local"],
                            cache["cluster_ids"],
                            evaluation_indices,
                            session=session,
                            protection_candidates=method_candidates[
                                method
                            ],
                            maximum_reservations=cap,
                            reservation_equivalent_bits=reservation_bits,
                            task_loss_probability=loss,
                            random_values=common_random[trajectory],
                            max_age_minutes=float(
                                grid["max_state_age_minutes"]
                            ),
                            outage_penalty_db=float(
                                grid["outage_penalty_db"]
                            ),
                        )
                        for trajectory in range(trajectory_count)
                    ]
                    method_runs[method] = runs
                    method_summaries[method] = _summarize(
                        runs, len(evaluation_indices)
                    )
                baseline = method_summaries["no_protection"]
                comparisons = {}
                for method in protocol["methods"]:
                    if method == "no_protection":
                        continue
                    summary = method_summaries[method]
                    interval = _paired_clean_interval(
                        method_runs[method],
                        method_runs["no_protection"],
                        replicates=int(
                            monte_carlo["paired_bootstrap_replicates"]
                        ),
                        confidence_level=float(
                            monte_carlo["confidence_level"]
                        ),
                        seed=(
                            int(monte_carlo["seed"])
                            + n_position * 100_000
                            + loss_position * 10_000
                            + budget_position * 100
                            + protocol["methods"].index(method)
                        ),
                    )
                    reserved = summary[
                        "mean_reserved_capacity_bits_per_scene"
                    ]
                    comparisons[method] = {
                        **interval,
                        "application_bit_increase_per_scene": float(
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
                budget_results[str(fraction)] = {
                    "maximum_reservations": maximum_reservations,
                    "reservation_equivalent_bits": reservation_bits,
                    "training_score_thresholds": thresholds,
                    "methods": method_summaries,
                    "comparisons_vs_no_protection": comparisons,
                }
            n_results[str(loss)] = {
                "task_packet_loss_probability": loss,
                "budgets": budget_results,
            }
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "reservation_equivalent_bits": reservation_bits,
            "link_conditions": n_results,
        }

    positive_by_n = {}
    learned_beats_periodic = 0
    comparison_cells = 0
    for n, n_result in all_results.items():
        learned_gains = []
        for condition in n_result["link_conditions"].values():
            for budget in condition["budgets"].values():
                comparisons = budget["comparisons_vs_no_protection"]
                learned = comparisons["learned_logistic"]
                periodic = comparisons["periodic_schedule"]
                learned_gains.append(
                    learned[
                        "mean_clean_rate_gain_percentage_points"
                    ]
                )
                learned_beats_periodic += (
                    learned["clean_gain_per_reserved_bit"]
                    > periodic["clean_gain_per_reserved_bit"]
                )
                comparison_cells += 1
        positive_by_n[n] = any(value > 0 for value in learned_gains)
    retention = {
        "positive_clean_gain_exists_for_every_n": all(
            positive_by_n.values()
        ),
        "positive_by_n": positive_by_n,
        "learned_beats_periodic_efficiency_cells": int(
            learned_beats_periodic
        ),
        "total_nonzero_budget_cells": int(comparison_cells),
        "required_efficiency_cells": int(
            math.ceil(comparison_cells / 2)
        ),
    }
    retention["learned_repetition_retained"] = (
        retention["positive_clean_gain_exists_for_every_n"]
        and retention["learned_beats_periodic_efficiency_cells"]
        >= retention["required_efficiency_cells"]
    )
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_real_reservation_lengths_positive": all(
            result["reservation_equivalent_bits"] > 0
            for result in all_results.values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"predictive repetition check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_predictive_repetition_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_scene_count": int(len(evaluation_indices)),
        },
        "results": all_results,
        "retention_decision": retention,
        "checks": checks,
        "safety_boundary": safety,
        "governance": {
            "development_only": True,
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
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
