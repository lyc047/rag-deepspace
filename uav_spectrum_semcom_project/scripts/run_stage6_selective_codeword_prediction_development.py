#!/usr/bin/env python
"""Run nested selective two-stage next-codeword development evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage6_codeword_prediction import (  # noqa: E402
    build_next_codeword_datasets,
    fit_conditional_transition_destination,
    fit_first_order_markov,
    predict_first_order_markov,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)


def _pipeline(*, c_value: float, maximum_iterations: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(maximum_iterations),
            random_state=int(seed),
        ),
    )


def _binary_probability(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_evaluation: np.ndarray,
    *,
    config: dict,
) -> np.ndarray:
    classes = np.unique(labels)
    if classes.size == 1:
        return np.full(x_evaluation.shape[0], float(classes[0]))
    estimator = _pipeline(
        c_value=float(config["c"]),
        maximum_iterations=int(config["maximum_iterations"]),
        seed=int(config["random_seed"]),
    )
    estimator.fit(x_train, labels)
    class_index = int(
        np.flatnonzero(estimator.named_steps["logisticregression"].classes_ == 1)[0]
    )
    return estimator.predict_proba(x_evaluation)[:, class_index]


def _multiclass_prediction(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_evaluation: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
    seed: int,
) -> np.ndarray:
    classes = np.unique(labels)
    if classes.size == 1:
        return np.full(x_evaluation.shape[0], int(classes[0]), dtype=np.int64)
    estimator = _pipeline(
        c_value=c_value,
        maximum_iterations=maximum_iterations,
        seed=seed,
    )
    estimator.fit(x_train, labels)
    return estimator.predict(x_evaluation).astype(np.int64)


def _destination_prediction(
    method: str,
    x_train: np.ndarray,
    labels: np.ndarray,
    current: np.ndarray,
    x_evaluation: np.ndarray,
    evaluation_current: np.ndarray,
    *,
    class_count: int,
    config: dict,
) -> np.ndarray:
    table = fit_conditional_transition_destination(
        current,
        labels,
        class_count=class_count,
        laplace_alpha=float(config["markov_laplace_alpha"]),
    )
    markov = predict_first_order_markov(table, evaluation_current)
    if method == "conditional_transition_markov":
        return markov
    if method != "changed_sample_logistic_regression":
        raise ValueError(f"unknown destination method: {method}")
    changed = labels != current
    if not np.any(changed):
        return markov
    learned = _multiclass_prediction(
        x_train[changed],
        labels[changed],
        x_evaluation,
        c_value=float(config["learned_c"]),
        maximum_iterations=int(config["learned_maximum_iterations"]),
        seed=int(config["learned_random_seed"]),
    )
    learned[learned == evaluation_current] = markov[
        learned == evaluation_current
    ]
    return learned


def _policy_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    current: np.ndarray,
    *,
    class_count: int,
) -> dict:
    changed = labels != current
    triggered = predictions != current
    stable = ~changed
    true_trigger = changed & triggered
    return {
        "sample_count": int(labels.size),
        "exact_class_accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(class_count),
                average="macro",
                zero_division=0,
            )
        ),
        "transition_prevalence": float(np.mean(changed)),
        "trigger_rate": float(np.mean(triggered)),
        "transition_recall": (
            float(np.sum(true_trigger) / np.sum(changed))
            if np.any(changed)
            else 0.0
        ),
        "trigger_precision": (
            float(np.sum(true_trigger) / np.sum(triggered))
            if np.any(triggered)
            else 0.0
        ),
        "false_switch_rate": (
            float(np.sum(stable & triggered) / np.sum(stable))
            if np.any(stable)
            else 0.0
        ),
        "transition_event_f1": float(
            f1_score(
                changed,
                triggered,
                average="binary",
                zero_division=0,
            )
        ),
        "changed_sample_exact_class_accuracy": (
            float(np.mean(predictions[changed] == labels[changed]))
            if np.any(changed)
            else None
        ),
    }


def _threshold_grid(config: dict) -> np.ndarray:
    start = float(config["threshold_grid_start"])
    stop = float(config["threshold_grid_stop"])
    step = float(config["threshold_grid_step"])
    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values[-1] > stop + 1e-12:
        raise ValueError("invalid selective threshold grid")
    return np.round(values, 12)


def _select_threshold(
    probabilities: np.ndarray,
    destinations: np.ndarray,
    labels: np.ndarray,
    current: np.ndarray,
    *,
    class_count: int,
    config: dict,
) -> dict:
    candidates = []
    for threshold in _threshold_grid(config):
        predictions = current.copy()
        trigger = probabilities >= threshold
        predictions[trigger] = destinations[trigger]
        metrics = _policy_metrics(
            labels, predictions, current, class_count=class_count
        )
        if (
            metrics["false_switch_rate"]
            <= float(config["maximum_inner_false_switch_rate"]) + 1e-12
            and metrics["transition_recall"]
            >= float(config["minimum_inner_transition_recall"]) - 1e-12
        ):
            candidates.append(
                {
                    "threshold": float(threshold),
                    "metrics": metrics,
                }
            )
    if not candidates:
        predictions = current.copy()
        return {
            "threshold": 1.01,
            "feasible": False,
            "metrics": _policy_metrics(
                labels, predictions, current, class_count=class_count
            ),
        }
    selected = sorted(
        candidates,
        key=lambda row: (
            -row["metrics"]["exact_class_accuracy"],
            -row["metrics"]["transition_event_f1"],
            row["metrics"]["trigger_rate"],
            -row["threshold"],
        ),
    )[0]
    selected["feasible"] = True
    return selected


def _inner_oof(
    features: np.ndarray,
    labels: np.ndarray,
    current: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    detector_config: dict,
    destination_config: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probabilities = np.full(labels.size, np.nan, dtype=np.float64)
    destinations = {
        method: np.full(labels.size, -1, dtype=np.int64)
        for method in destination_config["candidates"]
    }
    for held_group in sorted(set(groups.tolist())):
        validation = groups == held_group
        training = ~validation
        probabilities[validation] = _binary_probability(
            features[training],
            (labels[training] != current[training]).astype(np.uint8),
            features[validation],
            config=detector_config,
        )
        for method in destination_config["candidates"]:
            destinations[method][validation] = _destination_prediction(
                method,
                features[training],
                labels[training],
                current[training],
                features[validation],
                current[validation],
                class_count=class_count,
                config=destination_config,
            )
    if not np.all(np.isfinite(probabilities)) or any(
        np.any(values < 0) for values in destinations.values()
    ):
        raise AssertionError("incomplete inner out-of-fold predictions")
    return probabilities, destinations


def _select_inner_policy(
    probabilities: np.ndarray,
    destinations: dict[str, np.ndarray],
    labels: np.ndarray,
    current: np.ndarray,
    *,
    class_count: int,
    destination_order: list[str],
    rejection_config: dict,
) -> dict:
    rows = []
    for method in destination_order:
        selected = _select_threshold(
            probabilities,
            destinations[method],
            labels,
            current,
            class_count=class_count,
            config=rejection_config,
        )
        rows.append(
            {
                "method": method,
                **selected,
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    pool = feasible if feasible else rows
    selected = sorted(
        pool,
        key=lambda row: (
            -row["metrics"]["exact_class_accuracy"],
            -row["metrics"]["transition_event_f1"],
            destination_order.index(row["method"]),
        ),
    )[0]
    return {
        "selected_method": selected["method"],
        "selected_threshold": selected["threshold"],
        "selected_feasible": selected["feasible"],
        "selected_inner_metrics": selected["metrics"],
        "candidate_selections": rows,
    }


def _cluster_bootstrap_accuracy_gain(
    labels: np.ndarray,
    predictions: np.ndarray,
    persistence: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict:
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        chosen = np.concatenate([indices[group] for group in sampled])
        values[replicate] = float(
            np.mean(predictions[chosen] == labels[chosen])
            - np.mean(persistence[chosen] == labels[chosen])
        )
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean_accuracy_gain": float(
            np.mean(predictions == labels)
            - np.mean(persistence == labels)
        ),
        "confidence_interval_lower": float(np.quantile(values, tail)),
        "confidence_interval_upper": float(
            np.quantile(values, 1.0 - tail)
        ),
        "bootstrap_unit": "cluster",
        "replicates": int(replicates),
    }


def _safe_forecast_rate(
    predictions: np.ndarray,
    current: np.ndarray,
    source_indices: np.ndarray,
    states: list,
    codebook,
) -> dict:
    triggered = predictions != current
    count = int(np.sum(triggered))
    safe = 0
    for prediction, source_index, is_triggered in zip(
        predictions, source_indices, triggered
    ):
        if not is_triggered or prediction >= len(codebook.codewords):
            continue
        actions = codebook.codewords[int(prediction)].decoder_actions
        if (
            states[int(source_index) + 1].max_regret_db(actions)
            <= codebook.epsilon_db + 1e-12
        ):
            safe += 1
    return {
        "triggered_destination_count": count,
        "safe_triggered_destination_count": int(safe),
        "conditional_triggered_destination_safe_rate": (
            float(safe / count) if count else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_selective_codeword_prediction_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/selective_codeword_prediction_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "selective_codeword_prediction_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite selective result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if any(protocol["governance"].values()):
        raise ValueError("selective prediction governance is invalid")
    safety = protocol["safety_boundary"]
    if not all(safety.values()):
        raise ValueError("selective prediction safety boundary is invalid")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("selective prediction cache hash mismatch")
    predecessor = protocol["frozen_predecessor"]
    if sha256_file(PROJECT_DIR / predecessor["path"]) != predecessor["sha256"]:
        raise ValueError("selective prediction predecessor hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    if access_before["access_count"] != 0:
        raise ValueError("external Final access is not pristine")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}

    validation = protocol["validation"]
    history_length = int(validation["history_length"])
    groups_all = cache[validation["group_field"]].astype(str)
    available_groups = tuple(sorted(set(groups_all.tolist())))
    fixed_codebook = protocol.get("fixed_codebook")
    if fixed_codebook is None:
        codebook_fit_groups = available_groups
        outer_groups = available_groups
    else:
        codebook_fit_groups = tuple(
            fixed_codebook["fit_groups"]
        )
        outer_groups = tuple(
            protocol["validation"]["outer_evaluation_groups"]
        )
        if (
            not set(codebook_fit_groups).issubset(available_groups)
            or not set(outer_groups).issubset(available_groups)
            or set(codebook_fit_groups) & set(outer_groups)
        ):
            raise ValueError("invalid fixed-codebook group declaration")
    grid = protocol["task_grid"]
    detector_config = protocol["change_detector"]
    destination_config = protocol["destination_predictors"]
    rejection_config = protocol["rejection_policy"]
    statistics = protocol["statistics"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    results = {}

    for n_value in grid["n_channels"]:
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
        codebook_fit_indices = np.flatnonzero(
            np.isin(groups_all, codebook_fit_groups)
        )
        codebook = fit_greedy_task_codebook(
            [states[int(index)] for index in codebook_fit_indices]
        )
        dataset = build_next_codeword_datasets(
            cache[f"channel_power_n{n_channels}"],
            states,
            cache["timestamps_local"],
            groups_all,
            codebook,
            history_lengths=(history_length,),
        )
        features = dataset["features_by_window"][str(history_length)]
        labels = dataset["labels"]
        current = dataset["current_labels"]
        groups = dataset["groups"].astype(str)
        class_count = int(dataset["class_count"])
        outer_evaluation = np.isin(groups, outer_groups)
        selected_predictions = np.full(labels.size, -1, dtype=np.int64)
        persistence_predictions = current.copy()
        markov_predictions = np.full(labels.size, -1, dtype=np.int64)
        one_stage_predictions = np.full(labels.size, -1, dtype=np.int64)
        outer_rows = []

        for outer_group in outer_groups:
            evaluation = groups == outer_group
            training = ~evaluation
            inner_probabilities, inner_destinations = _inner_oof(
                features[training],
                labels[training],
                current[training],
                groups[training],
                class_count=class_count,
                detector_config=detector_config,
                destination_config=destination_config,
            )
            selected = _select_inner_policy(
                inner_probabilities,
                inner_destinations,
                labels[training],
                current[training],
                class_count=class_count,
                destination_order=list(destination_config["candidates"]),
                rejection_config=rejection_config,
            )
            probability = _binary_probability(
                features[training],
                (labels[training] != current[training]).astype(np.uint8),
                features[evaluation],
                config=detector_config,
            )
            destination = _destination_prediction(
                selected["selected_method"],
                features[training],
                labels[training],
                current[training],
                features[evaluation],
                current[evaluation],
                class_count=class_count,
                config=destination_config,
            )
            prediction = current[evaluation].copy()
            trigger = probability >= float(
                selected["selected_threshold"]
            )
            prediction[trigger] = destination[trigger]
            selected_predictions[evaluation] = prediction

            markov_table = fit_first_order_markov(
                current[training],
                labels[training],
                class_count=class_count,
                laplace_alpha=float(
                    destination_config["markov_laplace_alpha"]
                ),
            )
            markov_predictions[evaluation] = predict_first_order_markov(
                markov_table, current[evaluation]
            )
            one_stage_predictions[evaluation] = _multiclass_prediction(
                features[training],
                labels[training],
                features[evaluation],
                c_value=float(detector_config["c"]),
                maximum_iterations=int(
                    detector_config["maximum_iterations"]
                ),
                seed=int(detector_config["random_seed"]),
            )
            outer_rows.append(
                {
                    "outer_group": outer_group,
                    "training_sample_count": int(np.sum(training)),
                    "evaluation_sample_count": int(np.sum(evaluation)),
                    "selected_destination_method": selected[
                        "selected_method"
                    ],
                    "selected_threshold": selected[
                        "selected_threshold"
                    ],
                    "inner_policy_feasible": selected[
                        "selected_feasible"
                    ],
                    "inner_selected_metrics": selected[
                        "selected_inner_metrics"
                    ],
                    "outer_metrics": _policy_metrics(
                        labels[evaluation],
                        prediction,
                        current[evaluation],
                        class_count=class_count,
                    ),
                }
            )
        if any(
            np.any(values[outer_evaluation] < 0)
            for values in (
                selected_predictions,
                markov_predictions,
                one_stage_predictions,
            )
        ):
            raise AssertionError("outer predictions are incomplete")

        method_metrics = {
            "selective_two_stage": _policy_metrics(
                labels[outer_evaluation],
                selected_predictions[outer_evaluation],
                current[outer_evaluation],
                class_count=class_count,
            ),
            "current_codeword_persistence": _policy_metrics(
                labels[outer_evaluation],
                persistence_predictions[outer_evaluation],
                current[outer_evaluation],
                class_count=class_count,
            ),
            "first_order_markov": _policy_metrics(
                labels[outer_evaluation],
                markov_predictions[outer_evaluation],
                current[outer_evaluation],
                class_count=class_count,
            ),
            "one_stage_balanced_logistic_regression": _policy_metrics(
                labels[outer_evaluation],
                one_stage_predictions[outer_evaluation],
                current[outer_evaluation],
                class_count=class_count,
            ),
        }
        comparison = _cluster_bootstrap_accuracy_gain(
            labels[outer_evaluation],
            selected_predictions[outer_evaluation],
            persistence_predictions[outer_evaluation],
            groups[outer_evaluation],
            replicates=int(statistics["cluster_bootstrap_replicates"]),
            confidence_level=float(statistics["confidence_level"]),
            seed=int(statistics["seed"]) + n_channels,
        )
        selected_metrics = method_metrics["selective_two_stage"]
        result_row = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "class_count": class_count,
            "development_sample_count": int(labels.size),
            "sample_count": int(np.sum(outer_evaluation)),
            "cluster_count": len(outer_groups),
            "codebook_fit_groups": list(codebook_fit_groups),
            "outer_evaluation_groups": list(outer_groups),
            "class_counts": np.bincount(
                labels[outer_evaluation], minlength=class_count
            ).tolist(),
            "outer_folds": outer_rows,
            "selected_destination_method_fold_counts": {
                method: int(
                    sum(
                        row["selected_destination_method"] == method
                        for row in outer_rows
                    )
                )
                for method in destination_config["candidates"]
            },
            "feasible_inner_policy_fold_count": int(
                sum(row["inner_policy_feasible"] for row in outer_rows)
            ),
            "method_metrics": method_metrics,
            "comparison_vs_persistence": comparison,
            "trigger_precision_lift_over_transition_prevalence": float(
                selected_metrics["trigger_precision"]
                - selected_metrics["transition_prevalence"]
            ),
            "selective_safe_forecast": _safe_forecast_rate(
                selected_predictions[outer_evaluation],
                current[outer_evaluation],
                dataset["source_indices"][outer_evaluation],
                states,
                codebook,
            ),
        }
        if fixed_codebook is None:
            result_row.pop("development_sample_count")
            result_row.pop("codebook_fit_groups")
            result_row.pop("outer_evaluation_groups")
        results[str(n_channels)] = result_row

    rule = protocol["retention_rule"]
    required = int(rule["required_n_count_for_each_gate"])
    accuracy_point_pass = sum(
        value["comparison_vs_persistence"]["mean_accuracy_gain"]
        >= float(rule["minimum_exact_accuracy_gain_vs_persistence"])
        for value in results.values()
    )
    accuracy_ci_pass = sum(
        value["comparison_vs_persistence"]["confidence_interval_lower"]
        >= float(rule["minimum_accuracy_gain_lower_confidence_bound"])
        for value in results.values()
    )
    false_switch_pass = sum(
        value["method_metrics"]["selective_two_stage"][
            "false_switch_rate"
        ]
        <= float(rule["maximum_false_switch_rate"])
        for value in results.values()
    )
    recall_pass = sum(
        value["method_metrics"]["selective_two_stage"][
            "transition_recall"
        ]
        >= float(rule["minimum_transition_recall"])
        for value in results.values()
    )
    precision_pass = sum(
        value["trigger_precision_lift_over_transition_prevalence"]
        >= float(
            rule[
                "minimum_trigger_precision_lift_over_transition_prevalence"
            ]
        )
        for value in results.values()
    )
    retention = {
        "accuracy_point_gate_n_count": int(accuracy_point_pass),
        "accuracy_confidence_gate_n_count": int(accuracy_ci_pass),
        "false_switch_gate_n_count": int(false_switch_pass),
        "transition_recall_gate_n_count": int(recall_pass),
        "trigger_precision_gate_n_count": int(precision_pass),
        "required_n_count_each_gate": required,
        "destination_prediction_retained": (
            accuracy_point_pass >= required
            and accuracy_ci_pass >= required
            and false_switch_pass >= required
            and recall_pass >= required
            and precision_pass >= required
        ),
        "change_risk_may_continue_as_scheduler_only": (
            recall_pass >= required and precision_pass >= required
        ),
    }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    outer_check_name = (
        "seven_outer_clusters_evaluated_once"
        if fixed_codebook is None
        else "declared_outer_clusters_evaluated_once"
    )
    checks = {
        outer_check_name: all(
            value["cluster_count"] == len(outer_groups)
            and len(value["outer_folds"]) == len(outer_groups)
            for value in results.values()
        ),
        "all_outer_predictions_complete": all(
            sum(
                row["evaluation_sample_count"]
                for row in value["outer_folds"]
            )
            == value["sample_count"]
            for value in results.values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"selective prediction check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_selective_codeword_prediction_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "results": results,
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
                "summary": {
                    n: {
                        "accuracy": value["method_metrics"][
                            "selective_two_stage"
                        ]["exact_class_accuracy"],
                        "persistence_accuracy": value["method_metrics"][
                            "current_codeword_persistence"
                        ]["exact_class_accuracy"],
                        "false_switch_rate": value["method_metrics"][
                            "selective_two_stage"
                        ]["false_switch_rate"],
                        "transition_recall": value["method_metrics"][
                            "selective_two_stage"
                        ]["transition_recall"],
                        "trigger_precision_lift": value[
                            "trigger_precision_lift_over_transition_prevalence"
                        ],
                    }
                    for n, value in results.items()
                },
                "checks": checks,
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
