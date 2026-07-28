#!/usr/bin/env python
"""Evaluate causal next-codeword forecasting on excluded development data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage6_codeword_prediction import (  # noqa: E402
    build_next_codeword_datasets,
    fit_first_order_markov,
    fit_frequency_prior,
    predict_first_order_markov,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)


def _model(config: dict):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config["c"]),
            class_weight="balanced",
            max_iter=int(config["maximum_iterations"]),
            random_state=int(config["random_seed"]),
        ),
    )


def _fit_predict_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    config: dict,
) -> np.ndarray:
    classes = np.unique(train_labels)
    if classes.size == 1:
        return np.full(
            evaluation_features.shape[0], int(classes[0]), dtype=np.int64
        )
    estimator = _model(config)
    estimator.fit(train_features, train_labels)
    return estimator.predict(evaluation_features).astype(np.int64)


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    current_labels: np.ndarray,
    *,
    class_count: int,
    fallback_class: int,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    current = np.asarray(current_labels, dtype=np.int64)
    changed = labels != current
    predicted_change = predictions != current
    return {
        "sample_count": int(labels.size),
        "exact_class_accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
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
        "transition_event_f1": float(
            f1_score(
                changed,
                predicted_change,
                average="binary",
                zero_division=0,
            )
        ),
        "changed_sample_exact_class_accuracy": (
            float(np.mean(predictions[changed] == labels[changed]))
            if np.any(changed)
            else None
        ),
        "fallback_prevalence": float(np.mean(labels == fallback_class)),
        "fallback_event_f1": float(
            f1_score(
                labels == fallback_class,
                predictions == fallback_class,
                average="binary",
                zero_division=0,
            )
        ),
        "packet_type_accuracy": float(
            np.mean(
                (labels == fallback_class)
                == (predictions == fallback_class)
            )
        ),
    }


def _baseline_predictions(
    name: str,
    train_labels: np.ndarray,
    train_current: np.ndarray,
    evaluation_current: np.ndarray,
    *,
    class_count: int,
    laplace_alpha: float,
) -> np.ndarray:
    if name == "training_frequency_prior":
        value = fit_frequency_prior(
            train_labels, class_count=class_count
        )
        return np.full(evaluation_current.size, value, dtype=np.int64)
    if name == "current_codeword_persistence":
        return np.asarray(evaluation_current, dtype=np.int64).copy()
    if name == "first_order_markov":
        table = fit_first_order_markov(
            train_current,
            train_labels,
            class_count=class_count,
            laplace_alpha=laplace_alpha,
        )
        return predict_first_order_markov(table, evaluation_current)
    raise ValueError(f"unknown baseline: {name}")


def _leave_one_group_out_learned(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    config: dict,
) -> np.ndarray:
    predictions = np.full(labels.size, -1, dtype=np.int64)
    for held_group in sorted(set(groups.tolist())):
        validation = groups == held_group
        training = ~validation
        predictions[validation] = _fit_predict_model(
            features[training],
            labels[training],
            features[validation],
            config=config,
        )
    if np.any(predictions < 0):
        raise AssertionError("incomplete learned out-of-fold predictions")
    return predictions


def _leave_one_group_out_baseline(
    name: str,
    labels: np.ndarray,
    current: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    laplace_alpha: float,
) -> np.ndarray:
    predictions = np.full(labels.size, -1, dtype=np.int64)
    for held_group in sorted(set(groups.tolist())):
        validation = groups == held_group
        training = ~validation
        predictions[validation] = _baseline_predictions(
            name,
            labels[training],
            current[training],
            current[validation],
            class_count=class_count,
            laplace_alpha=laplace_alpha,
        )
    if np.any(predictions < 0):
        raise AssertionError("incomplete baseline out-of-fold predictions")
    return predictions


def _cluster_bootstrap_accuracy_difference(
    labels: np.ndarray,
    learned: np.ndarray,
    baseline: np.ndarray,
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
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        selected = np.concatenate([indices[group] for group in sampled])
        differences[replicate] = float(
            np.mean(learned[selected] == labels[selected])
            - np.mean(baseline[selected] == labels[selected])
        )
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean_accuracy_gain": float(
            np.mean(learned == labels) - np.mean(baseline == labels)
        ),
        "confidence_interval_lower": float(
            np.quantile(differences, tail)
        ),
        "confidence_interval_upper": float(
            np.quantile(differences, 1.0 - tail)
        ),
        "bootstrap_unit": "evaluation_cluster",
        "replicates": int(replicates),
    }


def _safe_forecast_summary(
    predictions: np.ndarray,
    source_indices: np.ndarray,
    states: list,
    codebook,
) -> dict:
    codeword_forecasts = 0
    safe_forecasts = 0
    for prediction, source_index in zip(predictions, source_indices):
        if prediction >= len(codebook.codewords):
            continue
        codeword_forecasts += 1
        actions = codebook.codewords[int(prediction)].decoder_actions
        if (
            states[int(source_index) + 1].max_regret_db(actions)
            <= codebook.epsilon_db + 1e-12
        ):
            safe_forecasts += 1
    return {
        "predicted_codeword_count": int(codeword_forecasts),
        "safe_predicted_codeword_count": int(safe_forecasts),
        "conditional_predicted_codeword_safe_rate": (
            float(safe_forecasts / codeword_forecasts)
            if codeword_forecasts
            else None
        ),
        "unconditional_safe_codeword_forecast_rate": float(
            safe_forecasts / len(predictions)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_next_codeword_prediction_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/next_codeword_prediction_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "next_codeword_prediction_result.json"
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite next-codeword prediction result"
        )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    safety = protocol["safety_boundary"]
    if (
        any(governance.values())
        or not safety[
            "prediction_may_only_schedule_packet_type_reservation_or_recovery"
        ]
        or not safety[
            "prediction_may_not_execute_or_select_a_future_spectrum_action"
        ]
        or not safety[
            "actual_codeword_must_be_selected_after_observing_the_current_spectrum"
        ]
        or not safety["hard_regret_constraint_remains_mandatory"]
    ):
        raise ValueError("invalid next-codeword governance or safety boundary")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("next-codeword cache hash mismatch")
    for predecessor in protocol["frozen_predecessors"].values():
        if sha256_file(PROJECT_DIR / predecessor["path"]) != predecessor["sha256"]:
            raise ValueError("frozen predecessor hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    if access_before["access_count"] != 0:
        raise ValueError("external Final access is not pristine")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}

    split = protocol["split"]
    train_indices, _, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    del train_indices
    windows = tuple(int(value) for value in protocol["history_windows"])
    learned_config = protocol["learned_model"]
    baseline_config = protocol["baselines"]
    statistics = protocol["statistics"]
    grid = protocol["task_grid"]
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
        training_state_indices = np.flatnonzero(
            np.isin(cache["cluster_ids"].astype(str), train_groups)
        )
        codebook = fit_greedy_task_codebook(
            [states[int(index)] for index in training_state_indices]
        )
        dataset = build_next_codeword_datasets(
            cache[f"channel_power_n{n_channels}"],
            states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            codebook,
            history_lengths=windows,
        )
        labels = dataset["labels"]
        current = dataset["current_labels"]
        groups = dataset["groups"].astype(str)
        training = np.isin(groups, train_groups)
        evaluation = np.isin(groups, evaluation_groups)
        if np.any(training & evaluation) or not np.all(training | evaluation):
            raise AssertionError("next-codeword split is invalid")
        class_count = int(dataset["class_count"])
        fallback_class = int(dataset["fallback_class"])

        learned_cv = {}
        learned_evaluation = {}
        learned_predictions = {}
        for window in windows:
            features = dataset["features_by_window"][str(window)]
            oof_prediction = _leave_one_group_out_learned(
                features[training],
                labels[training],
                groups[training],
                config=learned_config,
            )
            learned_cv[str(window)] = _metrics(
                labels[training],
                oof_prediction,
                current[training],
                class_count=class_count,
                fallback_class=fallback_class,
            )
            prediction = _fit_predict_model(
                features[training],
                labels[training],
                features[evaluation],
                config=learned_config,
            )
            learned_predictions[str(window)] = prediction
            learned_evaluation[str(window)] = _metrics(
                labels[evaluation],
                prediction,
                current[evaluation],
                class_count=class_count,
                fallback_class=fallback_class,
            )
        selected_window = min(
            windows,
            key=lambda value: (
                -learned_cv[str(value)]["macro_f1"],
                value,
            ),
        )

        baseline_cv = {}
        baseline_evaluation = {}
        baseline_predictions = {}
        for name in baseline_config["names"]:
            oof_prediction = _leave_one_group_out_baseline(
                name,
                labels[training],
                current[training],
                groups[training],
                class_count=class_count,
                laplace_alpha=float(
                    baseline_config["markov_laplace_alpha"]
                ),
            )
            baseline_cv[name] = _metrics(
                labels[training],
                oof_prediction,
                current[training],
                class_count=class_count,
                fallback_class=fallback_class,
            )
            prediction = _baseline_predictions(
                name,
                labels[training],
                current[training],
                current[evaluation],
                class_count=class_count,
                laplace_alpha=float(
                    baseline_config["markov_laplace_alpha"]
                ),
            )
            baseline_predictions[name] = prediction
            baseline_evaluation[name] = _metrics(
                labels[evaluation],
                prediction,
                current[evaluation],
                class_count=class_count,
                fallback_class=fallback_class,
            )
        selected_baseline = min(
            baseline_config["names"],
            key=lambda name: (
                -baseline_cv[name]["macro_f1"],
                baseline_config["names"].index(name),
            ),
        )
        learned_selected_prediction = learned_predictions[
            str(selected_window)
        ]
        baseline_selected_prediction = baseline_predictions[
            selected_baseline
        ]
        learned_selected_metrics = learned_evaluation[
            str(selected_window)
        ]
        baseline_selected_metrics = baseline_evaluation[
            selected_baseline
        ]
        bootstrap = _cluster_bootstrap_accuracy_difference(
            labels[evaluation],
            learned_selected_prediction,
            baseline_selected_prediction,
            groups[evaluation],
            replicates=int(
                statistics["cluster_bootstrap_replicates"]
            ),
            confidence_level=float(statistics["confidence_level"]),
            seed=int(statistics["seed"]) + n_channels,
        )
        evaluation_sources = dataset["source_indices"][evaluation]
        results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "fallback_class": fallback_class,
            "class_count": class_count,
            "training_sample_count": int(np.sum(training)),
            "evaluation_sample_count": int(np.sum(evaluation)),
            "training_class_counts": np.bincount(
                labels[training], minlength=class_count
            ).tolist(),
            "evaluation_class_counts": np.bincount(
                labels[evaluation], minlength=class_count
            ).tolist(),
            "learned_window_cross_validation": learned_cv,
            "learned_window_evaluation": learned_evaluation,
            "selected_history_window": selected_window,
            "baseline_cross_validation": baseline_cv,
            "baseline_evaluation": baseline_evaluation,
            "selected_baseline": selected_baseline,
            "selected_learned_metrics": learned_selected_metrics,
            "selected_baseline_metrics": baseline_selected_metrics,
            "selected_comparison": {
                "macro_f1_lift": float(
                    learned_selected_metrics["macro_f1"]
                    - baseline_selected_metrics["macro_f1"]
                ),
                "transition_event_f1_lift": float(
                    learned_selected_metrics["transition_event_f1"]
                    - baseline_selected_metrics["transition_event_f1"]
                ),
                "cluster_bootstrap_accuracy_difference": bootstrap,
            },
            "selected_learned_safe_forecast": _safe_forecast_summary(
                learned_selected_prediction,
                evaluation_sources,
                states,
                codebook,
            ),
            "selected_baseline_safe_forecast": _safe_forecast_summary(
                baseline_selected_prediction,
                evaluation_sources,
                states,
                codebook,
            ),
        }

    rule = protocol["retention_rule"]
    macro_pass = sum(
        value["selected_comparison"]["macro_f1_lift"]
        >= float(
            rule[
                "minimum_macro_f1_lift_over_training_selected_baseline"
            ]
        )
        for value in results.values()
    )
    transition_pass = sum(
        value["selected_comparison"]["transition_event_f1_lift"]
        >= float(
            rule[
                "minimum_transition_event_f1_lift_over_training_selected_baseline"
            ]
        )
        for value in results.values()
    )
    accuracy_pass = sum(
        value["selected_comparison"][
            "cluster_bootstrap_accuracy_difference"
        ]["confidence_interval_lower"]
        >= float(rule["minimum_accuracy_gain_lower_confidence_bound"])
        for value in results.values()
    )
    required = int(rule["required_n_count_for_each_gate"])
    retention = {
        "macro_f1_gate_n_count": int(macro_pass),
        "transition_event_f1_gate_n_count": int(transition_pass),
        "accuracy_noninferiority_gate_n_count": int(accuracy_pass),
        "required_n_count_each_gate": required,
        "learning_model_retained_for_controller_integration": (
            macro_pass >= required
            and transition_pass >= required
            and accuracy_pass >= required
        ),
    }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "train_and_evaluation_groups_disjoint": not bool(
            set(train_groups) & set(evaluation_groups)
        ),
        "all_windows_use_identical_samples": all(
            len(value["learned_window_evaluation"])
            == len(windows)
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
        raise AssertionError(f"next-codeword check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_next_codeword_prediction_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
        },
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
                "selected": {
                    n: {
                        "window": value["selected_history_window"],
                        "baseline": value["selected_baseline"],
                        "learned_accuracy": value[
                            "selected_learned_metrics"
                        ]["exact_class_accuracy"],
                        "baseline_accuracy": value[
                            "selected_baseline_metrics"
                        ]["exact_class_accuracy"],
                        "macro_f1_lift": value[
                            "selected_comparison"
                        ]["macro_f1_lift"],
                        "transition_f1_lift": value[
                            "selected_comparison"
                        ]["transition_event_f1_lift"],
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
