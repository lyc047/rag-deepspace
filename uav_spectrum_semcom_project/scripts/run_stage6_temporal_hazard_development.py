#!/usr/bin/env python
"""Evaluate causal one-step semantic-action hazard predictors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
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
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_temporal_hazard import (  # noqa: E402
    build_temporal_hazard_dataset,
)


def _model(c_value: float, *, maximum_iterations: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(maximum_iterations),
            random_state=0,
        ),
    )


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _threshold_for_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_recall: float,
) -> float:
    positives = int(np.sum(labels))
    if positives < 1:
        raise ValueError("training labels contain no positive hazards")
    candidates = np.unique(scores)[::-1]
    eligible = [
        float(threshold)
        for threshold in candidates
        if np.sum((scores >= threshold) & (labels == 1)) / positives
        >= target_recall
    ]
    if not eligible:
        return float(np.nextafter(np.min(scores), -np.inf))
    return float(max(eligible))


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "sample_count": int(labels.size),
        "positive_count": int(np.sum(labels)),
        "positive_prevalence": float(np.mean(labels)),
        "roc_auc": _safe_auc(labels, scores),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "brier_score": float(brier_score_loss(labels, scores)),
        "frozen_threshold": float(threshold),
        "recall_at_frozen_threshold": (
            float(tp / (tp + fn)) if tp + fn else None
        ),
        "false_positive_rate_at_frozen_threshold": (
            float(fp / (fp + tn)) if fp + tn else None
        ),
    }


def _fit_oof(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, list[float]]:
    predictions = np.full(labels.size, np.nan, dtype=np.float64)
    fold_scores = []
    for held_group in sorted(set(groups.tolist())):
        validation = groups == held_group
        training = ~validation
        if np.unique(labels[training]).size < 2:
            raise ValueError("training fold lacks both hazard classes")
        estimator = _model(
            c_value, maximum_iterations=maximum_iterations
        )
        estimator.fit(features[training], labels[training])
        fold_prediction = estimator.predict_proba(
            features[validation]
        )[:, 1]
        predictions[validation] = fold_prediction
        fold_scores.append(
            float(
                average_precision_score(
                    labels[validation], fold_prediction
                )
            )
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("out-of-fold predictions are incomplete")
    return predictions, fold_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_temporal_hazard_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/temporal_hazard_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "temporal_hazard_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite temporal hazard result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    safety = protocol["safety_boundary"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not safety[
            "prediction_may_only_schedule_early_updates_or_context_probes"
        ]
        or not safety["prediction_may_not_suppress_a_hard_regret_trigger"]
        or not safety["prediction_may_not_replace_context_belief_recovery"]
    ):
        raise ValueError("Stage-6 temporal hazard governance is invalid")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("temporal hazard cache hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
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
    learned = protocol["predictors"]["learned"]
    target_recall = float(
        protocol["predictors"]["operating_threshold"]["target_recall"]
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    results = {}
    for n_value in grid["n_channels"]:
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
        if np.any(training & evaluation) or not np.all(training | evaluation):
            raise ValueError("temporal hazard group split is invalid")
        x_train, y_train, g_train = (
            features[training],
            labels[training],
            groups[training],
        )
        x_eval, y_eval, g_eval = (
            features[evaluation],
            labels[evaluation],
            groups[evaluation],
        )

        cv_rows = []
        oof_by_c = {}
        for c_value in learned["c_grid"]:
            oof, fold_scores = _fit_oof(
                x_train,
                y_train,
                g_train,
                c_value=float(c_value),
                maximum_iterations=int(learned["maximum_iterations"]),
            )
            oof_by_c[float(c_value)] = oof
            cv_rows.append(
                {
                    "c": float(c_value),
                    "mean_group_average_precision": float(
                        np.mean(fold_scores)
                    ),
                    "group_average_precision": fold_scores,
                }
            )
        selected = sorted(
            cv_rows,
            key=lambda row: (
                -row["mean_group_average_precision"],
                row["c"],
            ),
        )[0]
        selected_c = float(selected["c"])
        oof = oof_by_c[selected_c]
        estimator = _model(
            selected_c,
            maximum_iterations=int(learned["maximum_iterations"]),
        )
        estimator.fit(x_train, y_train)
        learned_eval = estimator.predict_proba(x_eval)[:, 1]
        learned_threshold = _threshold_for_recall(
            y_train, oof, target_recall=target_recall
        )

        name_to_index = {
            name: index
            for index, name in enumerate(dataset["feature_names"])
        }
        training_prevalence = float(np.mean(y_train))
        baseline_train = {
            "constant_training_prevalence": np.full(
                y_train.size, training_prevalence
            ),
            "previous_transition_violation": x_train[
                :, name_to_index["previous_action_violation"]
            ],
            "past_mean_absolute_power_change": x_train[
                :, name_to_index["past_change_mean_db"]
            ],
        }
        baseline_eval = {
            "constant_training_prevalence": np.full(
                y_eval.size, training_prevalence
            ),
            "previous_transition_violation": x_eval[
                :, name_to_index["previous_action_violation"]
            ],
            "past_mean_absolute_power_change": x_eval[
                :, name_to_index["past_change_mean_db"]
            ],
        }
        model_metrics = {
            "learned_logistic": _metrics(
                y_eval,
                learned_eval,
                threshold=learned_threshold,
            )
        }
        baseline_thresholds = {}
        for name in protocol["predictors"]["baselines"]:
            threshold = _threshold_for_recall(
                y_train,
                baseline_train[name],
                target_recall=target_recall,
            )
            baseline_thresholds[name] = threshold
            scores = baseline_eval[name]
            if name == "past_mean_absolute_power_change":
                score_min = float(np.min(baseline_train[name]))
                score_max = float(np.max(baseline_train[name]))
                scale = max(score_max - score_min, 1e-12)
                scores = np.clip((scores - score_min) / scale, 0.0, 1.0)
                threshold = float(
                    np.clip((threshold - score_min) / scale, 0.0, 1.0)
                )
            model_metrics[name] = _metrics(
                y_eval, scores, threshold=threshold
            )

        coefficients = estimator.named_steps[
            "logisticregression"
        ].coef_[0]
        top_features = sorted(
            [
                {
                    "feature": name,
                    "standardized_coefficient": float(value),
                }
                for name, value in zip(
                    dataset["feature_names"], coefficients
                )
            ],
            key=lambda row: abs(row["standardized_coefficient"]),
            reverse=True,
        )[:8]
        per_group = {}
        for group in sorted(set(g_eval.tolist())):
            mask = g_eval == group
            per_group[group] = _metrics(
                y_eval[mask],
                learned_eval[mask],
                threshold=learned_threshold,
            )
        best_baseline_ap = max(
            model_metrics[name]["average_precision"]
            for name in protocol["predictors"]["baselines"]
        )
        results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "feature_count": int(features.shape[1]),
            "training_pair_count": int(np.sum(training)),
            "evaluation_pair_count": int(np.sum(evaluation)),
            "selected_c": selected_c,
            "cross_validation": cv_rows,
            "model_metrics": model_metrics,
            "learned_average_precision_lift_over_best_baseline": float(
                model_metrics["learned_logistic"]["average_precision"]
                - best_baseline_ap
            ),
            "learned_per_evaluation_group": per_group,
            "top_standardized_features": top_features,
        }

    rule = protocol["retention_rule"]
    lifts = [
        value[
            "learned_average_precision_lift_over_best_baseline"
        ]
        for value in results.values()
    ]
    auc_passes = sum(
        value["model_metrics"]["learned_logistic"]["roc_auc"] is not None
        and value["model_metrics"]["learned_logistic"]["roc_auc"] > 0.5
        for value in results.values()
    )
    retention = {
        "median_average_precision_lift": float(np.median(lifts)),
        "required_median_lift": float(
            rule[
                "learned_predictor_median_average_precision_lift_over_best_baseline_minimum"
            ]
        ),
        "roc_auc_above_random_scale_count": int(auc_passes),
        "required_roc_auc_scale_count": int(
            rule[
                "learned_predictor_roc_auc_must_exceed_random_for_at_least_n_scales"
            ]
        ),
    }
    retention["learned_predictor_retained_for_controller_integration"] = (
        retention["median_average_precision_lift"]
        >= retention["required_median_lift"]
        and retention["roc_auc_above_random_scale_count"]
        >= retention["required_roc_auc_scale_count"]
    )
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "train_and_evaluation_groups_disjoint": not bool(
            set(train_groups) & set(evaluation_groups)
        ),
        "all_pairs_stay_within_declared_groups": all(
            value["training_pair_count"] + value["evaluation_pair_count"]
            == 357
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
        raise AssertionError(f"temporal hazard check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_temporal_hazard_development_complete",
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
                "metrics": {
                    n: {
                        name: {
                            "roc_auc": metrics["roc_auc"],
                            "average_precision": metrics[
                                "average_precision"
                            ],
                        }
                        for name, metrics in value[
                            "model_metrics"
                        ].items()
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
