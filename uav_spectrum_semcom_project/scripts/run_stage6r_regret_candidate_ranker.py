#!/usr/bin/env python
"""Screen lightweight regret-aware action candidate rankers for Stage-6R."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.run_stage6r_few_shot_adaptation import (  # noqa: E402
    _load_combined,
    _metrics_with_accounting,
    _queries,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6r_candidate_ranker import (  # noqa: E402
    candidate_feature_matrix,
    deterministic_training_subset,
    future_coverage_labels,
    pooled_context_features,
    select_ranked_codebook,
)
from spectrum_semcom.stage6r_regret_codebook import (  # noqa: E402
    evaluate_selected_actions,
    exhaustive_action_tuples,
    regret_matrix,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_regret_candidate_ranker_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/regret_candidate_ranker_v1/result.json"
)


def _chronological_campaign_indices(
    campaigns: np.ndarray, timestamps: np.ndarray, campaign: str
) -> np.ndarray:
    indices = np.flatnonzero(campaigns == campaign)
    return indices[np.argsort(timestamps[indices], kind="stable")]


def _context_slices(
    *,
    ordered_indices: np.ndarray,
    calibration_budget: int,
    origins: int,
    future_horizon: int,
    minimum_future: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    maximum_start = (
        ordered_indices.size - calibration_budget - minimum_future
    )
    if maximum_start < 0:
        return []
    final_start = max(
        0,
        min(
            maximum_start,
            ordered_indices.size
            - calibration_budget
            - future_horizon,
        ),
    )
    starts = np.unique(
        np.linspace(0, final_start, origins, dtype=int)
    ).tolist()
    output = []
    for start in starts:
        calibration = ordered_indices[
            start : start + calibration_budget
        ]
        future = ordered_indices[
            start
            + calibration_budget : start
            + calibration_budget
            + future_horizon
        ]
        if future.size >= minimum_future:
            output.append((calibration, future))
    return output


def _feature_matrix(
    *,
    calibration: np.ndarray,
    powers: np.ndarray,
    states: list,
    regrets: np.ndarray,
    actions: tuple[tuple[int, ...], ...],
    action_maxima: np.ndarray,
    epsilon_db: float,
    pooled_bins: int,
) -> np.ndarray:
    mean_optimal = np.mean(
        np.asarray(
            [states[int(index)].optimal_actions for index in calibration],
            dtype=np.float64,
        ),
        axis=0,
    )
    context = pooled_context_features(
        powers[calibration], output_bins=pooled_bins
    )
    return candidate_feature_matrix(
        calibration_regrets=regrets[calibration],
        actions=actions,
        action_maxima=action_maxima,
        mean_optimal_actions=mean_optimal,
        context_features=context,
        epsilon_db=epsilon_db,
    )


def _build_training_pairs(
    *,
    training_campaigns: list[str],
    campaigns: np.ndarray,
    timestamps: np.ndarray,
    powers: np.ndarray,
    states: list,
    regrets: np.ndarray,
    actions: tuple[tuple[int, ...], ...],
    action_maxima: np.ndarray,
    epsilon_db: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    features = []
    labels = []
    contexts = []
    settings = config["training_contexts"]
    for campaign in training_campaigns:
        ordered = _chronological_campaign_indices(
            campaigns, timestamps, campaign
        )
        for budget in map(int, settings["calibration_budgets"]):
            for calibration, future in _context_slices(
                ordered_indices=ordered,
                calibration_budget=budget,
                origins=int(settings["origins_per_training_activity"]),
                future_horizon=int(settings["future_horizon_scenes"]),
                minimum_future=int(settings["minimum_future_scenes"]),
            ):
                full_features = _feature_matrix(
                    calibration=calibration,
                    powers=powers,
                    states=states,
                    regrets=regrets,
                    actions=actions,
                    action_maxima=action_maxima,
                    epsilon_db=epsilon_db,
                    pooled_bins=int(
                        config["features"]["pooled_spectral_shape_bins"]
                    ),
                )
                future_labels = future_coverage_labels(
                    regrets[future], epsilon_db
                )
                calibration_coverage = np.mean(
                    regrets[calibration] <= epsilon_db + 1e-12,
                    axis=0,
                )
                selected = deterministic_training_subset(
                    calibration_coverage=calibration_coverage,
                    future_coverage=future_labels,
                    maximum_rows=int(
                        settings["maximum_candidate_rows_per_context"]
                    ),
                )
                features.append(full_features[selected])
                labels.append(future_labels[selected])
                contexts.append(
                    {
                        "campaign": campaign,
                        "calibration_scenes": int(calibration.size),
                        "future_scenes": int(future.size),
                        "candidate_rows": int(selected.size),
                    }
                )
    if not features:
        raise ValueError("no candidate-ranker training contexts were built")
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        {
            "context_count": len(contexts),
            "candidate_pair_count": int(
                sum(row["candidate_rows"] for row in contexts)
            ),
            "contexts": contexts,
        },
    )


def _models(config: dict) -> dict:
    settings = config["models"]
    ridge = settings["ridge"]
    tree = settings["hist_gradient_boosting"]
    mlp = settings["mlp_32"]
    return {
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(ridge["alpha"])),
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=float(tree["learning_rate"]),
            max_iter=int(tree["max_iter"]),
            max_leaf_nodes=int(tree["max_leaf_nodes"]),
            l2_regularization=float(tree["l2_regularization"]),
            random_state=int(tree["random_state"]),
        ),
        "mlp_32": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(
                    map(int, mlp["hidden_layer_sizes"])
                ),
                alpha=float(mlp["alpha"]),
                max_iter=int(mlp["max_iter"]),
                early_stopping=bool(mlp["early_stopping"]),
                random_state=int(mlp["random_state"]),
            ),
        ),
    }


def _aggregate(rows: list[dict], method: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    coverage = np.asarray(
        [row["future"]["coverage_rate"] for row in selected]
    )
    savings = np.asarray(
        [
            row["future"]["full_session_payload_savings_percentage"]
            for row in selected
        ]
    )
    return {
        "method": method,
        "fold_count": len(selected),
        "macro_future_coverage": float(np.mean(coverage)),
        "worst_activity_future_coverage": float(np.min(coverage)),
        "macro_full_session_payload_savings_percentage": float(
            np.mean(savings)
        ),
        "worst_full_session_payload_savings_percentage": float(
            np.min(savings)
        ),
        "maximum_serialized_model_size_bytes": int(
            max(row["model"]["serialized_size_bytes"] for row in selected)
        ),
        "mean_training_mae": float(
            np.mean([row["model"]["training_mae"] for row in selected])
        ),
        "mean_full_action_prediction_ms": float(
            np.mean(
                [
                    row["model"]["full_action_prediction_ms"]
                    for row in selected
                    if row["model"]["full_action_prediction_ms"] is not None
                ]
                or [0.0]
            )
        ),
        "all_unsafe_coded_action_counts_zero": all(
            row["future"]["unsafe_coded_action_count"] == 0
            for row in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite candidate-ranker result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adaptation_config_path = (
        PROJECT_DIR / config["inputs"]["adaptation_config"]
    )
    adaptation_result_path = (
        PROJECT_DIR / config["inputs"]["adaptation_result"]
    )
    temporal_config_path = (
        PROJECT_DIR / config["inputs"]["temporal_gate_config"]
    )
    temporal_result_path = (
        PROJECT_DIR / config["inputs"]["temporal_gate_result"]
    )
    adaptation = json.loads(
        adaptation_config_path.read_text(encoding="utf-8")
    )
    adaptation_result = json.loads(
        adaptation_result_path.read_text(encoding="utf-8")
    )
    temporal_result = json.loads(
        temporal_result_path.read_text(encoding="utf-8")
    )
    epsilon = float(adaptation["task"]["epsilon_db"])
    cvar_alpha = float(adaptation["task"]["cvar_alpha"])
    install_header = int(
        adaptation["protocol_accounting"][
            "adapted_codebook_install_header_bits"
        ]
    )
    bank_header = int(
        adaptation["protocol_accounting"][
            "preinstalled_bank_selection_header_bits"
        ]
    )
    campaigns_expected = list(adaptation["splitting"]["campaign_ids"])
    k = int(config["selection"]["codeword_count"])
    methods = list(config["models"])
    n_results = {}
    for n_channels in map(int, adaptation["task"]["n_channels"]):
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in powers
        ]
        actions = exhaustive_action_tuples(states)
        action_to_index = {
            value: index for index, value in enumerate(actions)
        }
        regrets = regret_matrix(states, actions)
        action_maxima = np.asarray(
            [
                n_channels - int(query.demand_channels)
                for query in queries
            ],
            dtype=np.float64,
        )
        exact_bits = int(
            adaptation_result["n_results"][str(n_channels)][
                "exact_action_payload_bits"
            ]
        )
        rows = []
        for holdout in campaigns_expected:
            training_campaigns = sorted(
                set(campaigns_expected) - {holdout}
            )
            train_x, train_y, training_summary = _build_training_pairs(
                training_campaigns=training_campaigns,
                campaigns=campaigns,
                timestamps=timestamps,
                powers=powers,
                states=states,
                regrets=regrets,
                actions=actions,
                action_maxima=action_maxima,
                epsilon_db=epsilon,
                config=config,
            )
            fitted = {}
            for name, model in _models(config).items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_x, train_y)
                prediction = np.clip(model.predict(train_x), 0.0, 1.0)
                fitted[name] = {
                    "model": model,
                    "training_mae": float(
                        mean_absolute_error(train_y, prediction)
                    ),
                    "serialized_size_bytes": len(
                        pickle.dumps(model, protocol=5)
                    ),
                    "convergence_warning_count": int(
                        sum(
                            issubclass(item.category, ConvergenceWarning)
                            for item in caught
                        )
                    ),
                }
            temporal_rows = [
                row
                for row in temporal_result["n_results"][str(n_channels)][
                    "rows"
                ]
                if row["k"] == k
                and row["holdout_campaign"] == holdout
            ]
            if len(temporal_rows) != 1:
                raise ValueError("temporal-gate fold is missing or duplicated")
            temporal = temporal_rows[0]
            calibration_count = int(temporal["calibration_scenes"])
            ordered = _chronological_campaign_indices(
                campaigns, timestamps, holdout
            )
            calibration = ordered[:calibration_count]
            future_indices = ordered[calibration_count:]
            evaluation_features = _feature_matrix(
                calibration=calibration,
                powers=powers,
                states=states,
                regrets=regrets,
                actions=actions,
                action_maxima=action_maxima,
                epsilon_db=epsilon,
                pooled_bins=int(
                    config["features"]["pooled_spectral_shape_bins"]
                ),
            )
            temporal_actions = tuple(
                tuple(map(int, row)) for row in temporal["selected_actions"]
            )
            temporal_selected = np.asarray(
                [action_to_index[value] for value in temporal_actions],
                dtype=int,
            )
            for method, fit in fitted.items():
                prediction_time_ms = None
                if temporal["decision"] == "OOD":
                    started = time.perf_counter()
                    predicted = np.clip(
                        fit["model"].predict(evaluation_features),
                        0.0,
                        1.0,
                    )
                    prediction_time_ms = float(
                        1000.0 * (time.perf_counter() - started)
                    )
                    selected = select_ranked_codebook(
                        calibration_regrets=regrets[calibration],
                        predicted_future_coverage=predicted,
                        actions=actions,
                        epsilon_db=epsilon,
                        codeword_count=k,
                        prediction_weight=float(
                            config["selection"]["prediction_weight"]
                        ),
                        redundancy_weight=float(
                            config["selection"]["redundancy_weight"]
                        ),
                    )
                    adapted = True
                    source = f"learned_ood:{method}"
                else:
                    selected = temporal_selected
                    adapted = False
                    source = temporal["decision"]
                base = evaluate_selected_actions(
                    regrets=regrets[future_indices],
                    selected_candidate_indices=selected,
                    epsilon_db=epsilon,
                    exact_action_payload_bits=exact_bits,
                    cvar_alpha=cvar_alpha,
                )
                future = _metrics_with_accounting(
                    method="bank_or_adapt",
                    metrics=base,
                    exact_bits=exact_bits,
                    calibration_count=calibration_count,
                    evaluation_count=int(future_indices.size),
                    selected_count=int(selected.size),
                    install_header_bits=install_header,
                    bank_header_bits=bank_header,
                    adapted=adapted,
                    oracle=False,
                )
                rows.append(
                    {
                        "holdout_campaign": holdout,
                        "method": method,
                        "calibration_scenes": calibration_count,
                        "temporal_decision": temporal["decision"],
                        "ranker_applied": temporal["decision"] == "OOD",
                        "codebook_source": source,
                        "selected_actions": [
                            list(actions[int(index)]) for index in selected
                        ],
                        "future": future,
                        "model": {
                            "training_mae": fit["training_mae"],
                            "serialized_size_bytes": fit[
                                "serialized_size_bytes"
                            ],
                            "convergence_warning_count": fit[
                                "convergence_warning_count"
                            ],
                            "full_action_prediction_ms": prediction_time_ms,
                            "training_context_count": training_summary[
                                "context_count"
                            ],
                            "training_candidate_pair_count": (
                                training_summary["candidate_pair_count"]
                            ),
                        },
                    }
                )
            print(
                f"Stage-6R ranker N={n_channels} "
                f"holdout={holdout} complete",
                flush=True,
            )
        aggregates = {
            method: _aggregate(rows, method) for method in methods
        }
        baseline = temporal_result["n_results"][str(n_channels)][
            "aggregate_by_k"
        ][str(k)]
        gate = config["primary_gate"]
        comparisons = {}
        for method, aggregate in aggregates.items():
            coverage_gain = float(
                aggregate["macro_future_coverage"]
                - baseline["macro_compact_future_coverage"]
            )
            savings_loss = float(
                baseline[
                    "macro_full_session_payload_savings_percentage"
                ]
                - aggregate[
                    "macro_full_session_payload_savings_percentage"
                ]
            )
            comparisons[method] = {
                "absolute_macro_coverage_gain": coverage_gain,
                "macro_payload_savings_loss_percentage_points": savings_loss,
                "gate_passed": bool(
                    coverage_gain
                    >= float(
                        gate["minimum_absolute_macro_coverage_gain"]
                    )
                    and aggregate["worst_activity_future_coverage"]
                    >= float(
                        gate["minimum_worst_activity_coverage"]
                    )
                    and savings_loss
                    <= float(
                        gate[
                            "maximum_macro_payload_savings_loss_percentage_points"
                        ]
                    )
                    and aggregate[
                        "maximum_serialized_model_size_bytes"
                    ]
                    <= int(
                        gate["maximum_serialized_model_size_bytes"]
                    )
                    and aggregate[
                        "all_unsafe_coded_action_counts_zero"
                    ]
                ),
            }
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "rows": rows,
            "baseline_temporal_gate": baseline,
            "aggregate_by_method": aggregates,
            "comparison_by_method": comparisons,
        }
    method_passes = {
        method: [
            int(n_key)
            for n_key, value in n_results.items()
            if value["comparison_by_method"][method]["gate_passed"]
        ]
        for method in methods
    }
    minimum_n = int(config["primary_gate"]["minimum_n_values_passing"])
    retained = [
        method
        for method, passed in method_passes.items()
        if len(passed) >= minimum_n
    ]
    result = {
        "version": "1.0",
        "status": "stage6r_regret_candidate_ranker_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "adaptation_config": sha256_file(adaptation_config_path),
            "adaptation_result": sha256_file(adaptation_result_path),
            "temporal_gate_config": sha256_file(temporal_config_path),
            "temporal_gate_result": sha256_file(temporal_result_path),
        },
        "n_results": n_results,
        "decision_summary": {
            "retention_gate_passed": bool(retained),
            "retained_methods": retained,
            "gate_passing_n_values_by_method": method_passes,
            "recommended_next_route": (
                "nested_ranker_selection_then_full_protocol_reintegration"
                if retained
                else "retain_temporal_analytic_codebook_and_record_ai_negative"
            ),
        },
        "checks": {
            "all_unsafe_coded_action_counts_zero": all(
                row["future"]["unsafe_coded_action_count"] == 0
                for value in n_results.values()
                for row in value["rows"]
            ),
            "heldout_future_used_for_training": False,
            "ranker_bypasses_exact_regret_projection": False,
            "new_final_data_accessed": False,
        },
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision_summary": result["decision_summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
