#!/usr/bin/env python
"""Run Stage-7 S7.2 site-robust lightweight risk models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.run_stage7_s7_1_strong_baselines import (  # noqa: E402
    _concatenate,
    _ece,
    _metrics,
    _safe_auc,
    _threshold_for_recall,
)
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    resolve_context_and_actions,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)
from spectrum_semcom.stage7_causal_baselines import (  # noqa: E402
    build_causal_task_risk_dataset,
)
from spectrum_semcom.stage7_lightweight_models import (  # noqa: E402
    fit_direct_logistic,
    fit_discrete_survival_logistic,
    model_matrix,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_s7_2_lightweight_models_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/s7_2_lightweight_models_v1/result.json"
)


def _fit_model(
    kind: str,
    matrix: np.ndarray,
    dataset: dict,
    indices: np.ndarray,
    *,
    numeric_count: int,
    horizon_position: int,
    maximum_horizon: int,
    regularization_c: float,
    class_weight: str | None,
    config: dict,
):
    selection = config["selection"]
    if kind == "direct_logistic":
        return fit_direct_logistic(
            matrix[indices],
            dataset["labels"][indices, horizon_position],
            numeric_count=numeric_count,
            regularization_c=regularization_c,
            class_weight=class_weight,
            maximum_iterations=int(selection["maximum_iterations"]),
            random_seed=int(selection["random_seed"]),
        )
    if kind == "discrete_time_survival_logistic":
        return fit_discrete_survival_logistic(
            matrix[indices],
            dataset["task_time_to_failure"][indices],
            numeric_count=numeric_count,
            maximum_horizon=maximum_horizon,
            regularization_c=regularization_c,
            class_weight=class_weight,
            maximum_iterations=int(selection["maximum_iterations"]),
            random_seed=int(selection["random_seed"]),
        )
    raise ValueError(f"unknown model kind {kind}")


def _predict_h1(model, matrix: np.ndarray) -> np.ndarray:
    return model.predict_risk(matrix, (1,))[:, 0]


def _site_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> list[dict]:
    rows = []
    for site in sorted(set(groups.tolist())):
        mask = groups == site
        site_labels = labels[mask]
        positive_count = int(np.sum(site_labels))
        rows.append(
            {
                "site": site,
                "sample_count": int(np.sum(mask)),
                "positive_count": positive_count,
                "negative_count": int(np.sum(mask) - positive_count),
                "positive_prevalence": float(np.mean(site_labels)),
                "roc_auc": _safe_auc(site_labels, scores[mask]),
                "average_precision": (
                    float(average_precision_score(site_labels, scores[mask]))
                    if positive_count
                    else None
                ),
            }
        )
    return rows


def _macro_site_metrics(rows: list[dict]) -> dict:
    evaluable = [
        row
        for row in rows
        if row["positive_count"] > 0
        and row["negative_count"] > 0
        and row["roc_auc"] is not None
    ]
    if not evaluable:
        raise ValueError("no site contains both risk classes")
    return {
        "evaluable_site_count": len(evaluable),
        "macro_within_site_auc": float(
            np.mean([row["roc_auc"] for row in evaluable])
        ),
        "macro_within_site_ap_lift": float(
            np.mean(
                [
                    row["average_precision"]
                    - row["positive_prevalence"]
                    for row in evaluable
                ]
            )
        ),
    }


def _site_bootstrap(
    rows: list[dict],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict:
    evaluable = [
        row
        for row in rows
        if row["positive_count"] > 0
        and row["negative_count"] > 0
        and row["roc_auc"] is not None
    ]
    auc = np.asarray([row["roc_auc"] for row in evaluable])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, auc.size, size=(int(replicates), auc.size)
    )
    values = np.mean(auc[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "unit": "evaluable_validation_site",
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "macro_auc_interval": [
            float(np.quantile(values, alpha / 2.0)),
            float(np.quantile(values, 1.0 - alpha / 2.0)),
        ],
    }


def _training_oof(
    kind: str,
    matrix: np.ndarray,
    dataset: dict,
    *,
    numeric_count: int,
    horizon_position: int,
    maximum_horizon: int,
    regularization_c: float,
    class_weight: str | None,
    config: dict,
) -> np.ndarray:
    groups = np.asarray(dataset["groups"])
    splitter = GroupKFold(
        n_splits=int(config["selection"]["training_group_folds"])
    )
    scores = np.full(groups.size, np.nan, dtype=np.float64)
    for fitting, held in splitter.split(matrix, groups=groups):
        model = _fit_model(
            kind,
            matrix,
            dataset,
            fitting,
            numeric_count=numeric_count,
            horizon_position=horizon_position,
            maximum_horizon=maximum_horizon,
            regularization_c=regularization_c,
            class_weight=class_weight,
            config=config,
        )
        scores[held] = _predict_h1(model, matrix[held])
    if not np.all(np.isfinite(scores)):
        raise ValueError("grouped OOF predictions are incomplete")
    return scores


def _select_training_hyperparameters(
    kind: str,
    matrix: np.ndarray,
    dataset: dict,
    *,
    numeric_count: int,
    horizon_position: int,
    maximum_horizon: int,
    class_weight: str | None,
    config: dict,
) -> dict:
    labels = dataset["labels"][:, horizon_position]
    candidates = []
    for regularization_c in config["selection"]["regularization_c_grid"]:
        scores = _training_oof(
            kind,
            matrix,
            dataset,
            numeric_count=numeric_count,
            horizon_position=horizon_position,
            maximum_horizon=maximum_horizon,
            regularization_c=float(regularization_c),
            class_weight=class_weight,
            config=config,
        )
        sites = _site_rows(labels, scores, dataset["groups"])
        macro = _macro_site_metrics(sites)
        candidates.append(
            {
                "regularization_c": float(regularization_c),
                "oof_scores": scores,
                "macro_within_site_auc": macro[
                    "macro_within_site_auc"
                ],
                "brier_score": float(brier_score_loss(labels, scores)),
                "expected_calibration_error": _ece(
                    labels,
                    scores,
                    int(config["selection"]["calibration_bins"]),
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_within_site_auc"],
            -row["brier_score"],
            -row["expected_calibration_error"],
            -row["regularization_c"],
        ),
    )
    return {
        "selected_regularization_c": selected["regularization_c"],
        "training_oof_macro_within_site_auc": selected[
            "macro_within_site_auc"
        ],
        "training_oof_brier_score": selected["brier_score"],
        "training_oof_expected_calibration_error": selected[
            "expected_calibration_error"
        ],
        "training_oof_scores": selected["oof_scores"],
        "grid": [
            {key: value for key, value in row.items() if key != "oof_scores"}
            for row in candidates
        ],
    }


def _load_datasets_for_n(
    *,
    n_channels: int,
    config: dict,
    protocol: dict,
    registry: dict,
    freeze: dict,
    loaded: dict,
    train_sites: list[str],
    validation_sites: list[str],
) -> tuple[dict, dict, dict]:
    frozen_n = freeze["n_artifacts"][str(n_channels)]
    queries = task_queries(n_channels, config["task"]["demand_ratios"])
    prototypes = {
        source: np.asarray(values, dtype=np.float64)
        for source, values in frozen_n["prototypes"].items()
    }
    train_rows = []
    validation_rows = []
    site_audit = {}
    selected_sites = set(train_sites + validation_sites)
    metadata_lookup = registry["selected_site_members"]
    for site in sorted(selected_sites):
        metadata = metadata_lookup[site]
        power = aggregate_frequency_bins(
            loaded[metadata["member"]], n_channels
        )
        resolution = resolve_context_and_actions(
            power=power,
            queries=queries,
            epsilon_db=float(config["task"]["epsilon_db"]),
            prototypes=prototypes,
            novelty_threshold=float(frozen_n["novelty_threshold"]),
            bank_actions=frozen_n["bank_actions"],
            gate=protocol["context_gate"],
            primary_k=int(config["task"]["primary_k"]),
        )
        states = resolution.pop("states")
        row = {
            "split": "train" if site in train_sites else "validation",
            "scene_count": len(states),
            "resolved": bool(resolution["resolved"]),
            "decision": resolution["decision"],
            "calibration_scenes": int(resolution["calibration_scenes"]),
        }
        if resolution["resolved"]:
            calibration = int(resolution["calibration_scenes"])
            codebook = codebook_from_selected_actions(
                states[:calibration], resolution["selected_actions"]
            )
            dataset = build_causal_task_risk_dataset(
                power[calibration:],
                states[calibration:],
                [site] * (len(states) - calibration),
                codebook,
                horizons=config["task"]["prediction_horizons_scenes"],
            )
            row["model_sample_count"] = int(len(dataset["features"]))
            (
                train_rows
                if site in train_sites
                else validation_rows
            ).append(dataset)
        else:
            row["model_sample_count"] = 0
        site_audit[site] = row
    return _concatenate(train_rows), _concatenate(validation_rows), site_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.2 result")
    config = read_json(args.config)
    paths = {
        name: PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    split = read_json(paths["development_split"])["split"]
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    s7_1 = read_json(paths["s7_1_result"])
    robustness = read_json(paths["s7_1_robustness"])
    train_sites = list(split["train"])
    validation_sites = list(split["validation"])
    internal_sites = list(split["internal_development_test"])
    if (
        len(train_sites) != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
        or int(access["roles"]["reserve"]["access_count"]) != 0
    ):
        raise ValueError("S7.2 frozen data governance mismatch")
    selected_sites = set(train_sites + validation_sites)
    if selected_sites.intersection(internal_sites):
        raise ValueError("internal test entered S7.2 selection")
    if not robustness["decision"]["any_robustness_caution"]:
        raise ValueError("S7.2 site-robust trigger is absent")
    metadata_lookup = registry["selected_site_members"]
    metadata_rows = [metadata_lookup[site] for site in sorted(selected_sites)]
    archive = Path(protocol["inputs"]["archive"])
    loaded = load_npy_members(
        archive, [row["member"] for row in metadata_rows]
    )
    started = time.perf_counter()
    n_results = {}
    passing_n_count = 0
    for n_position, n_channels in enumerate(
        map(int, config["task"]["n_channels"])
    ):
        train, validation, site_audit = _load_datasets_for_n(
            n_channels=n_channels,
            config=config,
            protocol=protocol,
            registry=registry,
            freeze=freeze,
            loaded=loaded,
            train_sites=train_sites,
            validation_sites=validation_sites,
        )
        train_matrix, numeric_count, feature_names = model_matrix(
            train,
            numeric_names=config["features"]["numeric"],
            categorical_names=config["features"]["categorical"],
        )
        validation_matrix, validation_numeric_count, validation_names = (
            model_matrix(
                validation,
                numeric_names=config["features"]["numeric"],
                categorical_names=config["features"]["categorical"],
            )
        )
        if (
            numeric_count != validation_numeric_count
            or feature_names != validation_names
        ):
            raise ValueError("train-validation feature schema mismatch")
        horizons = tuple(map(int, train["horizons"]))
        primary_horizon = int(config["task"]["primary_horizon_scenes"])
        primary_position = horizons.index(primary_horizon)
        maximum_horizon = max(horizons)
        y_train = train["labels"][:, primary_position]
        y_validation = validation["labels"][:, primary_position]
        constant_scores = np.full(
            y_validation.size, float(np.mean(y_train))
        )
        constant_brier = float(
            brier_score_loss(y_validation, constant_scores)
        )
        model_results = {}
        model_objects = {}
        for model_config in config["models"]:
            kind = model_config["name"]
            for class_weight in model_config["class_weight_options"]:
                suffix = "unweighted" if class_weight is None else class_weight
                name = f"{kind}__{suffix}"
                selected = _select_training_hyperparameters(
                    kind,
                    train_matrix,
                    train,
                    numeric_count=numeric_count,
                    horizon_position=primary_position,
                    maximum_horizon=maximum_horizon,
                    class_weight=class_weight,
                    config=config,
                )
                threshold = _threshold_for_recall(
                    y_train,
                    selected["training_oof_scores"],
                    float(
                        config["selection"][
                            "threshold_target_training_oof_recall"
                        ]
                    ),
                )
                all_indices = np.arange(len(train_matrix))
                model = _fit_model(
                    kind,
                    train_matrix,
                    train,
                    all_indices,
                    numeric_count=numeric_count,
                    horizon_position=primary_position,
                    maximum_horizon=maximum_horizon,
                    regularization_c=selected[
                        "selected_regularization_c"
                    ],
                    class_weight=class_weight,
                    config=config,
                )
                scores = _predict_h1(model, validation_matrix)
                sites = _site_rows(
                    y_validation, scores, validation["groups"]
                )
                macro = _macro_site_metrics(sites)
                aggregate = _metrics(
                    y_validation,
                    scores,
                    threshold=threshold,
                    constant_brier=constant_brier,
                    calibration_bins=int(
                        config["selection"]["calibration_bins"]
                    ),
                )
                edlv = next(
                    row
                    for row in sites
                    if row["site"]
                    == config["validation_model_selection"][
                        "dominant_positive_site"
                    ]
                )
                model_results[name] = {
                    "kind": kind,
                    "class_weight": class_weight,
                    "selected_regularization_c": selected[
                        "selected_regularization_c"
                    ],
                    "training_selection": {
                        key: value
                        for key, value in selected.items()
                        if key != "training_oof_scores"
                    },
                    "frozen_training_oof_threshold": threshold,
                    "validation_aggregate": aggregate,
                    "validation_macro": macro,
                    "dominant_positive_site_auc": edlv["roc_auc"],
                    "validation_site_metrics": sites,
                }
                model_objects[name] = model
        selected_name = max(
            model_results,
            key=lambda name: (
                model_results[name]["validation_macro"][
                    "macro_within_site_auc"
                ],
                model_results[name]["dominant_positive_site_auc"] or -1.0,
                -model_results[name]["validation_aggregate"][
                    "expected_calibration_error"
                ],
                -model_results[name]["validation_aggregate"][
                    "alert_fraction_at_frozen_threshold"
                ],
            ),
        )
        candidate = model_results[selected_name]
        selected_model = model_objects[selected_name]
        secondary = {}
        if candidate["kind"] == "discrete_time_survival_logistic":
            predictions = selected_model.predict_risk(
                validation_matrix, horizons
            )
            for position, horizon in enumerate(horizons):
                labels = validation["labels"][:, position]
                site_rows = _site_rows(
                    labels, predictions[:, position], validation["groups"]
                )
                secondary[str(horizon)] = {
                    "aggregate_auc": _safe_auc(
                        labels, predictions[:, position]
                    ),
                    "average_precision": float(
                        average_precision_score(
                            labels, predictions[:, position]
                        )
                    ),
                    **_macro_site_metrics(site_rows),
                }
        bootstrap = _site_bootstrap(
            candidate["validation_site_metrics"],
            replicates=int(config["bootstrap"]["replicates"]),
            confidence=float(config["bootstrap"]["confidence_level"]),
            seed=int(config["bootstrap"]["seed"]) + n_position * 1000,
        )
        s7_1_row = robustness["n_results"][str(n_channels)]
        edlv_improvement = float(
            candidate["dominant_positive_site_auc"]
            - s7_1_row["dominant_site_auc"]
        )
        gate = config["candidate_freeze_gate"]
        aggregate = candidate["validation_aggregate"]
        checks = {
            "macro_auc_lower_bound": bool(
                bootstrap["macro_auc_interval"][0]
                >= float(
                    gate[
                        "minimum_macro_within_site_auc_lower_bound"
                    ]
                )
            ),
            "dominant_site_auc_improvement": bool(
                edlv_improvement
                >= float(
                    gate[
                        "minimum_dominant_site_auc_improvement_over_s7_1"
                    ]
                )
            ),
            "positive_brier_skill": bool(
                aggregate["brier_skill_over_constant"] is not None
                and aggregate["brier_skill_over_constant"] > 0.0
            ),
            "calibration": bool(
                aggregate["expected_calibration_error"]
                <= float(gate["maximum_expected_calibration_error"])
            ),
            "recall": bool(
                aggregate["recall_at_frozen_threshold"] is not None
                and aggregate["recall_at_frozen_threshold"]
                >= float(
                    gate[
                        "minimum_recall_at_training_oof_threshold"
                    ]
                )
            ),
            "alert_fraction": bool(
                aggregate["alert_fraction_at_frozen_threshold"]
                <= float(
                    gate[
                        "maximum_alert_fraction_at_training_oof_threshold"
                    ]
                )
            ),
        }
        passed = all(checks.values())
        passing_n_count += int(passed)
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "resolved_train_site_count": len(
                set(train["groups"].tolist())
            ),
            "resolved_validation_site_count": len(
                set(validation["groups"].tolist())
            ),
            "feature_names": feature_names,
            "site_audit": site_audit,
            "models": model_results,
            "selected_model": selected_name,
            "selected_model_secondary_horizons": secondary,
            "selected_model_site_bootstrap": bootstrap,
            "s7_1_comparison": {
                "s7_1_macro_within_site_auc": s7_1_row[
                    "macro_within_site_auc"
                ],
                "s7_2_macro_within_site_auc": candidate[
                    "validation_macro"
                ]["macro_within_site_auc"],
                "s7_1_dominant_site_auc": s7_1_row[
                    "dominant_site_auc"
                ],
                "s7_2_dominant_site_auc": candidate[
                    "dominant_positive_site_auc"
                ],
                "dominant_site_auc_improvement": edlv_improvement,
            },
            "freeze_gate_checks": checks,
            "freeze_gate_passed": passed,
        }
        print(
            f"S7.2 N={n_channels} selected={selected_name} "
            f"pass={passed}",
            flush=True,
        )
    required = int(config["candidate_freeze_gate"]["minimum_n_count"])
    if passing_n_count >= required:
        next_action = config["decision_policy"]["if_freeze_gate_passes"]
    elif any(
        row["models"][row["selected_model"]]["validation_macro"][
            "macro_within_site_auc"
        ]
        > row["s7_1_comparison"]["s7_1_macro_within_site_auc"]
        for row in n_results.values()
    ):
        next_action = config["decision_policy"][
            "if_predictive_but_freeze_gate_fails"
        ]
    else:
        next_action = config["decision_policy"]["if_no_within_site_gain"]
    result = {
        "version": "1.0",
        "status": "stage7_s7_2_lightweight_models_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        },
        "loaded_sites": {
            "train": train_sites,
            "validation": validation_sites,
            "internal_development_test": [],
            "reserve": [],
        },
        "n_results": n_results,
        "decision": {
            "passing_n_count": passing_n_count,
            "required_n_count": required,
            "candidate_freeze_gate_passed": passing_n_count >= required,
            "next_action": next_action,
        },
        "governance_checks": {
            "site_identifier_used_as_feature": False,
            "internal_development_test_not_loaded": True,
            "reserve_remained_unread": True,
            "external_final_claim_forbidden": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": config["claim_boundary"],
        "s7_1_registered_aggregate_decision_preserved": s7_1["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
