#!/usr/bin/env python
"""Run Stage-7 S7.1 cross-site causal strong baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

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
    dwell_bin,
    quantile_lookup_probabilities,
    smoothed_lookup_probabilities,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_s7_1_strong_baselines_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/s7_1_strong_baselines_v1/result.json"
)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size >= 2
        else None
    )


def _safe_average_precision(
    labels: np.ndarray, scores: np.ndarray
) -> float | None:
    return (
        float(average_precision_score(labels, scores))
        if np.any(labels == 1)
        else None
    )


def _threshold_for_recall(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> float:
    positives = int(np.sum(labels))
    if positives < 1:
        raise ValueError("threshold labels contain no positive rows")
    candidates = np.unique(scores)[::-1]
    eligible = [
        float(value)
        for value in candidates
        if np.sum((scores >= value) & (labels == 1)) / positives
        >= float(target)
    ]
    return (
        max(eligible)
        if eligible
        else float(np.nextafter(np.min(scores), -np.inf))
    )


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.minimum(
        np.searchsorted(edges, scores, side="right") - 1,
        int(bins) - 1,
    )
    total = labels.size
    value = 0.0
    for index in range(int(bins)):
        mask = indices == index
        if np.any(mask):
            value += (
                np.sum(mask)
                / total
                * abs(float(np.mean(scores[mask]) - np.mean(labels[mask])))
            )
    return float(value)


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    constant_brier: float,
    calibration_bins: int,
) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    predictions = scores >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    brier = float(brier_score_loss(labels, scores))
    prevalence = float(np.mean(labels))
    return {
        "sample_count": int(labels.size),
        "positive_count": int(np.sum(labels)),
        "positive_prevalence": prevalence,
        "roc_auc": _safe_auc(labels, scores),
        "average_precision": float(average_precision_score(labels, scores)),
        "average_precision_lift_over_prevalence": float(
            average_precision_score(labels, scores) - prevalence
        ),
        "brier_score": brier,
        "brier_skill_over_constant": float(
            1.0 - brier / constant_brier
        )
        if constant_brier > 0
        else None,
        "expected_calibration_error": _ece(
            labels, scores, calibration_bins
        ),
        "frozen_threshold": float(threshold),
        "recall_at_frozen_threshold": (
            float(tp / (tp + fn)) if tp + fn else None
        ),
        "false_positive_rate_at_frozen_threshold": (
            float(fp / (fp + tn)) if fp + tn else None
        ),
        "alert_fraction_at_frozen_threshold": float(np.mean(predictions)),
    }


def _baseline_scores(
    name: str,
    train: dict,
    train_labels: np.ndarray,
    evaluation: dict,
    config: dict,
) -> np.ndarray:
    features = {
        value: index
        for index, value in enumerate(train["feature_names"])
    }
    prior_strength = float(config["lookup"]["prior_strength"])
    bins = int(config["lookup"]["continuous_quantile_bins"])
    if name == "constant_training_prevalence":
        return np.full(
            len(evaluation["features"]), float(np.mean(train_labels))
        )
    if name == "previous_action_violation":
        index = features["previous_action_violation"]
        return smoothed_lookup_probabilities(
            train["features"][:, index].astype(int).tolist(),
            train_labels,
            evaluation["features"][:, index].astype(int).tolist(),
            prior_strength=prior_strength,
        )
    if name == "past_power_change_quantile":
        index = features["past_change_mean_db"]
        return quantile_lookup_probabilities(
            train["features"][:, index],
            train_labels,
            evaluation["features"][:, index],
            bin_count=bins,
            prior_strength=prior_strength,
        )
    if name == "runner_up_margin_quantile":
        index = features["minimum_runner_up_margin_db"]
        return quantile_lookup_probabilities(
            train["features"][:, index],
            train_labels,
            evaluation["features"][:, index],
            bin_count=bins,
            prior_strength=prior_strength,
        )
    if name == "action_duration_lookup":
        return smoothed_lookup_probabilities(
            dwell_bin(train["action_dwell_scenes"]).tolist(),
            train_labels,
            dwell_bin(evaluation["action_dwell_scenes"]).tolist(),
            prior_strength=prior_strength,
        )
    if name == "symbol_markov_duration_lookup":
        train_keys = list(
            zip(
                train["previous_symbol"].tolist(),
                train["current_symbol"].tolist(),
                dwell_bin(train["action_dwell_scenes"]).tolist(),
            )
        )
        evaluation_keys = list(
            zip(
                evaluation["previous_symbol"].tolist(),
                evaluation["current_symbol"].tolist(),
                dwell_bin(evaluation["action_dwell_scenes"]).tolist(),
            )
        )
        return smoothed_lookup_probabilities(
            train_keys,
            train_labels,
            evaluation_keys,
            prior_strength=prior_strength,
        )
    raise ValueError(f"unknown baseline {name}")


def _concatenate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no resolved semantic site datasets")
    return {
        "features": np.concatenate([row["features"] for row in rows]),
        "feature_names": rows[0]["feature_names"],
        "labels": np.concatenate([row["labels"] for row in rows]),
        "task_time_to_failure": np.concatenate(
            [row["task_time_to_failure"] for row in rows]
        ),
        "horizons": rows[0]["horizons"],
        "groups": np.concatenate([row["groups"] for row in rows]),
        "source_indices": np.concatenate(
            [row["source_indices"] for row in rows]
        ),
        "current_symbol": np.concatenate(
            [row["current_symbol"] for row in rows]
        ),
        "previous_symbol": np.concatenate(
            [row["previous_symbol"] for row in rows]
        ),
        "action_dwell_scenes": np.concatenate(
            [row["action_dwell_scenes"] for row in rows]
        ),
    }


def _oof_scores(
    name: str,
    train: dict,
    labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    for held_site in sorted(set(train["groups"].tolist())):
        held = train["groups"] == held_site
        fitting = ~held
        fit = {
            key: (
                value[fitting]
                if isinstance(value, np.ndarray)
                and value.shape[:1] == (labels.size,)
                else value
            )
            for key, value in train.items()
        }
        evaluation = {
            key: (
                value[held]
                if isinstance(value, np.ndarray)
                and value.shape[:1] == (labels.size,)
                else value
            )
            for key, value in train.items()
        }
        scores[held] = _baseline_scores(
            name, fit, labels[fitting], evaluation, config
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("out-of-fold baseline scores are incomplete")
    return scores


def _context_labels(
    dataset: dict,
    *,
    horizon: int,
    trajectories: int,
    reset_probability: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    labels = []
    dwell = []
    phase = []
    groups = []
    for site_position, site in enumerate(sorted(set(dataset["groups"].tolist()))):
        mask = dataset["groups"] == site
        count = int(np.sum(mask))
        source = dataset["source_indices"][mask]
        site_dwell = dataset["action_dwell_scenes"][mask]
        rng = np.random.default_rng(seed + site_position * 10_000)
        for trajectory in range(int(trajectories)):
            events = rng.random(count + int(horizon)) < float(
                reset_probability
            )
            target = np.asarray(
                [
                    np.any(events[index + 1 : index + 1 + int(horizon)])
                    for index in range(count)
                ],
                dtype=np.uint8,
            )
            labels.append(target)
            dwell.append(site_dwell)
            phase.append(source % 10)
            groups.append(
                np.asarray([f"{site}::t{trajectory}"] * count)
            )
    return np.concatenate(labels), {
        "dwell": np.concatenate(dwell),
        "phase": np.concatenate(phase),
        "groups": np.concatenate(groups),
    }


def _context_audit(
    train: dict,
    validation: dict,
    *,
    horizon: int,
    n_position: int,
    config: dict,
) -> dict:
    audit = config["context_risk_audit"]
    probability = float(audit["reset_probability_per_scene"])
    trajectories = int(audit["trajectories"])
    seed = int(audit["random_seed"]) + n_position * 1_000_000 + horizon * 1000
    train_labels, train_features = _context_labels(
        train,
        horizon=horizon,
        trajectories=trajectories,
        reset_probability=probability,
        seed=seed,
    )
    labels, features = _context_labels(
        validation,
        horizon=horizon,
        trajectories=trajectories,
        reset_probability=probability,
        seed=seed + 500_000,
    )
    analytic = float(1.0 - (1.0 - probability) ** int(horizon))
    constant = np.full(labels.size, analytic)
    duration = smoothed_lookup_probabilities(
        dwell_bin(train_features["dwell"]).tolist(),
        train_labels,
        dwell_bin(features["dwell"]).tolist(),
        prior_strength=float(config["lookup"]["prior_strength"]),
    )
    phase = smoothed_lookup_probabilities(
        train_features["phase"].tolist(),
        train_labels,
        features["phase"].tolist(),
        prior_strength=float(config["lookup"]["prior_strength"]),
    )
    constant_brier = float(brier_score_loss(labels, constant))
    threshold = analytic
    return {
        "analytic_probability": analytic,
        "empirical_validation_prevalence": float(np.mean(labels)),
        "absolute_prevalence_error": float(abs(np.mean(labels) - analytic)),
        "constant_analytic_hazard": _metrics(
            labels,
            constant,
            threshold=threshold,
            constant_brier=constant_brier,
            calibration_bins=int(config["lookup"]["calibration_bins"]),
        ),
        "action_duration_lookup": _metrics(
            labels,
            duration,
            threshold=_threshold_for_recall(
                train_labels,
                smoothed_lookup_probabilities(
                    dwell_bin(train_features["dwell"]).tolist(),
                    train_labels,
                    dwell_bin(train_features["dwell"]).tolist(),
                    prior_strength=float(config["lookup"]["prior_strength"]),
                ),
                float(config["lookup"]["threshold_target_training_oof_recall"]),
            ),
            constant_brier=constant_brier,
            calibration_bins=int(config["lookup"]["calibration_bins"]),
        ),
        "heartbeat_phase_lookup": _metrics(
            labels,
            phase,
            threshold=_threshold_for_recall(
                train_labels,
                smoothed_lookup_probabilities(
                    train_features["phase"].tolist(),
                    train_labels,
                    train_features["phase"].tolist(),
                    prior_strength=float(config["lookup"]["prior_strength"]),
                ),
                float(config["lookup"]["threshold_target_training_oof_recall"]),
            ),
            constant_brier=constant_brier,
            calibration_bins=int(config["lookup"]["calibration_bins"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.1 result")
    config = read_json(args.config)
    paths = {
        name: PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    governance = read_json(paths["governance"])
    split = read_json(paths["development_split"])["split"]
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    scan_config = read_json(paths["matched_scan_config"])
    s7_0 = read_json(paths["s7_0_result"])
    if not s7_0["decision"]["oracle_headroom_gate_passed"]:
        raise ValueError("S7.0 oracle gate did not pass")
    if int(access["roles"]["reserve"]["access_count"]) != 0:
        raise ValueError("reserve values must remain unread")
    train_sites = list(split["train"])
    validation_sites = list(split["validation"])
    internal_sites = list(split["internal_development_test"])
    if (
        len(train_sites) != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
        or len(internal_sites)
        != int(
            config["data"]["expected_internal_development_test_site_count"]
        )
    ):
        raise ValueError("frozen split count mismatch")
    selected_sites = set(train_sites + validation_sites)
    metadata_lookup = registry["selected_site_members"]
    metadata_rows = [metadata_lookup[site] for site in sorted(selected_sites)]
    if any(row["site"] in internal_sites for row in metadata_rows):
        raise ValueError("internal development test entered S7.1 selection")
    archive = Path(protocol["inputs"]["archive"])
    loaded = load_npy_members(
        archive, [row["member"] for row in metadata_rows]
    )
    started = time.perf_counter()
    n_results = {}
    task_structure_passes = 0
    full_baseline_passes = 0
    context_consistency_passes = 0
    for n_position, n_channels in enumerate(
        map(int, config["task"]["n_channels"])
    ):
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        queries = task_queries(n_channels, config["task"]["demand_ratios"])
        prototypes = {
            source: np.asarray(values, dtype=np.float64)
            for source, values in frozen_n["prototypes"].items()
        }
        train_rows = []
        validation_rows = []
        site_output = {}
        for metadata in metadata_rows:
            site = metadata["site"]
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
            output = {
                "split": (
                    "train" if site in train_sites else "validation"
                ),
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
                output["model_sample_count"] = int(
                    len(dataset["features"])
                )
                (
                    train_rows
                    if site in train_sites
                    else validation_rows
                ).append(dataset)
            else:
                output["model_sample_count"] = 0
            site_output[site] = output
        train = _concatenate(train_rows)
        validation = _concatenate(validation_rows)
        horizon_results = {}
        for horizon_position, horizon in enumerate(train["horizons"]):
            y_train = train["labels"][:, horizon_position]
            y_validation = validation["labels"][:, horizon_position]
            train_prevalence = float(np.mean(y_train))
            constant_validation = np.full(
                y_validation.size, train_prevalence
            )
            constant_brier = float(
                brier_score_loss(y_validation, constant_validation)
            )
            baseline_results = {}
            for name in config["task_risk_baselines"]:
                oof = _oof_scores(name, train, y_train, config)
                threshold = _threshold_for_recall(
                    y_train,
                    oof,
                    float(
                        config["lookup"][
                            "threshold_target_training_oof_recall"
                        ]
                    ),
                )
                scores = _baseline_scores(
                    name, train, y_train, validation, config
                )
                baseline_results[name] = _metrics(
                    y_validation,
                    scores,
                    threshold=threshold,
                    constant_brier=constant_brier,
                    calibration_bins=int(
                        config["lookup"]["calibration_bins"]
                    ),
                )
            best_name = max(
                baseline_results,
                key=lambda name: (
                    baseline_results[name]["average_precision"],
                    baseline_results[name]["roc_auc"] or -1.0,
                ),
            )
            per_site = {}
            best_scores = _baseline_scores(
                best_name, train, y_train, validation, config
            )
            for site in sorted(set(validation["groups"].tolist())):
                mask = validation["groups"] == site
                per_site[site] = {
                    "sample_count": int(np.sum(mask)),
                    "positive_prevalence": float(
                        np.mean(y_validation[mask])
                    ),
                    "roc_auc": _safe_auc(
                        y_validation[mask], best_scores[mask]
                    ),
                    "average_precision": _safe_average_precision(
                        y_validation[mask], best_scores[mask]
                    ),
                }
            horizon_results[str(horizon)] = {
                "horizon_scenes": int(horizon),
                "train_sample_count": int(y_train.size),
                "validation_sample_count": int(y_validation.size),
                "train_positive_prevalence": train_prevalence,
                "validation_positive_prevalence": float(
                    np.mean(y_validation)
                ),
                "baselines": baseline_results,
                "best_baseline": best_name,
                "best_baseline_per_validation_site": per_site,
            }
        one = horizon_results[
            str(
                config["decision_rules"][
                    "task_has_useful_baseline_structure"
                ]["evaluate_horizon_scenes"]
            )
        ]
        best = one["baselines"][one["best_baseline"]]
        structure_rule = config["decision_rules"][
            "task_has_useful_baseline_structure"
        ]
        structure_pass = bool(
            best["roc_auc"] is not None
            and best["roc_auc"] >= float(structure_rule["minimum_auc"])
            and best["average_precision_lift_over_prevalence"]
            >= float(structure_rule["minimum_ap_lift_over_prevalence"])
        )
        prediction_rule = config["decision_rules"][
            "baseline_already_meets_stage7_prediction_gate"
        ]
        full_pass = bool(
            best["roc_auc"] is not None
            and best["roc_auc"] >= float(prediction_rule["minimum_auc"])
            and best["average_precision_lift_over_prevalence"]
            >= float(
                prediction_rule["minimum_ap_lift_over_prevalence"]
            )
            and best["brier_skill_over_constant"] is not None
            and best["brier_skill_over_constant"] > 0.0
            and best["expected_calibration_error"]
            <= float(prediction_rule["maximum_ece"])
            and best["recall_at_frozen_threshold"] is not None
            and best["recall_at_frozen_threshold"]
            >= float(prediction_rule["minimum_recall"])
            and best["alert_fraction_at_frozen_threshold"]
            <= float(prediction_rule["maximum_alert_fraction"])
        )
        context = {
            str(horizon): _context_audit(
                train,
                validation,
                horizon=int(horizon),
                n_position=n_position,
                config=config,
            )
            for horizon in train["horizons"]
        }
        context_rule = config["decision_rules"][
            "context_iid_empirical_consistency"
        ]
        context_one = context["1"]
        dwell_auc = context_one["action_duration_lookup"]["roc_auc"]
        context_pass = bool(
            dwell_auc is not None
            and float(context_rule["minimum_dwell_auc"])
            <= dwell_auc
            <= float(context_rule["maximum_dwell_auc"])
            and context_one["absolute_prevalence_error"]
            <= float(
                context_rule["maximum_absolute_prevalence_error"]
            )
        )
        task_structure_passes += int(structure_pass)
        full_baseline_passes += int(full_pass)
        context_consistency_passes += int(context_pass)
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "resolved_train_site_count": len(train_rows),
            "resolved_validation_site_count": len(validation_rows),
            "excluded_unresolved_train_site_count": (
                len(train_sites) - len(train_rows)
            ),
            "excluded_unresolved_validation_site_count": (
                len(validation_sites) - len(validation_rows)
            ),
            "site_audit": site_output,
            "task_risk": horizon_results,
            "context_iid_audit": context,
            "decision": {
                "task_has_useful_baseline_structure": structure_pass,
                "best_baseline_already_meets_full_prediction_gate": full_pass,
                "context_iid_consistency_pass": context_pass,
            },
        }
        print(f"S7.1 N={n_channels} complete", flush=True)
    structure_required = int(
        config["decision_rules"]["task_has_useful_baseline_structure"][
            "minimum_n_count"
        ]
    )
    full_required = int(
        config["decision_rules"][
            "baseline_already_meets_stage7_prediction_gate"
        ]["minimum_n_count"]
    )
    context_required = 3
    if task_structure_passes >= structure_required:
        next_action = (
            "freeze_strong_baseline_candidate_before_internal_test"
            if full_baseline_passes >= full_required
            else config["decision_rules"][
                "if_task_structure_exists_but_baseline_gate_fails"
            ]
        )
    else:
        next_action = config["decision_rules"]["if_no_task_structure"]
    result = {
        "version": "1.0",
        "status": "stage7_s7_1_strong_baselines_complete",
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
            "task_structure_passing_n_count": task_structure_passes,
            "task_structure_required_n_count": structure_required,
            "full_prediction_gate_passing_n_count": full_baseline_passes,
            "full_prediction_gate_required_n_count": full_required,
            "context_iid_consistency_passing_n_count": (
                context_consistency_passes
            ),
            "context_iid_consistency_required_n_count": context_required,
            "next_action": next_action,
            "context_action": config["decision_rules"]["context_action"],
        },
        "governance_checks": {
            "stage7_governance_status": governance["status"],
            "internal_development_test_not_loaded": True,
            "reserve_remained_unread": (
                int(access["roles"]["reserve"]["access_count"]) == 0
            ),
            "external_final_claim_forbidden": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
