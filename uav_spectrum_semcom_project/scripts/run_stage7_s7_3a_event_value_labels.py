#!/usr/bin/env python
"""Run S7.3a task-event decomposition and forced-refresh value audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.run_stage7_s7_1_strong_baselines import _safe_auc  # noqa: E402
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
    dwell_bin,
    smoothed_lookup_probabilities,
)
from spectrum_semcom.stage7_event_value import (  # noqa: E402
    build_task_event_value_labels,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_s7_3a_event_value_labels_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/s7_3a_event_value_labels_v1/result.json"
)


def _concatenate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no resolved event-value datasets")
    nested = (
        "persistent_failure",
        "net_positive_by_clean_scene_value_bits",
    )
    result = {}
    for key in rows[0]:
        if key in nested:
            result[key] = {
                subkey: np.concatenate([row[key][subkey] for row in rows])
                for subkey in rows[0][key]
            }
        elif key in ("incremental_bits", "horizon", "refresh_offset"):
            values = {row[key] for row in rows}
            if len(values) != 1:
                raise ValueError(f"inconsistent event metadata {key}")
            result[key] = values.pop()
        else:
            result[key] = np.concatenate([row[key] for row in rows])
    return result


def _site_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> list[dict]:
    rows = []
    for site in sorted(set(groups.tolist())):
        mask = groups == site
        target = labels[mask]
        positives = int(np.sum(target))
        rows.append(
            {
                "site": site,
                "sample_count": int(np.sum(mask)),
                "positive_count": positives,
                "positive_prevalence": float(np.mean(target)),
                "roc_auc": _safe_auc(target, scores[mask]),
                "average_precision": (
                    float(average_precision_score(target, scores[mask]))
                    if positives
                    else None
                ),
            }
        )
    return rows


def _distribution(labels: np.ndarray, groups: np.ndarray) -> dict:
    target = np.asarray(labels, dtype=np.uint8)
    counts = {
        site: int(np.sum(target[groups == site]))
        for site in sorted(set(groups.tolist()))
    }
    total = int(np.sum(target))
    dominant_site = max(counts, key=counts.get)
    return {
        "sample_count": int(target.size),
        "positive_count": total,
        "positive_prevalence": float(np.mean(target)),
        "positive_site_count": int(
            sum(value > 0 for value in counts.values())
        ),
        "dominant_positive_site": dominant_site,
        "dominant_positive_count": counts[dominant_site],
        "dominant_site_event_share": (
            float(counts[dominant_site] / total) if total else None
        ),
        "positive_count_by_site": counts,
    }


def _baseline_scores(
    name: str,
    train: dict,
    train_labels: np.ndarray,
    validation: dict,
    *,
    prior_strength: float,
) -> np.ndarray:
    if name == "action_duration_lookup":
        train_keys = dwell_bin(train["action_dwell_scenes"]).tolist()
        validation_keys = dwell_bin(
            validation["action_dwell_scenes"]
        ).tolist()
    elif name == "symbol_markov_duration_lookup":
        train_keys = list(
            zip(
                train["previous_symbol"].tolist(),
                train["current_symbol"].tolist(),
                dwell_bin(train["action_dwell_scenes"]).tolist(),
            )
        )
        validation_keys = list(
            zip(
                validation["previous_symbol"].tolist(),
                validation["current_symbol"].tolist(),
                dwell_bin(validation["action_dwell_scenes"]).tolist(),
            )
        )
    else:
        raise ValueError(f"unknown strong baseline {name}")
    return smoothed_lookup_probabilities(
        train_keys,
        train_labels,
        validation_keys,
        prior_strength=prior_strength,
    )


def _predictability(
    train: dict,
    validation: dict,
    *,
    label_key: str,
    baselines: list[str],
    prior_strength: float,
) -> dict:
    train_labels = np.asarray(train[label_key], dtype=np.uint8)
    labels = np.asarray(validation[label_key], dtype=np.uint8)
    results = {}
    score_lookup = {}
    for name in baselines:
        scores = _baseline_scores(
            name,
            train,
            train_labels,
            validation,
            prior_strength=prior_strength,
        )
        sites = _site_metrics(labels, scores, validation["groups"])
        evaluable = [
            row for row in sites if row["roc_auc"] is not None
        ]
        results[name] = {
            "aggregate_auc": _safe_auc(labels, scores),
            "average_precision": (
                float(average_precision_score(labels, scores))
                if np.any(labels)
                else None
            ),
            "macro_within_site_auc": (
                float(np.mean([row["roc_auc"] for row in evaluable]))
                if evaluable
                else None
            ),
            "evaluable_site_count": len(evaluable),
            "site_metrics": sites,
        }
        score_lookup[name] = scores
    best_name = max(
        results,
        key=lambda name: (
            results[name]["macro_within_site_auc"]
            if results[name]["macro_within_site_auc"] is not None
            else -1.0,
            results[name]["aggregate_auc"]
            if results[name]["aggregate_auc"] is not None
            else -1.0,
        ),
    )
    return {"baselines": results, "best_baseline": best_name}


def _training_oof_predictability(
    train: dict,
    *,
    label_key: str,
    baselines: list[str],
    prior_strength: float,
) -> dict:
    labels = np.asarray(train[label_key], dtype=np.uint8)
    groups = np.asarray(train["groups"])
    results = {}
    for name in baselines:
        scores = np.full(labels.size, np.nan, dtype=np.float64)
        for held_site in sorted(set(groups.tolist())):
            held = groups == held_site
            fitting = ~held
            fit = {
                "action_dwell_scenes": train["action_dwell_scenes"][
                    fitting
                ],
                "previous_symbol": train["previous_symbol"][fitting],
                "current_symbol": train["current_symbol"][fitting],
            }
            evaluation = {
                "action_dwell_scenes": train["action_dwell_scenes"][held],
                "previous_symbol": train["previous_symbol"][held],
                "current_symbol": train["current_symbol"][held],
            }
            scores[held] = _baseline_scores(
                name,
                fit,
                labels[fitting],
                evaluation,
                prior_strength=prior_strength,
            )
        sites = _site_metrics(labels, scores, groups)
        evaluable = [
            row for row in sites if row["roc_auc"] is not None
        ]
        results[name] = {
            "aggregate_auc": _safe_auc(labels, scores),
            "macro_within_site_auc": (
                float(np.mean([row["roc_auc"] for row in evaluable]))
                if evaluable
                else None
            ),
            "evaluable_site_count": len(evaluable),
            "site_metrics": sites,
        }
    best_name = max(
        results,
        key=lambda name: (
            results[name]["macro_within_site_auc"]
            if results[name]["macro_within_site_auc"] is not None
            else -1.0
        ),
    )
    return {"baselines": results, "best_baseline": best_name}


def _load_for_n(
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
    cost = config["incremental_bit_cost"]
    incremental_bits = (
        int(cost["compact_frame_bits_by_n"][str(n_channels)])
        * int(
            cost["task_update_open_loop_attempts_by_n"][str(n_channels)]
        )
        + int(cost["ack_bits"])
    )
    train_rows = []
    validation_rows = []
    audit = {}
    metadata_lookup = registry["selected_site_members"]
    for site in sorted(set(train_sites + validation_sites)):
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
            "resolved": bool(resolution["resolved"]),
            "decision": resolution["decision"],
            "scene_count": len(states),
            "calibration_scenes": int(resolution["calibration_scenes"]),
        }
        if resolution["resolved"]:
            calibration = int(resolution["calibration_scenes"])
            codebook = codebook_from_selected_actions(
                states[:calibration], resolution["selected_actions"]
            )
            events = build_task_event_value_labels(
                states[calibration:],
                [site] * (len(states) - calibration),
                codebook,
                horizon=int(
                    config["task"]["evaluation_horizon_scenes"]
                ),
                persistent_run_lengths=config["task"][
                    "persistent_run_lengths"
                ],
                refresh_offset=int(
                    config["task"]["forced_refresh_offset_scenes"]
                ),
                incremental_bits=incremental_bits,
                clean_scene_value_bits=config["incremental_bit_cost"][
                    "clean_scene_value_bits_grid"
                ],
            )
            row["model_sample_count"] = len(events["groups"])
            (
                train_rows
                if site in train_sites
                else validation_rows
            ).append(events)
        else:
            row["model_sample_count"] = 0
        audit[site] = row
    return _concatenate(train_rows), _concatenate(validation_rows), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.3a result")
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
    s7_2 = read_json(paths["s7_2_result"])
    train_sites = list(split["train"])
    validation_sites = list(split["validation"])
    internal_sites = list(split["internal_development_test"])
    if (
        s7_2["decision"]["candidate_freeze_gate_passed"]
        or len(train_sites)
        != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
        or int(access["roles"]["reserve"]["access_count"]) != 0
    ):
        raise ValueError("S7.3a governance or trigger mismatch")
    selected_sites = set(train_sites + validation_sites)
    if selected_sites.intersection(internal_sites):
        raise ValueError("internal test entered S7.3a selection")
    metadata_lookup = registry["selected_site_members"]
    metadata_rows = [metadata_lookup[site] for site in sorted(selected_sites)]
    loaded = load_npy_members(
        Path(protocol["inputs"]["archive"]),
        [row["member"] for row in metadata_rows],
    )
    started = time.perf_counter()
    n_results = {}
    passing = 0
    concentration_failures = 0
    for n_channels in map(int, config["task"]["n_channels"]):
        train, validation, site_audit = _load_for_n(
            n_channels=n_channels,
            config=config,
            protocol=protocol,
            registry=registry,
            freeze=freeze,
            loaded=loaded,
            train_sites=train_sites,
            validation_sites=validation_sites,
        )
        any_distribution = _distribution(
            validation["any_failure"], validation["groups"]
        )
        train_any_distribution = _distribution(
            train["any_failure"], train["groups"]
        )
        transient_distribution = _distribution(
            validation["transient_failure"], validation["groups"]
        )
        persistent = {}
        train_persistent = {}
        for run_length in config["task"]["persistent_run_lengths"]:
            key = str(run_length)
            persistent[key] = _distribution(
                validation["persistent_failure"][key],
                validation["groups"],
            )
            train_persistent[key] = _distribution(
                train["persistent_failure"][key],
                train["groups"],
            )
        beneficial_distribution = _distribution(
            validation["forced_refresh_beneficial"],
            validation["groups"],
        )
        train_beneficial_distribution = _distribution(
            train["forced_refresh_beneficial"], train["groups"]
        )
        train_for_model = dict(train)
        validation_for_model = dict(validation)
        primary_run = str(
            config["task"]["primary_persistent_run_length"]
        )
        train_for_model["persistent_primary"] = train[
            "persistent_failure"
        ][primary_run]
        validation_for_model["persistent_primary"] = validation[
            "persistent_failure"
        ][primary_run]
        persistent_prediction = _predictability(
            train_for_model,
            validation_for_model,
            label_key="persistent_primary",
            baselines=config["strong_baselines"],
            prior_strength=float(config["lookup"]["prior_strength"]),
        )
        beneficial_prediction = _predictability(
            train,
            validation,
            label_key="forced_refresh_beneficial",
            baselines=config["strong_baselines"],
            prior_strength=float(config["lookup"]["prior_strength"]),
        )
        persistent_training_oof = _training_oof_predictability(
            train_for_model,
            label_key="persistent_primary",
            baselines=config["strong_baselines"],
            prior_strength=float(config["lookup"]["prior_strength"]),
        )
        beneficial_training_oof = _training_oof_predictability(
            train,
            label_key="forced_refresh_beneficial",
            baselines=config["strong_baselines"],
            prior_strength=float(config["lookup"]["prior_strength"]),
        )
        beneficial_mask = validation["forced_refresh_beneficial"] == 1
        avoided = validation["avoided_dirty_scenes"][beneficial_mask]
        avoided_regret = validation["avoided_excess_regret_db"][
            beneficial_mask
        ]
        break_even = validation[
            "break_even_bits_per_avoided_dirty_scene"
        ][beneficial_mask]
        value_grid = {
            key: _distribution(values, validation["groups"])
            for key, values in validation[
                "net_positive_by_clean_scene_value_bits"
            ].items()
        }
        gate = config["label_viability_gate"]
        persistent_primary = persistent[primary_run]
        best_beneficial = beneficial_prediction["baselines"][
            beneficial_prediction["best_baseline"]
        ]
        checks = {
            "persistent_positive_sites": bool(
                persistent_primary["positive_site_count"]
                >= int(gate["minimum_positive_validation_site_count"])
            ),
            "persistent_not_overconcentrated": bool(
                persistent_primary["dominant_site_event_share"] is not None
                and persistent_primary["dominant_site_event_share"]
                <= float(gate["maximum_dominant_site_event_share"])
            ),
            "beneficial_positive_sites": bool(
                beneficial_distribution["positive_site_count"]
                >= int(gate["minimum_positive_validation_site_count"])
            ),
            "beneficial_not_overconcentrated": bool(
                beneficial_distribution["dominant_site_event_share"]
                is not None
                and beneficial_distribution["dominant_site_event_share"]
                <= float(gate["maximum_dominant_site_event_share"])
            ),
            "beneficial_predictability": bool(
                best_beneficial["macro_within_site_auc"] is not None
                and best_beneficial["macro_within_site_auc"]
                >= float(gate["minimum_macro_within_site_auc"])
            ),
            "avoided_dirty_scenes": bool(
                avoided.size > 0
                and float(np.median(avoided))
                >= float(
                    gate[
                        "minimum_median_avoided_dirty_scenes_among_beneficial"
                    ]
                )
            ),
        }
        passed = all(checks.values())
        passing += int(passed)
        concentration_failures += int(
            not checks["persistent_not_overconcentrated"]
            or not checks["beneficial_not_overconcentrated"]
        )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "incremental_bits": validation["incremental_bits"],
            "resolved_train_site_count": len(
                set(train["groups"].tolist())
            ),
            "resolved_validation_site_count": len(
                set(validation["groups"].tolist())
            ),
            "site_audit": site_audit,
            "training_event_distributions": {
                "any_failure": train_any_distribution,
                "persistent_failure": train_persistent,
                "forced_refresh_beneficial": (
                    train_beneficial_distribution
                ),
            },
            "event_distributions": {
                "any_failure": any_distribution,
                "transient_failure": transient_distribution,
                "persistent_failure": persistent,
                "forced_refresh_beneficial": beneficial_distribution,
                "net_positive_by_clean_scene_value_bits": value_grid,
                "transient_fraction_among_any_failures": (
                    transient_distribution["positive_count"]
                    / any_distribution["positive_count"]
                    if any_distribution["positive_count"]
                    else None
                ),
                "persistent_fraction_among_any_failures": (
                    persistent_primary["positive_count"]
                    / any_distribution["positive_count"]
                    if any_distribution["positive_count"]
                    else None
                ),
            },
            "forced_refresh_effect": {
                "beneficial_count": int(np.sum(beneficial_mask)),
                "median_avoided_dirty_scenes": (
                    float(np.median(avoided)) if avoided.size else None
                ),
                "mean_avoided_dirty_scenes": (
                    float(np.mean(avoided)) if avoided.size else None
                ),
                "median_avoided_excess_regret_db": (
                    float(np.median(avoided_regret))
                    if avoided_regret.size
                    else None
                ),
                "median_break_even_bits_per_avoided_dirty_scene": (
                    float(np.median(break_even))
                    if break_even.size
                    else None
                ),
                "negative_value_count": int(
                    np.sum(validation["avoided_dirty_scenes"] < 0)
                ),
            },
            "strong_baseline_predictability": {
                "persistent_failure": persistent_prediction,
                "forced_refresh_beneficial": beneficial_prediction,
                "training_leave_one_site_out": {
                    "persistent_failure": persistent_training_oof,
                    "forced_refresh_beneficial": beneficial_training_oof,
                },
            },
            "viability_checks": checks,
            "label_viability_gate_passed": passed,
        }
        print(f"S7.3a N={n_channels} pass={passed}", flush=True)
    required = int(
        config["label_viability_gate"]["minimum_passing_n_count"]
    )
    if passing >= required:
        next_action = config["decision_policy"]["if_gate_passes"]
    elif concentration_failures:
        next_action = config["decision_policy"][
            "if_events_exist_but_remain_site_concentrated"
        ]
    else:
        next_action = config["decision_policy"][
            "if_forced_refresh_has_little_counterfactual_value"
        ]
    result = {
        "version": "1.0",
        "status": "stage7_s7_3a_event_value_labels_complete",
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
            "passing_n_count": passing,
            "required_n_count": required,
            "label_viability_gate_passed": passing >= required,
            "next_action": next_action,
        },
        "governance_checks": {
            "internal_development_test_not_loaded": True,
            "reserve_remained_unread": True,
            "future_used_only_for_labels_and_counterfactuals": True,
            "full_protocol_benefit_claim_forbidden": True,
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
