#!/usr/bin/env python
"""Audit S7.1 aggregate performance for site concentration and instability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import read_json  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_s7_1b_site_robustness_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/s7_1b_site_robustness_v1/result.json"
)


def _interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = 1.0 - float(confidence)
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


def _site_bootstrap(
    rows: list[dict],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(int(seed))
    auc = np.asarray([row["roc_auc"] for row in rows], dtype=np.float64)
    ap_lift = np.asarray(
        [
            row["average_precision"] - row["positive_prevalence"]
            for row in rows
        ],
        dtype=np.float64,
    )
    indices = rng.integers(
        0, len(rows), size=(int(replicates), len(rows))
    )
    auc_samples = np.mean(auc[indices], axis=1)
    ap_lift_samples = np.mean(ap_lift[indices], axis=1)
    return {
        "unit": "evaluable_validation_site",
        "replicates": int(replicates),
        "confidence_level": float(confidence),
        "macro_auc_interval": _interval(auc_samples, confidence),
        "macro_ap_lift_interval": _interval(ap_lift_samples, confidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.1b result")
    config = read_json(args.config)
    source_path = PROJECT_DIR / config["input_result"]
    source = read_json(source_path)
    if source["status"] != "stage7_s7_1_strong_baselines_complete":
        raise ValueError("unexpected S7.1 source status")
    if source["loaded_sites"]["internal_development_test"]:
        raise ValueError("S7.1 source loaded the internal test split")
    if source["loaded_sites"]["reserve"]:
        raise ValueError("S7.1 source loaded reserve values")
    horizon = str(config["primary_horizon_scenes"])
    bootstrap = config["bootstrap"]
    cautions = config["caution_rules"]
    n_results = {}
    any_caution = False
    for n_position, n_property in enumerate(
        source["n_results"].items()
    ):
        n_channels, n_source = n_property
        risk = n_source["task_risk"][horizon]
        best_name = risk["best_baseline"]
        aggregate = risk["baselines"][best_name]
        all_sites = []
        evaluable = []
        for site, metrics in sorted(
            risk["best_baseline_per_validation_site"].items()
        ):
            sample_count = int(metrics["sample_count"])
            positive_count = int(
                round(
                    float(metrics["positive_prevalence"]) * sample_count
                )
            )
            row = {
                "site": site,
                "sample_count": sample_count,
                "positive_count": positive_count,
                "negative_count": sample_count - positive_count,
                "positive_prevalence": float(
                    metrics["positive_prevalence"]
                ),
                "roc_auc": (
                    None
                    if metrics["roc_auc"] is None
                    else float(metrics["roc_auc"])
                ),
                "average_precision": (
                    None
                    if positive_count == 0
                    else float(metrics["average_precision"])
                ),
                "discrimination_evaluable": bool(
                    positive_count > 0 and positive_count < sample_count
                ),
            }
            all_sites.append(row)
            if row["discrimination_evaluable"]:
                evaluable.append(row)
        total_positives = sum(row["positive_count"] for row in all_sites)
        dominant = max(all_sites, key=lambda row: row["positive_count"])
        dominant_share = (
            dominant["positive_count"] / total_positives
            if total_positives
            else 0.0
        )
        macro_auc = float(
            np.mean([row["roc_auc"] for row in evaluable])
        )
        macro_ap_lift = float(
            np.mean(
                [
                    row["average_precision"]
                    - row["positive_prevalence"]
                    for row in evaluable
                ]
            )
        )
        intervals = _site_bootstrap(
            evaluable,
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]) + n_position * 1000,
        )
        flags = {
            "positive_events_overconcentrated": bool(
                dominant_share
                > float(cautions["dominant_site_positive_event_share"])
            ),
            "too_few_evaluable_sites": bool(
                len(evaluable)
                < int(cautions["minimum_evaluable_site_count"])
            ),
            "dominant_site_auc_below_minimum": bool(
                dominant["roc_auc"] is None
                or dominant["roc_auc"]
                < float(cautions["minimum_dominant_site_auc"])
            ),
        }
        caution = any(flags.values())
        any_caution = any_caution or caution
        n_results[n_channels] = {
            "n_channels": int(n_channels),
            "primary_horizon_scenes": int(horizon),
            "selected_baseline": best_name,
            "aggregate_auc": aggregate["roc_auc"],
            "aggregate_average_precision": aggregate[
                "average_precision"
            ],
            "site_count": len(all_sites),
            "evaluable_site_count": len(evaluable),
            "total_positive_count": total_positives,
            "dominant_positive_site": dominant["site"],
            "dominant_positive_count": dominant["positive_count"],
            "dominant_positive_event_share": float(dominant_share),
            "dominant_site_auc": dominant["roc_auc"],
            "macro_within_site_auc": macro_auc,
            "macro_within_site_ap_lift_over_site_prevalence": (
                macro_ap_lift
            ),
            "aggregate_minus_macro_auc": float(
                aggregate["roc_auc"] - macro_auc
            ),
            "bootstrap": intervals,
            "site_metrics": all_sites,
            "caution_flags": flags,
            "robustness_caution": caution,
        }
    next_action = (
        config["decision_policy"]["if_any_caution_rule_triggers"]
        if any_caution
        else config["decision_policy"]["otherwise"]
    )
    result = {
        "version": "1.0",
        "status": "stage7_s7_1b_site_robustness_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "source_result_sha256": sha256_file(source_path),
        "analysis_timing": "posthoc_after_registered_s7_1_result",
        "n_results": n_results,
        "decision": {
            "any_robustness_caution": any_caution,
            "next_action": next_action,
            "registered_s7_1_aggregate_gate_result_unchanged": True,
            "internal_development_test_may_be_opened": not any_caution,
        },
        "governance_checks": {
            "internal_development_test_loaded": False,
            "reserve_loaded": False,
            "external_final_claim_forbidden": True,
        },
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
