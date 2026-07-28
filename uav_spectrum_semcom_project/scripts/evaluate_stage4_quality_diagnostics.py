#!/usr/bin/env python
"""Evaluate quality calibration and controlled front-end anomaly sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectrum_semcom.multi_uav_fusion import fit_monotonic_confidence_calibrator
from spectrum_semcom.quality_diagnostics import (
    audited_quality_scores,
    binary_auc,
    brier_score,
    controlled_faults,
    expected_calibration_error,
)


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        occupancy = data["node_occupancy"].astype(np.float64)
        truth = data["truth_occupancy"].astype(np.float64)
        quality = data["node_quality"].astype(np.float64)
        names = data["quality_names"].tolist()
    target = 1.0 - np.mean(np.abs(occupancy - truth[:, None, :]), axis=2)
    return quality.reshape(-1, quality.shape[-1]), target.reshape(-1), occupancy, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/stage4/quality_diagnostics_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    cal_q, cal_target, _, names = load_rows(root / "results/stage4/multinode_cache_v1/multinode_calibration.npz")
    val_q, val_target, _, val_names = load_rows(root / "results/stage4/multinode_cache_v1/multinode_validation.npz")
    if names != val_names or names[1] != "prediction_confidence":
        raise ValueError("unexpected quality schema")
    calibrator = fit_monotonic_confidence_calibrator(cal_q[:, 1], cal_target, n_bins=10)
    raw = val_q[:, 1]
    calibrated = calibrator.predict(raw)
    raw_ece, raw_table = expected_calibration_error(raw, val_target)
    calibrated_ece, calibrated_table = expected_calibration_error(calibrated, val_target)

    healthy = audited_quality_scores(val_q)
    fault_metrics = {}
    for name, rows in controlled_faults(val_q).items():
        faulty = audited_quality_scores(rows)
        fault_metrics[name] = {
            "pairwise_detection_rate": float(np.mean(faulty < healthy)),
            "quality_score_auc": binary_auc(healthy, faulty),
            "healthy_mean_score": float(np.mean(healthy)),
            "fault_mean_score": float(np.mean(faulty)),
        }

    result = {
        "experiment_id": "stage4_quality_diagnostics_v1",
        "fit_split": "calibration",
        "evaluation_split": "validation",
        "calibration_target": "one_minus_node_occupancy_MAE_against_truth",
        "calibration_scene_count": 50,
        "validation_scene_count": 150,
        "validation_node_rows": int(val_q.shape[0]),
        "quality_names": names,
        "calibrator": {"score_knots": calibrator.score_knots.tolist(), "quality_knots": calibrator.quality_knots.tolist()},
        "calibration_metrics": {
            "raw_brier": brier_score(raw, val_target),
            "calibrated_brier": brier_score(calibrated, val_target),
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "raw_reliability_table": raw_table,
            "calibrated_reliability_table": calibrated_table,
        },
        "controlled_anomaly_metrics": fault_metrics,
        "acceptance": {"completed": True, "performance_gate": "diagnostic_only_no_posthoc_threshold_tuning"},
        "claim_boundary": "Calibration uses calibration truth and reports validation aggregates only. Injected faults validate score sensitivity, not real-world anomaly prevalence or H3.",
        "final_holdout_accessed": False,
    }
    (output / "quality_diagnostics_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage-4 quality diagnostics", "", f"Validation rows: {val_q.shape[0]}; final holdout accessed: false.", "", "## Calibration", "", "| Metric | Raw | Calibrated |", "|---|---:|---:|", f"| Brier | {result['calibration_metrics']['raw_brier']:.6f} | {result['calibration_metrics']['calibrated_brier']:.6f} |", f"| ECE | {raw_ece:.6f} | {calibrated_ece:.6f} |", "", "## Controlled anomaly sensitivity", "", "| Fault | Pairwise detection | Score AUC |", "|---|---:|---:|"]
    lines += [f"| {name} | {m['pairwise_detection_rate']:.4f} | {m['quality_score_auc']:.4f} |" for name, m in fault_metrics.items()]
    lines += ["", f"> {result['claim_boundary']}", ""]
    (output / "quality_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["calibration_metrics"] | {"faults": len(fault_metrics)}, indent=2))


if __name__ == "__main__":
    main()
