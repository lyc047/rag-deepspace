"""Calibration and controlled-fault diagnostics for stage-4 quality vectors."""

from __future__ import annotations

import numpy as np


def brier_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size != target.size or prediction.size == 0:
        raise ValueError("prediction and target must be non-empty and equally sized")
    return float(np.mean((prediction - target) ** 2))


def expected_calibration_error(prediction: np.ndarray, target: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float | int]]]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.size != target.size or prediction.size == 0 or n_bins < 2:
        raise ValueError("invalid calibration inputs")
    order = np.argsort(prediction)
    chunks = [x for x in np.array_split(order, min(n_bins, prediction.size)) if x.size]
    table: list[dict[str, float | int]] = []
    ece = 0.0
    for indices in chunks:
        mean_prediction = float(np.mean(prediction[indices]))
        mean_target = float(np.mean(target[indices]))
        weight = indices.size / prediction.size
        ece += weight * abs(mean_prediction - mean_target)
        table.append({"count": int(indices.size), "mean_prediction": mean_prediction, "mean_target": mean_target})
    return float(ece), table


def audited_quality_scores(quality: np.ndarray, freshness_tau_s: float = 0.5, noise_scale_db: float = 4.0) -> np.ndarray:
    """Vectorized form of the deployable audited quality product."""

    q = np.asarray(quality, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 10 or not np.all(np.isfinite(q)):
        raise ValueError("quality must have shape [samples, 10] and be finite")
    sensing = 1.0 / (1.0 + np.exp(-q[:, 0] / 4.0))
    frontend = (1.0 - q[:, 3]) * (1.0 - q[:, 4]) * np.exp(-q[:, 5] / noise_scale_db)
    freshness = np.exp(-q[:, 7] / freshness_tau_s)
    return np.clip(sensing * q[:, 1] * q[:, 8] * freshness * frontend * (1.0 - q[:, 9]), 0.0, 1.0)


def binary_auc(healthy_scores: np.ndarray, faulty_scores: np.ndarray) -> float:
    """Pairwise AUC where a lower quality score denotes a fault."""

    healthy = np.asarray(healthy_scores, dtype=np.float64).reshape(-1)
    faulty = np.asarray(faulty_scores, dtype=np.float64).reshape(-1)
    if healthy.size == 0 or faulty.size == 0:
        raise ValueError("both score sets must be non-empty")
    comparisons = healthy[:, None] - faulty[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def controlled_faults(quality: np.ndarray) -> dict[str, np.ndarray]:
    base = np.asarray(quality, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 10:
        raise ValueError("quality must have shape [samples, 10]")
    faults: dict[str, np.ndarray] = {}
    specifications = {
        "clipping": (3, 0.50, "max"),
        "out_of_band_leakage": (4, 0.50, "max"),
        "noise_instability": (5, 6.0, "add"),
        "stale_report": (7, 2.0, "add"),
        "report_failure_risk": (8, 0.25, "min"),
        "calibration_failure": (9, 0.50, "max"),
    }
    for name, (index, value, operation) in specifications.items():
        fault = base.copy()
        if operation == "max":
            fault[:, index] = np.maximum(fault[:, index], value)
        elif operation == "min":
            fault[:, index] = np.minimum(fault[:, index], value)
        else:
            fault[:, index] += value
        faults[name] = fault
    contradictory = base.copy()
    contradictory[:, 1] = 0.99
    contradictory[:, 9] = np.maximum(contradictory[:, 9], 0.75)
    faults["high_confidence_miscalibrated"] = contradictory
    return faults
