"""Quality-aware fusion primitives for correlated multi-UAV observations.

The module is deliberately independent of a detector.  Each UAV contributes a
channel-occupancy vector plus auditable quality metadata.  This keeps classical
fusion baselines and the later learned fusion model on exactly the same input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NodeOccupancyReport:
    occupancy: np.ndarray
    sensing_snr_db: float
    confidence: float
    report_success_probability: float
    age_s: float = 0.0


@dataclass(frozen=True)
class MonotonicConfidenceCalibrator:
    """Piecewise-linear validation-only mapping from raw score to report quality."""

    score_knots: np.ndarray
    quality_knots: np.ndarray

    def predict(self, score: float | np.ndarray) -> np.ndarray:
        values = np.asarray(score, dtype=np.float64)
        return np.clip(np.interp(values, self.score_knots, self.quality_knots), 0.0, 1.0)


def fit_monotonic_confidence_calibrator(
    scores: np.ndarray,
    qualities: np.ndarray,
    n_bins: int = 10,
) -> MonotonicConfidenceCalibrator:
    """Fit an equal-frequency monotonic calibration table using PAV merging."""

    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(qualities, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("scores and qualities must be equally sized finite arrays with at least two values")
    order = np.argsort(x)
    chunks = [chunk for chunk in np.array_split(order, min(int(n_bins), x.size)) if chunk.size]
    blocks = [[float(np.mean(x[c])), float(np.mean(y[c])), float(c.size)] for c in chunks]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][1] <= blocks[index + 1][1]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[2] + right[2]
        merged = [
            (left[0] * left[2] + right[0] * right[2]) / weight,
            (left[1] * left[2] + right[1] * right[2]) / weight,
            weight,
        ]
        blocks[index : index + 2] = [merged]
        index = max(0, index - 1)
    score_knots = np.asarray([block[0] for block in blocks], dtype=np.float64)
    quality_knots = np.clip(np.asarray([block[1] for block in blocks], dtype=np.float64), 0.0, 1.0)
    if score_knots.size == 1:
        score_knots = np.asarray([0.0, 1.0], dtype=np.float64)
        quality_knots = np.repeat(quality_knots, 2)
    return MonotonicConfidenceCalibrator(score_knots, quality_knots)


def maximum_pair_disagreement(reports: list[NodeOccupancyReport]) -> float:
    """Maximum mean absolute occupancy disagreement between any two reports."""

    reports = _validated_reports(reports)
    if len(reports) < 2:
        return 0.0
    values = [np.asarray(report.occupancy, dtype=np.float64) for report in reports]
    return float(
        max(
            np.mean(np.abs(values[left] - values[right]))
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
    )


def _validated_reports(reports: list[NodeOccupancyReport]) -> list[NodeOccupancyReport]:
    if not reports:
        raise ValueError("at least one received report is required")
    width = int(np.asarray(reports[0].occupancy).size)
    if width <= 0:
        raise ValueError("occupancy vectors must be non-empty")
    for report in reports:
        values = np.asarray(report.occupancy, dtype=np.float32)
        if values.ndim != 1 or values.size != width:
            raise ValueError("all occupancy vectors must be one-dimensional and equally sized")
        if not np.all(np.isfinite(values)):
            raise ValueError("occupancy values must be finite")
    return reports


def quality_weight(report: NodeOccupancyReport, freshness_tau_s: float = 0.5) -> float:
    """Return an interpretable product weight in [0, 1]."""

    if freshness_tau_s <= 0:
        raise ValueError("freshness_tau_s must be positive")
    snr_score = 1.0 / (1.0 + np.exp(-float(report.sensing_snr_db) / 4.0))
    confidence = float(np.clip(report.confidence, 0.0, 1.0))
    reliability = float(np.clip(report.report_success_probability, 0.0, 1.0))
    freshness = float(np.exp(-max(0.0, float(report.age_s)) / freshness_tau_s))
    return float(snr_score * confidence * reliability * freshness)


def fuse_occupancy(
    reports: list[NodeOccupancyReport],
    method: str,
    occupancy_threshold: float = 0.02,
    freshness_tau_s: float = 0.5,
    disagreement_scale: float = 0.05,
    max_weight_ratio: float = 2.0,
) -> np.ndarray:
    """Fuse received node reports using a declared classical baseline.

    Supported methods are ``or``, ``majority``, ``mean``, ``median``, ``snr_weighted``,
    ``quality_weighted`` and ``robust_quality_weighted``.  The robust rule adds
    cross-node agreement gating and conservative weight clipping.
    """

    reports = _validated_reports(reports)
    values = np.stack([np.clip(report.occupancy, 0.0, 1.0) for report in reports])
    if method == "or":
        return np.max(values, axis=0).astype(np.float32)
    if method == "majority":
        votes = values > float(occupancy_threshold)
        return (np.mean(votes, axis=0) >= 0.5).astype(np.float32)
    if method == "mean":
        return np.mean(values, axis=0).astype(np.float32)
    if method == "median":
        return np.median(values, axis=0).astype(np.float32)
    if method == "snr_weighted":
        weights = np.asarray(
            [1.0 / (1.0 + np.exp(-float(report.sensing_snr_db) / 4.0)) for report in reports],
            dtype=np.float64,
        )
    elif method in {"quality_weighted", "robust_quality_weighted"}:
        weights = np.asarray([quality_weight(report, freshness_tau_s) for report in reports], dtype=np.float64)
        if method == "robust_quality_weighted" and len(reports) >= 3:
            if disagreement_scale <= 0 or max_weight_ratio < 1:
                raise ValueError("robust fusion requires positive disagreement_scale and max_weight_ratio >= 1")
            consensus = np.median(values, axis=0)
            disagreement = np.mean(np.abs(values - consensus[None, :]), axis=1)
            weights *= np.exp(-disagreement / float(disagreement_scale))
            positive = weights[weights > 1e-12]
            if positive.size:
                weights = np.minimum(weights, float(max_weight_ratio) * float(np.median(positive)))
    else:
        raise ValueError(f"unknown fusion method: {method}")
    if float(weights.sum()) <= 1e-12:
        return np.mean(values, axis=0).astype(np.float32)
    return np.average(values, axis=0, weights=weights).astype(np.float32)


def perturb_correlated_spectrogram(
    image: np.ndarray,
    sensing_snr_db: float,
    rng: np.random.Generator,
    gain_db: float = 0.0,
    frequency_shift_bins: int = 0,
) -> np.ndarray:
    """Create a controlled node view from the same underlying spectrogram.

    This is a semi-synthetic diagnostic, not a substitute for synchronized SDR
    captures.  It preserves scene identity while varying node observation
    quality, gain and a small frequency offset.
    """

    source = np.asarray(image, dtype=np.float32) / 255.0
    shifted = np.roll(source, int(frequency_shift_bins), axis=0)
    scaled = shifted * (10.0 ** (float(gain_db) / 20.0))
    signal_power = max(float(np.mean(np.square(scaled))), 1e-8)
    noise_power = signal_power / (10.0 ** (float(sensing_snr_db) / 10.0))
    noisy = scaled + rng.normal(0.0, np.sqrt(noise_power), size=scaled.shape)
    return np.rint(np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
