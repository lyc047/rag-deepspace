"""Lightweight context learning for safe Stage-6R codebook adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def robust_spectral_shape_features(power_dbm: np.ndarray) -> np.ndarray:
    """Return one scale-invariant mean spectral-shape feature vector."""
    values = np.asarray(power_dbm, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("power_dbm must be a non-empty scenes-by-channels array")
    if not np.all(np.isfinite(values)):
        raise ValueError("power_dbm must be finite")
    centered = values - np.median(values, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(scale, 1e-6)
    return np.mean(normalized, axis=0)


def normalized_feature_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        first.size < 1
        or first.shape != second.shape
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("features must be finite vectors with equal shape")
    return float(np.linalg.norm(first - second) / np.sqrt(first.size))


def nearest_context(
    feature: np.ndarray, prototypes: Mapping[str, np.ndarray]
) -> tuple[str, float]:
    if not prototypes:
        raise ValueError("at least one context prototype is required")
    rows = [
        (normalized_feature_distance(feature, value), str(name))
        for name, value in prototypes.items()
    ]
    distance, name = min(rows)
    return name, float(distance)


def leave_one_context_novelty_threshold(
    prototypes: Mapping[str, np.ndarray],
) -> float:
    """Maximum training-only nearest-neighbour context distance."""
    names: Sequence[str] = sorted(map(str, prototypes))
    if len(names) < 2:
        raise ValueError("at least two training contexts are required")
    nearest_distances = []
    for name in names:
        distances = [
            normalized_feature_distance(
                prototypes[name], prototypes[other]
            )
            for other in names
            if other != name
        ]
        nearest_distances.append(min(distances))
    return float(max(nearest_distances))

