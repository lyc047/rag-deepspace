"""Small dependency-free ranking metrics used by value diagnostics."""

from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be non-empty and finite")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = average_ranks(left)
    y = average_ranks(right)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    x_energy = float(np.sum(x * x))
    y_energy = float(np.sum(y * y))
    if x_energy <= 1e-30 or y_energy <= 1e-30:
        return None
    # Explicit Pearson correlation avoids dispatching to a second BLAS/OpenMP
    # runtime in the packaged Windows NumPy + PyTorch test environment.
    return float(np.sum(x * y) / np.sqrt(x_energy * y_energy))


def top_k_recall(prediction: np.ndarray, target: np.ndarray, k: int) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape != target.shape or not 1 <= k <= prediction.shape[1]:
        raise ValueError("prediction and target must be aligned 2-D arrays and k must be valid")
    oracle = np.argmax(target, axis=1)
    top = np.argpartition(prediction, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == oracle[:, None], axis=1)))
