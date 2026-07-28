"""Paired hierarchical bootstrap for Gate A development comparisons."""

from __future__ import annotations

import numpy as np


def hierarchical_paired_bootstrap(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    statistic: str = "mean",
    cvar_alpha: float = 0.9,
    repetitions: int = 5000,
    seed: int = 20260716,
) -> dict[str, float]:
    left = np.asarray(proposed, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or min(left.shape) < 1 or not np.all(np.isfinite(left-right)):
        raise ValueError("paired arrays must be equally shaped finite [seeds, scenes] matrices")
    if statistic not in {"mean", "cvar"} or repetitions < 100 or not 0 <= cvar_alpha < 1:
        raise ValueError("invalid statistic, repetitions, or CVaR alpha")

    def value(array: np.ndarray) -> float:
        flat = array.reshape(-1)
        if statistic == "mean":
            return float(np.mean(flat))
        count = max(1, int(np.ceil((1-cvar_alpha) * flat.size)))
        return float(np.mean(np.partition(flat, flat.size-count)[-count:]))

    observed = value(left) - value(right)
    rng = np.random.default_rng(seed); n_seeds, n_scenes = left.shape; samples = np.empty(repetitions)
    for repeat in range(repetitions):
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        scene_indices = rng.integers(0, n_scenes, size=(n_seeds, n_scenes))
        sampled_left = np.stack([left[source, scene_indices[row]] for row, source in enumerate(seed_indices)])
        sampled_right = np.stack([right[source, scene_indices[row]] for row, source in enumerate(seed_indices)])
        samples[repeat] = value(sampled_left) - value(sampled_right)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"difference": observed, "ci95_low": float(low), "ci95_high": float(high), "bootstrap_repetitions": repetitions}
