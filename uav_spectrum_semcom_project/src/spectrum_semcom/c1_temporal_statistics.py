"""Paired cluster-bootstrap inference for the C1 temporal confirmation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def empirical_cvar_numpy(values: np.ndarray, alpha: float = 0.9) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 1 or not np.all(np.isfinite(data)) or not 0.0 <= alpha < 1.0:
        raise ValueError("values must be non-empty and finite, with alpha in [0, 1)")
    count = max(1, int(np.ceil((1.0 - alpha) * data.size)))
    return float(np.mean(np.partition(data, data.size - count)[-count:]))


def paired_cluster_bootstrap(
    proposed_regret: np.ndarray,
    baseline_regret: np.ndarray,
    proposed_bits: np.ndarray,
    baseline_bits: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    repetitions: int,
    seed: int,
    cvar_alpha: float = 0.9,
    one_sided_confidence: float = 0.95,
) -> dict:
    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for values in (proposed_regret, baseline_regret, proposed_bits, baseline_bits)]
    count = len(arrays[0])
    if count < 2 or any(len(values) != count or not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("all paired metric arrays must have equal non-trivial finite length")
    clusters = np.asarray([str(value) for value in cluster_ids])
    unique = np.unique(clusters)
    if len(clusters) != count or len(unique) < 2:
        raise ValueError("cluster_ids must provide at least two non-empty paired clusters")
    if repetitions < 100 or not 0.5 < one_sided_confidence < 1.0:
        raise ValueError("insufficient repetitions or invalid confidence")
    members = [np.flatnonzero(clusters == cluster) for cluster in unique]
    rng = np.random.default_rng(seed)
    mean_samples = np.empty(repetitions, dtype=np.float64)
    cvar_samples = np.empty(repetitions, dtype=np.float64)
    bit_samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        drawn = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[int(value)] for value in drawn])
        mean_samples[repetition] = float(np.mean(arrays[0][index] - arrays[1][index]))
        cvar_samples[repetition] = empirical_cvar_numpy(arrays[0][index], cvar_alpha) - empirical_cvar_numpy(arrays[1][index], cvar_alpha)
        bit_samples[repetition] = float(np.mean(arrays[2][index] - arrays[3][index]))
    mean_point = float(np.mean(arrays[0] - arrays[1]))
    cvar_point = empirical_cvar_numpy(arrays[0], cvar_alpha) - empirical_cvar_numpy(arrays[1], cvar_alpha)
    bit_point = float(np.mean(arrays[2] - arrays[3]))
    return {
        "cluster_count": int(len(unique)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "one_sided_confidence": float(one_sided_confidence),
        "mean_regret_difference": mean_point,
        "mean_regret_upper_bound": float(np.quantile(mean_samples, one_sided_confidence)),
        "mean_regret_one_sided_p": float((1 + np.count_nonzero(mean_samples >= 0.0)) / (repetitions + 1)),
        "cvar_difference": cvar_point,
        "cvar_upper_bound": float(np.quantile(cvar_samples, one_sided_confidence)),
        "cvar_one_sided_p": float((1 + np.count_nonzero(cvar_samples >= 0.0)) / (repetitions + 1)),
        "actual_bit_difference": bit_point,
        "actual_bit_upper_bound": float(np.quantile(bit_samples, one_sided_confidence)),
        "baseline_mean_actual_bits": float(np.mean(arrays[3])),
    }
