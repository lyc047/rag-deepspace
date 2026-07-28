"""Interpretable task-codeword transition probabilities for causal scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MarkovSchedulerModel:
    class_count: int
    duration_bin_lower_bounds: tuple[int, ...]
    first_order_probabilities: np.ndarray
    duration_probabilities: np.ndarray
    entropy_weight: float


def codeword_run_lengths(
    labels: np.ndarray, cluster_ids: np.ndarray
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(cluster_ids).astype(str)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("unaligned codeword run-length inputs")
    lengths = np.ones(values.size, dtype=np.int64)
    for index in range(1, values.size):
        if (
            groups[index] == groups[index - 1]
            and values[index] == values[index - 1]
        ):
            lengths[index] = lengths[index - 1] + 1
    return lengths


def _duration_bin(
    run_length: int, lower_bounds: tuple[int, ...]
) -> int:
    if run_length < 1:
        raise ValueError("run length must be positive")
    return int(
        np.searchsorted(
            np.asarray(lower_bounds, dtype=np.int64),
            int(run_length),
            side="right",
        )
        - 1
    )


def fit_markov_scheduler(
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    training_groups: Iterable[str],
    class_count: int,
    duration_bin_lower_bounds: Iterable[int],
    laplace_alpha: float,
    duration_shrinkage_strength: float,
    entropy_weight: float,
) -> MarkovSchedulerModel:
    values = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(cluster_ids).astype(str)
    selected_groups = set(str(value) for value in training_groups)
    bounds = tuple(int(value) for value in duration_bin_lower_bounds)
    if (
        values.ndim != 1
        or groups.shape != values.shape
        or class_count < 2
        or not bounds
        or bounds[0] != 1
        or tuple(sorted(set(bounds))) != bounds
        or laplace_alpha <= 0
        or duration_shrinkage_strength < 0
        or entropy_weight < 0
        or np.any(values < 0)
        or np.any(values >= class_count)
    ):
        raise ValueError("invalid Markov scheduler inputs")
    run_lengths = codeword_run_lengths(values, groups)
    first_counts = np.full(
        (class_count, class_count),
        float(laplace_alpha),
        dtype=np.float64,
    )
    duration_counts = np.zeros(
        (class_count, len(bounds), class_count), dtype=np.float64
    )
    for index in range(values.size - 1):
        if (
            groups[index] != groups[index + 1]
            or groups[index] not in selected_groups
        ):
            continue
        source = int(values[index])
        destination = int(values[index + 1])
        duration = _duration_bin(int(run_lengths[index]), bounds)
        first_counts[source, destination] += 1.0
        duration_counts[source, duration, destination] += 1.0
    first_probabilities = first_counts / np.sum(
        first_counts, axis=1, keepdims=True
    )
    duration_probabilities = np.empty_like(duration_counts)
    for source in range(class_count):
        prior = (
            float(duration_shrinkage_strength)
            * first_probabilities[source]
        )
        for duration in range(len(bounds)):
            posterior = duration_counts[source, duration] + prior
            if np.sum(posterior) <= 0:
                posterior = first_probabilities[source]
            duration_probabilities[source, duration] = (
                posterior / np.sum(posterior)
            )
    return MarkovSchedulerModel(
        class_count=class_count,
        duration_bin_lower_bounds=bounds,
        first_order_probabilities=first_probabilities,
        duration_probabilities=duration_probabilities,
        entropy_weight=float(entropy_weight),
    )


def score_markov_scheduler(
    model: MarkovSchedulerModel,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score each state using its current codeword and past run length only."""

    values = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(cluster_ids).astype(str)
    if (
        values.ndim != 1
        or groups.shape != values.shape
        or np.any(values < 0)
        or np.any(values >= model.class_count)
    ):
        raise ValueError("invalid Markov scoring inputs")
    run_lengths = codeword_run_lengths(values, groups)
    first_change = np.empty(values.size, dtype=np.float64)
    duration_change = np.empty(values.size, dtype=np.float64)
    duration_entropy = np.empty(values.size, dtype=np.float64)
    entropy_scale = np.log(float(model.class_count))
    for index, source_value in enumerate(values):
        source = int(source_value)
        duration = _duration_bin(
            int(run_lengths[index]), model.duration_bin_lower_bounds
        )
        first_distribution = model.first_order_probabilities[source]
        duration_distribution = model.duration_probabilities[
            source, duration
        ]
        first_change[index] = 1.0 - first_distribution[source]
        duration_change[index] = 1.0 - duration_distribution[source]
        positive = duration_distribution[duration_distribution > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        duration_entropy[index] = entropy / entropy_scale
    return {
        "first_order_change_probability": first_change,
        "duration_change_probability": duration_change,
        "duration_normalized_entropy": duration_entropy,
        "duration_entropy_priority": duration_change
        * (1.0 + model.entropy_weight * duration_entropy),
        "run_length_scenes": run_lengths.astype(np.float64),
    }


def online_token_bucket_candidates(
    scores: np.ndarray,
    evaluation_indices: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    threshold: float,
    state_count: int,
    token_accrual_per_scene: float,
    bucket_capacity: float,
    initial_tokens: float,
    reset_at_group_boundary: bool,
) -> np.ndarray:
    """Causally thin threshold crossings using only past token state."""

    values = np.asarray(scores, dtype=np.float64)
    indices = np.asarray(evaluation_indices, dtype=np.int64)
    groups = np.asarray(cluster_ids).astype(str)
    if (
        values.shape != (state_count,)
        or groups.shape != (state_count,)
        or indices.ndim != 1
        or np.any(indices < 0)
        or np.any(indices >= state_count)
        or token_accrual_per_scene <= 0
        or bucket_capacity < 1.0
        or initial_tokens < 0
        or initial_tokens > bucket_capacity
    ):
        raise ValueError("invalid token-bucket scheduler inputs")
    candidates = np.zeros(state_count, dtype=bool)
    tokens = float(initial_tokens)
    previous_group: str | None = None
    for index_value in indices:
        index = int(index_value)
        group = str(groups[index])
        if group != previous_group:
            if previous_group is not None and reset_at_group_boundary:
                tokens = float(initial_tokens)
            previous_group = group
        else:
            tokens = min(
                float(bucket_capacity),
                tokens + float(token_accrual_per_scene),
            )
        if values[index] >= float(threshold) and tokens >= 1.0:
            candidates[index] = True
            tokens -= 1.0
    return candidates
