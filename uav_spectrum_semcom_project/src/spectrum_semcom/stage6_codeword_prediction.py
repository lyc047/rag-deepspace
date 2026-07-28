"""Causal datasets and baselines for forecasting the next task codeword."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskState,
    encode_task_state,
)


def _elapsed_seconds(start: str, end: str) -> float:
    value = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds()
    if value <= 0:
        raise ValueError("timestamps must increase within a trajectory")
    return float(value)


def _feature_names(
    n_channels: int,
    class_count: int,
    history_length: int,
) -> tuple[str, ...]:
    names = [
        f"current_codeword_class_{class_id}"
        for class_id in range(class_count)
    ]
    for lag in range(history_length):
        names.extend(
            f"lag_{lag}_channel_{channel}_power_dbm"
            for channel in range(n_channels)
        )
    for lag in range(history_length - 1):
        names.extend(
            f"delta_lag_{lag}_minus_{lag + 1}_channel_{channel}_db"
            for channel in range(n_channels)
        )
        names.append(f"elapsed_lag_{lag}_to_{lag + 1}_seconds")
    return tuple(names)


def build_next_codeword_datasets(
    channel_power_dbm: np.ndarray,
    states: list[SpectrumTaskState],
    timestamps: np.ndarray,
    cluster_ids: np.ndarray,
    codebook: GreedyTaskCodebook,
    *,
    history_lengths: Iterable[int],
) -> dict:
    """Build aligned one-step datasets using information available through t.

    Every requested history length uses the samples that are eligible for the
    largest window. This prevents a shorter window from gaining an unfair
    advantage merely because it is evaluated on more transition pairs.
    """

    powers = np.asarray(channel_power_dbm, dtype=np.float64)
    times = np.asarray(timestamps).astype(str)
    groups = np.asarray(cluster_ids).astype(str)
    windows = tuple(sorted(set(int(value) for value in history_lengths)))
    if (
        not windows
        or windows[0] < 1
        or powers.ndim != 2
        or powers.shape[0] != len(states)
        or times.shape != (len(states),)
        or groups.shape != (len(states),)
        or powers.shape[1] != codebook.n_channels
        or not np.all(np.isfinite(powers))
    ):
        raise ValueError("invalid next-codeword dataset inputs")
    decisions = [encode_task_state(codebook, state) for state in states]
    fallback_class = len(codebook.codewords)
    labels_all = np.asarray(
        [
            fallback_class if decision.uses_fallback else decision.symbol_id
            for decision in decisions
        ],
        dtype=np.int64,
    )
    maximum_history = max(windows)
    source_indices = []
    for index in range(maximum_history - 1, len(states) - 1):
        span = groups[index - maximum_history + 1 : index + 2]
        if np.all(span == groups[index]):
            source_indices.append(index)
    sources = np.asarray(source_indices, dtype=np.int64)
    if sources.size < 1:
        raise ValueError("no within-group transitions support the largest window")

    class_count = fallback_class + 1
    features_by_window: dict[str, np.ndarray] = {}
    feature_names_by_window: dict[str, tuple[str, ...]] = {}
    for history_length in windows:
        rows = []
        for index in sources:
            current_class = int(labels_all[index])
            row = np.zeros(class_count, dtype=np.float64).tolist()
            row[current_class] = 1.0
            for lag in range(history_length):
                row.extend(powers[index - lag].tolist())
            for lag in range(history_length - 1):
                newer = int(index - lag)
                older = newer - 1
                row.extend((powers[newer] - powers[older]).tolist())
                row.append(_elapsed_seconds(times[older], times[newer]))
            rows.append(row)
        matrix = np.asarray(rows, dtype=np.float64)
        names = _feature_names(
            powers.shape[1], class_count, history_length
        )
        if matrix.shape != (sources.size, len(names)):
            raise AssertionError("next-codeword feature schema mismatch")
        features_by_window[str(history_length)] = matrix
        feature_names_by_window[str(history_length)] = names

    return {
        "features_by_window": features_by_window,
        "feature_names_by_window": feature_names_by_window,
        "labels": labels_all[sources + 1],
        "current_labels": labels_all[sources],
        "groups": groups[sources],
        "source_indices": sources,
        "fallback_class": fallback_class,
        "class_count": class_count,
    }


def fit_frequency_prior(labels: np.ndarray, *, class_count: int) -> int:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=int(class_count))
    return int(np.flatnonzero(counts == np.max(counts))[0])


def fit_first_order_markov(
    current_labels: np.ndarray,
    next_labels: np.ndarray,
    *,
    class_count: int,
    laplace_alpha: float,
) -> np.ndarray:
    current = np.asarray(current_labels, dtype=np.int64)
    following = np.asarray(next_labels, dtype=np.int64)
    if (
        current.shape != following.shape
        or current.ndim != 1
        or class_count < 1
        or laplace_alpha < 0
    ):
        raise ValueError("invalid Markov baseline inputs")
    counts = np.full(
        (class_count, class_count),
        float(laplace_alpha),
        dtype=np.float64,
    )
    np.add.at(counts, (current, following), 1.0)
    return np.argmax(counts, axis=1).astype(np.int64)


def predict_first_order_markov(
    transition_predictions: np.ndarray,
    current_labels: np.ndarray,
) -> np.ndarray:
    table = np.asarray(transition_predictions, dtype=np.int64)
    current = np.asarray(current_labels, dtype=np.int64)
    if table.ndim != 1 or np.any(current < 0) or np.any(current >= table.size):
        raise ValueError("invalid Markov prediction inputs")
    return table[current]


def fit_conditional_transition_destination(
    current_labels: np.ndarray,
    next_labels: np.ndarray,
    *,
    class_count: int,
    laplace_alpha: float,
) -> np.ndarray:
    """Fit a destination table conditional on a codeword change.

    Self-transitions are excluded from both fitting and prediction. Rows with
    no observed outgoing transition fall back to the global changed-destination
    frequency, with deterministic lowest-class tie breaking.
    """

    current = np.asarray(current_labels, dtype=np.int64)
    following = np.asarray(next_labels, dtype=np.int64)
    if (
        current.shape != following.shape
        or current.ndim != 1
        or class_count < 2
        or laplace_alpha < 0
    ):
        raise ValueError("invalid conditional destination inputs")
    changed = current != following
    global_counts = np.full(class_count, float(laplace_alpha))
    if np.any(changed):
        np.add.at(global_counts, following[changed], 1.0)
    table = np.empty(class_count, dtype=np.int64)
    for source in range(class_count):
        counts = global_counts.copy()
        local = changed & (current == source)
        if np.any(local):
            counts = np.full(class_count, float(laplace_alpha))
            np.add.at(counts, following[local], 1.0)
        counts[source] = -np.inf
        table[source] = int(np.flatnonzero(counts == np.max(counts))[0])
    return table
