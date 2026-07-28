"""Causal features for forecasting task-semantic spectrum changes."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskState,
    encode_task_state,
)


def temporal_hazard_feature_names(query_count: int) -> tuple[str, ...]:
    if query_count < 1:
        raise ValueError("query count must be positive")
    common = (
        "power_mean_dbm",
        "power_std_db",
        "power_range_db",
        "adjacent_difference_mean_db",
        "adjacent_difference_max_db",
        "history_available",
        "past_change_mean_db",
        "past_change_max_db",
        "past_change_std_db",
        "past_elapsed_seconds",
        "previous_action_violation",
        "previous_action_change_fraction",
        "codebook_fallback",
    )
    query = tuple(
        name
        for index in range(query_count)
        for name in (
            f"query_{index}_best_second_margin_db",
            f"query_{index}_encoded_action_regret_db",
        )
    )
    return common + query


def _elapsed_seconds(start: str, end: str) -> float:
    value = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds()
    if value <= 0:
        raise ValueError("timestamps must increase within a trajectory")
    return float(value)


def build_temporal_hazard_dataset(
    channel_power_dbm: np.ndarray,
    states: list[SpectrumTaskState],
    timestamps: np.ndarray,
    cluster_ids: np.ndarray,
    codebook: GreedyTaskCodebook,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Build one-step labels using only information available through time t."""

    powers = np.asarray(channel_power_dbm, dtype=np.float64)
    times = np.asarray(timestamps).astype(str)
    groups = np.asarray(cluster_ids).astype(str)
    if (
        powers.ndim != 2
        or powers.shape[0] != len(states)
        or times.shape != (len(states),)
        or groups.shape != (len(states),)
        or powers.shape[1] != codebook.n_channels
        or not np.all(np.isfinite(powers))
    ):
        raise ValueError("unaligned temporal hazard inputs")
    decisions = [encode_task_state(codebook, state) for state in states]
    names = temporal_hazard_feature_names(len(codebook.queries))
    rows = []
    labels = []
    next_regrets = []
    sample_groups = []
    source_indices = []

    for index in range(len(states) - 1):
        if groups[index] != groups[index + 1]:
            continue
        current = powers[index]
        adjacent = np.abs(np.diff(current))
        has_history = index > 0 and groups[index - 1] == groups[index]
        if has_history:
            past_change = np.abs(current - powers[index - 1])
            elapsed = _elapsed_seconds(times[index - 1], times[index])
            previous_actions = decisions[index - 1].decoder_actions
            previous_regret = states[index].max_regret_db(
                previous_actions
            )
            action_change = float(
                np.mean(
                    np.asarray(previous_actions)
                    != np.asarray(decisions[index].decoder_actions)
                )
            )
        else:
            past_change = np.zeros_like(current)
            elapsed = 0.0
            previous_regret = 0.0
            action_change = 0.0
        features = [
            float(np.mean(current)),
            float(np.std(current)),
            float(np.ptp(current)),
            float(np.mean(adjacent)) if adjacent.size else 0.0,
            float(np.max(adjacent)) if adjacent.size else 0.0,
            float(has_history),
            float(np.mean(past_change)),
            float(np.max(past_change)),
            float(np.std(past_change)),
            elapsed,
            float(previous_regret > codebook.epsilon_db + 1e-12),
            action_change,
            float(decisions[index].uses_fallback),
        ]
        for profile, action in zip(
            states[index].profiles,
            decisions[index].decoder_actions,
        ):
            ordered = np.sort(profile.candidate_costs_dbm)
            margin = (
                float(ordered[1] - ordered[0])
                if ordered.size > 1
                else 0.0
            )
            features.extend(
                [
                    margin,
                    float(profile.regret_db(int(action))),
                ]
            )
        next_regret = states[index + 1].max_regret_db(
            decisions[index].decoder_actions
        )
        rows.append(features)
        next_regrets.append(next_regret)
        labels.append(next_regret > codebook.epsilon_db + 1e-12)
        sample_groups.append(groups[index])
        source_indices.append(index)

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("temporal hazard feature schema mismatch")
    return {
        "features": matrix,
        "labels": np.asarray(labels, dtype=np.uint8),
        "next_regret_db": np.asarray(next_regrets, dtype=np.float64),
        "groups": np.asarray(sample_groups),
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "feature_names": names,
    }
