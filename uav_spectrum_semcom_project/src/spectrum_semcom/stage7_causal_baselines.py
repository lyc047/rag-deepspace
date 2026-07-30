"""Causal task-risk features and table-based Stage-7 baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskState,
    encode_task_state,
)
from spectrum_semcom.stage7_multihorizon_risk import (
    build_multihorizon_risk_labels,
)


FEATURE_NAMES = (
    "power_mean_dbm",
    "power_std_db",
    "power_range_db",
    "adjacent_difference_mean_db",
    "adjacent_difference_max_db",
    "history_available",
    "past_change_mean_db",
    "past_change_max_db",
    "past_change_std_db",
    "rolling3_change_mean_db",
    "rolling5_change_mean_db",
    "previous_action_violation",
    "previous_action_regret_db",
    "action_change_fraction",
    "action_dwell_scenes",
    "current_codeword_id",
    "current_codebook_fallback",
    "current_max_regret_db",
    "minimum_runner_up_margin_db",
    "mean_runner_up_margin_db",
    "maximum_query_action_regret_db",
    "mean_query_action_regret_db",
)


def _rolling_change_mean(
    powers: np.ndarray,
    groups: np.ndarray,
    index: int,
    window: int,
) -> float:
    start = max(0, index - int(window) + 1)
    rows = []
    for current in range(start + 1, index + 1):
        if groups[current - 1] != groups[index] or groups[current] != groups[index]:
            continue
        rows.append(float(np.mean(np.abs(powers[current] - powers[current - 1]))))
    return float(np.mean(rows)) if rows else 0.0


def build_causal_task_risk_dataset(
    channel_power_dbm: np.ndarray,
    states: Sequence[SpectrumTaskState],
    group_ids: Sequence[str],
    codebook: GreedyTaskCodebook,
    *,
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, np.ndarray | tuple[str, ...] | tuple[int, ...]]:
    """Build features observable through scene t and future task labels.

    Only rows with the full maximum horizon inside the same group are returned,
    so every reported horizon uses the same source-scene population.
    """

    powers = np.asarray(channel_power_dbm, dtype=np.float64)
    state_rows = tuple(states)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    if (
        powers.ndim != 2
        or powers.shape[0] != len(state_rows)
        or groups.shape != (len(state_rows),)
        or powers.shape[1] != codebook.n_channels
        or not np.all(np.isfinite(powers))
    ):
        raise ValueError("unaligned causal task-risk inputs")
    decisions = [encode_task_state(codebook, state) for state in state_rows]
    actions = [decision.decoder_actions for decision in decisions]
    labels = build_multihorizon_risk_labels(
        state_rows,
        actions,
        groups,
        horizons=horizons,
    )
    keep = np.asarray(labels["observed"][:, -1], dtype=bool)
    rows = []
    current_symbols = []
    previous_symbols = []
    dwell_rows = []
    source_indices = []
    dwell = 1
    for index, (state, decision) in enumerate(zip(state_rows, decisions)):
        same_previous_group = index > 0 and groups[index - 1] == groups[index]
        if (
            same_previous_group
            and decisions[index - 1].decoder_actions
            == decision.decoder_actions
        ):
            dwell += 1
        else:
            dwell = 1
        if not keep[index]:
            continue
        current = powers[index]
        adjacent = np.abs(np.diff(current))
        if same_previous_group:
            past_change = np.abs(current - powers[index - 1])
            previous_actions = decisions[index - 1].decoder_actions
            previous_regret = state.max_regret_db(previous_actions)
            action_change = float(
                np.mean(
                    np.asarray(previous_actions)
                    != np.asarray(decision.decoder_actions)
                )
            )
            previous_symbol = int(decisions[index - 1].symbol_id)
        else:
            past_change = np.zeros_like(current)
            previous_regret = 0.0
            action_change = 0.0
            previous_symbol = -1
        query_regrets = np.asarray(decision.regret_by_query_db, dtype=np.float64)
        margins = np.asarray(
            [profile.runner_up_gap_db for profile in state.profiles],
            dtype=np.float64,
        )
        finite_margins = margins[np.isfinite(margins)]
        if finite_margins.size == 0:
            finite_margins = np.zeros(1, dtype=np.float64)
        features = [
            float(np.mean(current)),
            float(np.std(current)),
            float(np.ptp(current)),
            float(np.mean(adjacent)) if adjacent.size else 0.0,
            float(np.max(adjacent)) if adjacent.size else 0.0,
            float(same_previous_group),
            float(np.mean(past_change)),
            float(np.max(past_change)),
            float(np.std(past_change)),
            _rolling_change_mean(powers, groups, index, 3),
            _rolling_change_mean(powers, groups, index, 5),
            float(previous_regret > codebook.epsilon_db + 1e-12),
            float(previous_regret),
            action_change,
            float(dwell),
            float(decision.symbol_id),
            float(decision.uses_fallback),
            float(decision.max_regret_db),
            float(np.min(finite_margins)),
            float(np.mean(finite_margins)),
            float(np.max(query_regrets)),
            float(np.mean(query_regrets)),
        ]
        rows.append(features)
        current_symbols.append(int(decision.symbol_id))
        previous_symbols.append(previous_symbol)
        dwell_rows.append(dwell)
        source_indices.append(index)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("causal feature schema mismatch")
    return {
        "features": matrix,
        "feature_names": FEATURE_NAMES,
        "labels": np.asarray(labels["task_failure"][keep], dtype=np.uint8),
        "horizons": labels["horizons"],
        "groups": groups[keep],
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "current_symbol": np.asarray(current_symbols, dtype=np.int64),
        "previous_symbol": np.asarray(previous_symbols, dtype=np.int64),
        "action_dwell_scenes": np.asarray(dwell_rows, dtype=np.int64),
    }


def dwell_bin(values: np.ndarray) -> np.ndarray:
    """Map causal action dwell to fixed bins 1, 2, 3-4, 5-8, and 9+."""

    rows = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(rows < 1):
        raise ValueError("dwell values must be positive")
    return np.digitize(rows, bins=np.asarray([1, 2, 4, 8]), right=True)


def smoothed_lookup_probabilities(
    train_keys: Sequence[object],
    train_labels: np.ndarray,
    evaluation_keys: Sequence[object],
    *,
    prior_strength: float = 10.0,
) -> np.ndarray:
    """Estimate a categorical hazard table with global-prevalence shrinkage."""

    labels = np.asarray(train_labels, dtype=np.float64).reshape(-1)
    train = list(train_keys)
    evaluation = list(evaluation_keys)
    strength = float(prior_strength)
    if (
        len(train) != labels.size
        or labels.size < 1
        or np.any((labels < 0.0) | (labels > 1.0))
        or strength < 0.0
    ):
        raise ValueError("invalid lookup training data")
    prior = float(np.mean(labels))
    counts: dict[object, tuple[int, float]] = {}
    for key, label in zip(train, labels):
        count, total = counts.get(key, (0, 0.0))
        counts[key] = (count + 1, total + float(label))
    output = []
    for key in evaluation:
        count, total = counts.get(key, (0, 0.0))
        output.append(
            (total + strength * prior) / (count + strength)
            if count + strength > 0
            else prior
        )
    return np.asarray(output, dtype=np.float64)


def quantile_lookup_probabilities(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    evaluation_values: np.ndarray,
    *,
    bin_count: int = 10,
    prior_strength: float = 10.0,
) -> np.ndarray:
    """Convert one continuous causal score into calibrated train-only bins."""

    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    evaluation = np.asarray(evaluation_values, dtype=np.float64).reshape(-1)
    if (
        train.size < 1
        or not np.all(np.isfinite(train))
        or not np.all(np.isfinite(evaluation))
        or int(bin_count) < 2
    ):
        raise ValueError("invalid continuous lookup values")
    edges = np.unique(
        np.quantile(train, np.linspace(0.0, 1.0, int(bin_count) + 1)[1:-1])
    )
    train_keys = np.searchsorted(edges, train, side="right")
    evaluation_keys = np.searchsorted(edges, evaluation, side="right")
    return smoothed_lookup_probabilities(
        train_keys.tolist(),
        train_labels,
        evaluation_keys.tolist(),
        prior_strength=prior_strength,
    )
