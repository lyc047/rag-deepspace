"""Causal multi-horizon labels and ideal-risk controls for Stage 7.

The labels intentionally use future observations only as supervised targets.
No future value is returned as an inference feature.  Group boundaries are
hard barriers so a site, activity, or temporal cluster cannot label another.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spectrum_semcom.stage6_task_codebook import SpectrumTaskState


def _validated_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if (
        not values
        or any(value < 1 for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError("horizons must be unique positive values in ascending order")
    return values


def _group_end_indices(groups: np.ndarray) -> np.ndarray:
    ends = np.empty(groups.size, dtype=np.int64)
    start = 0
    while start < groups.size:
        end = start + 1
        while end < groups.size and groups[end] == groups[start]:
            end += 1
        ends[start:end] = end
        start = end
    return ends


def build_multihorizon_risk_labels(
    states: Sequence[SpectrumTaskState],
    current_actions: Sequence[Sequence[int]],
    group_ids: Sequence[str],
    *,
    horizons: Sequence[int] = (1, 3, 5, 10),
    context_failure_events: Sequence[bool] | None = None,
    tolerance: float = 1e-12,
) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Build task, context, and union failure labels.

    For source scene ``t`` and horizon ``h``, the task label is one when the
    action available at ``t`` exceeds the task regret threshold at any scene
    from ``t + 1`` through ``t + h``.  The context label is one when a supplied
    context-failure event occurs in the same future interval.  A row is marked
    observed only when the complete horizon remains inside the same group.
    """

    state_rows = tuple(states)
    action_rows = tuple(tuple(int(value) for value in row) for row in current_actions)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    horizon_values = _validated_horizons(horizons)
    count = len(state_rows)
    if (
        count < 2
        or len(action_rows) != count
        or groups.shape != (count,)
    ):
        raise ValueError("states, actions, and groups must be aligned")
    reference = state_rows[0]
    for state, actions in zip(state_rows, action_rows):
        if (
            state.n_channels != reference.n_channels
            or state.schema != reference.schema
            or abs(state.epsilon_db - reference.epsilon_db) > 1e-12
            or len(actions) != len(state.profiles)
        ):
            raise ValueError("incompatible state or action schema")
        state.max_regret_db(actions)
    context = (
        np.zeros(count, dtype=bool)
        if context_failure_events is None
        else np.asarray(context_failure_events, dtype=bool).reshape(-1)
    )
    if context.shape != (count,):
        raise ValueError("context failure events must align with states")

    group_ends = _group_end_indices(groups)
    shape = (count, len(horizon_values))
    observed = np.zeros(shape, dtype=bool)
    task_labels = np.zeros(shape, dtype=np.uint8)
    context_labels = np.zeros(shape, dtype=np.uint8)
    task_max_regret = np.full(shape, np.nan, dtype=np.float64)
    task_time = np.full(count, -1, dtype=np.int64)
    context_time = np.full(count, -1, dtype=np.int64)
    any_time = np.full(count, -1, dtype=np.int64)

    max_horizon = horizon_values[-1]
    for index, (state, actions) in enumerate(zip(state_rows, action_rows)):
        del state
        available = min(max_horizon, int(group_ends[index] - index - 1))
        if available < 1:
            continue
        future_regrets = np.asarray(
            [
                state_rows[index + offset].max_regret_db(actions)
                for offset in range(1, available + 1)
            ],
            dtype=np.float64,
        )
        future_task_failures = (
            future_regrets > reference.epsilon_db + float(tolerance)
        )
        future_context_failures = context[index + 1 : index + available + 1]
        task_hits = np.flatnonzero(future_task_failures)
        context_hits = np.flatnonzero(future_context_failures)
        if task_hits.size:
            task_time[index] = int(task_hits[0] + 1)
        if context_hits.size:
            context_time[index] = int(context_hits[0] + 1)
        candidates = [
            value
            for value in (task_time[index], context_time[index])
            if value >= 1
        ]
        if candidates:
            any_time[index] = int(min(candidates))
        for position, horizon in enumerate(horizon_values):
            if horizon > available:
                continue
            observed[index, position] = True
            task_max_regret[index, position] = float(
                np.max(future_regrets[:horizon])
            )
            task_labels[index, position] = np.uint8(
                np.any(future_task_failures[:horizon])
            )
            context_labels[index, position] = np.uint8(
                np.any(future_context_failures[:horizon])
            )

    any_labels = np.maximum(task_labels, context_labels).astype(np.uint8)
    return {
        "horizons": horizon_values,
        "observed": observed,
        "task_failure": task_labels,
        "context_failure": context_labels,
        "any_failure": any_labels,
        "task_max_regret_db": task_max_regret,
        "task_time_to_failure": task_time,
        "context_time_to_failure": context_time,
        "any_time_to_failure": any_time,
        "source_indices": np.arange(count, dtype=np.int64),
        "groups": groups,
    }


def perfect_reset_oracle_trigger_mask(
    random_values: np.ndarray,
    *,
    reset_probability: float,
    reset_stream_index: int,
) -> np.ndarray:
    """Return a non-causal upper-bound trigger at actual receiver resets."""

    values = np.asarray(random_values, dtype=np.float64)
    probability = float(reset_probability)
    if (
        values.ndim != 2
        or not 0 <= int(reset_stream_index) < values.shape[1]
        or not 0.0 <= probability <= 1.0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid random stream or reset probability")
    return values[:, int(reset_stream_index)] < probability
