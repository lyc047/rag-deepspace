"""Task-event decomposition and forced-refresh value labels for S7.3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskState,
    encode_task_state,
)


def maximum_true_run(values: np.ndarray) -> int:
    """Return the maximum number of consecutive true entries."""

    flags = np.asarray(values, dtype=bool).reshape(-1)
    best = 0
    current = 0
    for value in flags:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def build_task_event_value_labels(
    states: Sequence[SpectrumTaskState],
    group_ids: Sequence[str],
    codebook: GreedyTaskCodebook,
    *,
    horizon: int = 10,
    persistent_run_lengths: Sequence[int] = (2, 3),
    refresh_offset: int = 1,
    incremental_bits: int,
    clean_scene_value_bits: Sequence[int] = (32, 64, 128, 256),
) -> dict:
    """Build transient, persistent, and next-scene refresh-value labels.

    At source scene t, the baseline holds the decoder action selected at t.
    The counterfactual schedules one forced update for t + refresh_offset,
    where the action is recomputed using that scene's then-current spectrum.
    """

    state_rows = tuple(states)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    horizon = int(horizon)
    offset = int(refresh_offset)
    run_lengths = tuple(sorted(set(map(int, persistent_run_lengths))))
    scene_values = tuple(sorted(set(map(int, clean_scene_value_bits))))
    if (
        len(state_rows) != groups.size
        or horizon < 1
        or not 1 <= offset <= horizon
        or not run_lengths
        or any(value < 2 or value > horizon for value in run_lengths)
        or int(incremental_bits) < 1
        or not scene_values
        or any(value < 1 for value in scene_values)
    ):
        raise ValueError("invalid task event-value specification")
    decisions = [encode_task_state(codebook, state) for state in state_rows]
    source_indices = []
    baseline_dirty_rows = []
    refreshed_dirty_rows = []
    baseline_excess_rows = []
    refreshed_excess_rows = []
    maximum_run_rows = []
    current_symbols = []
    previous_symbols = []
    dwell_rows = []
    persistent_rows = {value: [] for value in run_lengths}
    net_positive_rows = {value: [] for value in scene_values}
    dwell = 1
    for index, decision in enumerate(decisions):
        same_previous = index > 0 and groups[index - 1] == groups[index]
        if (
            same_previous
            and decisions[index - 1].decoder_actions
            == decision.decoder_actions
        ):
            dwell += 1
        else:
            dwell = 1
        end = index + horizon
        if end >= len(state_rows) or np.any(
            groups[index : end + 1] != groups[index]
        ):
            continue
        baseline_action = decision.decoder_actions
        refreshed_action = decisions[index + offset].decoder_actions
        future_states = state_rows[index + 1 : end + 1]
        baseline_regret = np.asarray(
            [state.max_regret_db(baseline_action) for state in future_states],
            dtype=np.float64,
        )
        refreshed_regret = np.asarray(
            [state.max_regret_db(refreshed_action) for state in future_states],
            dtype=np.float64,
        )
        epsilon = float(codebook.epsilon_db)
        baseline_dirty = baseline_regret > epsilon + 1e-12
        refreshed_dirty = refreshed_regret > epsilon + 1e-12
        baseline_count = int(np.sum(baseline_dirty))
        refreshed_count = int(np.sum(refreshed_dirty))
        avoided_dirty = baseline_count - refreshed_count
        baseline_excess = float(
            np.sum(np.maximum(0.0, baseline_regret - epsilon))
        )
        refreshed_excess = float(
            np.sum(np.maximum(0.0, refreshed_regret - epsilon))
        )
        source_indices.append(index)
        baseline_dirty_rows.append(baseline_count)
        refreshed_dirty_rows.append(refreshed_count)
        baseline_excess_rows.append(baseline_excess)
        refreshed_excess_rows.append(refreshed_excess)
        maximum_run = maximum_true_run(baseline_dirty)
        maximum_run_rows.append(maximum_run)
        current_symbols.append(int(decision.symbol_id))
        previous_symbols.append(
            int(decisions[index - 1].symbol_id) if same_previous else -1
        )
        dwell_rows.append(dwell)
        for value in run_lengths:
            persistent_rows[value].append(int(maximum_run >= value))
        for value in scene_values:
            net_positive_rows[value].append(
                int(avoided_dirty * value > int(incremental_bits))
            )
    baseline_dirty_array = np.asarray(
        baseline_dirty_rows, dtype=np.int64
    )
    refreshed_dirty_array = np.asarray(
        refreshed_dirty_rows, dtype=np.int64
    )
    avoided_dirty = baseline_dirty_array - refreshed_dirty_array
    baseline_excess = np.asarray(
        baseline_excess_rows, dtype=np.float64
    )
    refreshed_excess = np.asarray(
        refreshed_excess_rows, dtype=np.float64
    )
    primary_run = run_lengths[0]
    persistent_primary = np.asarray(
        persistent_rows[primary_run], dtype=np.uint8
    )
    any_failure = (baseline_dirty_array > 0).astype(np.uint8)
    return {
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "groups": groups[np.asarray(source_indices, dtype=np.int64)],
        "any_failure": any_failure,
        "transient_failure": (
            (any_failure == 1) & (persistent_primary == 0)
        ).astype(np.uint8),
        "persistent_failure": {
            str(value): np.asarray(rows, dtype=np.uint8)
            for value, rows in persistent_rows.items()
        },
        "baseline_dirty_scenes": baseline_dirty_array,
        "refreshed_dirty_scenes": refreshed_dirty_array,
        "avoided_dirty_scenes": avoided_dirty,
        "baseline_excess_regret_db": baseline_excess,
        "refreshed_excess_regret_db": refreshed_excess,
        "avoided_excess_regret_db": baseline_excess - refreshed_excess,
        "forced_refresh_beneficial": (avoided_dirty > 0).astype(np.uint8),
        "break_even_bits_per_avoided_dirty_scene": np.divide(
            float(incremental_bits),
            avoided_dirty,
            out=np.full(avoided_dirty.shape, np.inf, dtype=np.float64),
            where=avoided_dirty > 0,
        ),
        "net_positive_by_clean_scene_value_bits": {
            str(value): np.asarray(rows, dtype=np.uint8)
            for value, rows in net_positive_rows.items()
        },
        "current_symbol": np.asarray(current_symbols, dtype=np.int64),
        "previous_symbol": np.asarray(previous_symbols, dtype=np.int64),
        "action_dwell_scenes": np.asarray(dwell_rows, dtype=np.int64),
        "incremental_bits": int(incremental_bits),
        "horizon": horizon,
        "refresh_offset": offset,
    }
