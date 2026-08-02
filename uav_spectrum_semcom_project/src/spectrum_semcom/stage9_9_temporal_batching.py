"""Causal replay helpers for Stage-9.9 sequential update batching.

The module is intentionally data-agnostic.  It first derives the update events
that an immediate no-loss controller would send, then replays that frozen event
stream with a registered batch size.  It never reads Final data or predicts a
future spectrum state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

import numpy as np

from spectrum_semcom.stage6_task_codebook import CodewordDecision, SpectrumTaskState


@dataclass(frozen=True)
class ScheduledUpdate:
    """One action update on the immediate B1 event schedule."""

    scene_index: int
    decoder_actions: tuple[int, ...]


def packed_batch_payload_bits(
    item_count: int,
    *,
    item_bits: int = 42,
    header_bits: int = 8,
) -> int:
    """Return byte-padded H8 application payload bits for one batch."""

    if int(item_count) < 1 or int(item_bits) < 1 or int(header_bits) < 0:
        raise ValueError("batch widths must be positive")
    return int(header_bits + 8 * math.ceil(int(item_bits) * int(item_count) / 8))


def empirical_cvar(values: Sequence[float], alpha: float = 0.9) -> float:
    """Mean of the largest ceil((1-alpha) n) observations."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)) or not 0.0 <= alpha < 1.0:
        raise ValueError("invalid CVaR input")
    count = max(1, int(math.ceil((1.0 - float(alpha)) * array.size)))
    return float(np.mean(np.sort(array)[-count:]))


def build_immediate_schedule(
    states: Sequence[SpectrumTaskState],
    decisions: Sequence[CodewordDecision],
) -> tuple[tuple[int, ...], tuple[ScheduledUpdate, ...]]:
    """Build the causal no-loss B1 schedule for one temporal group."""

    if not states or len(states) != len(decisions):
        raise ValueError("states and decisions must be non-empty and aligned")
    receiver_actions = tuple(int(value) for value in decisions[0].decoder_actions)
    if not states[0].accepts(receiver_actions):
        raise ValueError("initial semantic decision must be task-safe")
    events: list[ScheduledUpdate] = []
    for index in range(1, len(states)):
        if states[index].accepts(receiver_actions):
            continue
        receiver_actions = tuple(
            int(value) for value in decisions[index].decoder_actions
        )
        if not states[index].accepts(receiver_actions):
            raise ValueError("semantic decision must be task-safe")
        events.append(ScheduledUpdate(index, receiver_actions))
    return tuple(int(value) for value in decisions[0].decoder_actions), tuple(events)


def _seconds_between(timestamps: Sequence[str], start: int, end: int) -> float:
    if end < start:
        raise ValueError("wait end precedes event")
    if end == start:
        return 0.0
    value = (
        datetime.fromisoformat(str(timestamps[end]))
        - datetime.fromisoformat(str(timestamps[start]))
    ).total_seconds()
    if value < 0:
        raise ValueError("timestamps must not decrease")
    return float(value)


def replay_frozen_schedule(
    states: Sequence[SpectrumTaskState],
    timestamps: Sequence[str],
    initial_actions: Sequence[int],
    events: Sequence[ScheduledUpdate],
    *,
    batch_size: int,
    loss_mask: Sequence[bool] | None = None,
    item_bits: int = 42,
    header_bits: int = 8,
) -> dict:
    """Replay one frozen event stream and return task/transport measurements.

    Full batches are sent before the task decision at the scene containing the
    final item.  A group-end remainder is charged after the final scene and thus
    cannot improve a past task decision.
    """

    count = len(states)
    if count < 1 or len(timestamps) != count or int(batch_size) < 1:
        raise ValueError("invalid replay inputs")
    mask = (
        np.zeros(count, dtype=np.bool_)
        if loss_mask is None
        else np.asarray(loss_mask, dtype=np.bool_)
    )
    if mask.shape != (count,):
        raise ValueError("loss mask must have one entry per scene")
    event_map: dict[int, ScheduledUpdate] = {}
    previous = 0
    for position, event in enumerate(events):
        index = int(event.scene_index)
        if index < 1 or index >= count or (position and index <= previous):
            raise ValueError("events must be unique, ordered, and inside the group")
        event_map[index] = event
        previous = index

    receiver_actions = tuple(int(value) for value in initial_actions)
    pending: list[ScheduledUpdate] = []
    regrets: list[float] = []
    waits_scenes: list[int] = []
    waits_seconds: list[float] = []
    datagrams = 0
    lost_datagrams = 0
    payload_bits = 0
    delivered_items = 0
    end_remainder_items = 0

    def transmit(scene_index: int, *, affects_current_scene: bool) -> None:
        nonlocal receiver_actions, datagrams, lost_datagrams, payload_bits
        nonlocal delivered_items, end_remainder_items
        if not pending:
            return
        item_count = len(pending)
        datagrams += 1
        payload_bits += packed_batch_payload_bits(
            item_count, item_bits=item_bits, header_bits=header_bits
        )
        for update in pending:
            waits_scenes.append(int(scene_index - update.scene_index))
            waits_seconds.append(
                _seconds_between(timestamps, update.scene_index, scene_index)
            )
        if bool(mask[scene_index]):
            lost_datagrams += 1
        else:
            delivered_items += item_count
            if affects_current_scene:
                receiver_actions = pending[-1].decoder_actions
        if not affects_current_scene:
            end_remainder_items += item_count
        pending.clear()

    for scene_index, state in enumerate(states):
        event = event_map.get(scene_index)
        if event is not None:
            pending.append(event)
            if len(pending) == int(batch_size):
                transmit(scene_index, affects_current_scene=True)
        regret = float(state.max_regret_db(receiver_actions))
        regrets.append(regret)

    if pending:
        transmit(count - 1, affects_current_scene=False)

    regret_array = np.asarray(regrets, dtype=np.float64)
    epsilon = float(states[0].epsilon_db)
    clean = regret_array <= epsilon + 1e-12
    return {
        "scene_count": int(count),
        "event_count": int(len(events)),
        "batch_size": int(batch_size),
        "datagram_count": int(datagrams),
        "lost_datagram_count": int(lost_datagrams),
        "delivered_update_item_count": int(delivered_items),
        "group_end_remainder_item_count": int(end_remainder_items),
        "h8_update_path_bits": int(payload_bits),
        "clean_scene_count": int(np.sum(clean)),
        "clean_rate": float(np.mean(clean)),
        "stale_action_scene_count": int(np.sum(~clean)),
        "mean_max_query_regret_db": float(np.mean(regret_array)),
        "cvar_0_9_max_query_regret_db": empirical_cvar(regret_array, 0.9),
        "maximum_max_query_regret_db": float(np.max(regret_array)),
        "mean_event_wait_scenes": float(np.mean(waits_scenes)) if waits_scenes else 0.0,
        "maximum_event_wait_scenes": int(max(waits_scenes, default=0)),
        "mean_event_wait_seconds": float(np.mean(waits_seconds)) if waits_seconds else 0.0,
        "maximum_event_wait_seconds": float(max(waits_seconds, default=0.0)),
        "scene_regrets_db": [float(value) for value in regret_array],
    }


def markov_loss_mask(
    scene_count: int,
    *,
    seed: int,
    good_to_bad: float,
    bad_to_good: float,
    drop_good: float,
    drop_bad: float,
    stationary_initial_state: bool = True,
) -> np.ndarray:
    """Generate a deterministic scene-indexed two-state Markov loss mask."""

    probabilities = (good_to_bad, bad_to_good, drop_good, drop_bad)
    if int(scene_count) < 1 or any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
        raise ValueError("invalid Markov loss configuration")
    if float(good_to_bad) + float(bad_to_good) <= 0.0:
        raise ValueError("Markov chain must permit a state transition")
    rng = np.random.default_rng(int(seed))
    stationary_bad = float(good_to_bad) / float(good_to_bad + bad_to_good)
    bad = bool(rng.random() < stationary_bad) if stationary_initial_state else False
    losses = np.zeros(int(scene_count), dtype=np.bool_)
    for index in range(int(scene_count)):
        losses[index] = bool(rng.random() < (drop_bad if bad else drop_good))
        if bad:
            if rng.random() < float(bad_to_good):
                bad = False
        elif rng.random() < float(good_to_bad):
            bad = True
    return losses
