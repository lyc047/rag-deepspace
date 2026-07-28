"""Closed-loop repetition protection for exact Stage-6 semantic updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from spectrum_semcom.stage6_context_codec import (
    decode_compact_update,
    encode_compact_update,
)
from spectrum_semcom.stage6_task_codebook import encode_task_state


@dataclass
class PredictiveReceiverState:
    decoder_actions: tuple[int, ...] | None = None
    success_timestamp: str | None = None


@dataclass(frozen=True)
class PredictiveRepetitionResult:
    total_application_bits: int
    extra_repetition_bits: int
    reserved_capacity_bits: int
    update_count: int
    reservation_count: int
    protected_update_count: int
    failed_update_count: int
    clean: tuple[bool, ...]
    available: tuple[bool, ...]
    effective_regret_db: tuple[float, ...]


def _age_minutes(previous: str | None, current: str) -> float:
    if previous is None:
        return float("inf")
    value = (
        datetime.fromisoformat(current) - datetime.fromisoformat(previous)
    ).total_seconds() / 60.0
    if value < 0:
        raise ValueError("timestamps are not causal")
    return float(value)


def ideal_hard_update_candidates(
    states,
    timestamps: np.ndarray,
    cluster_ids: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    session,
    max_age_minutes: float,
) -> np.ndarray:
    """Return a non-deployable no-loss schedule of exact hard triggers."""

    candidates = np.zeros(len(states), dtype=bool)
    actions: tuple[int, ...] | None = None
    success_timestamp: str | None = None
    previous_group: str | None = None
    for global_index in np.asarray(evaluation_indices, dtype=np.int64):
        group = str(cluster_ids[int(global_index)])
        timestamp = str(timestamps[int(global_index)])
        if previous_group is not None and group != previous_group:
            actions = None
            success_timestamp = None
        previous_group = group
        task_state = states[int(global_index)]
        regret = (
            float("inf")
            if actions is None
            else task_state.max_regret_db(actions)
        )
        age = _age_minutes(success_timestamp, timestamp)
        should_update = (
            actions is None
            or regret > session.codebook.epsilon_db + 1e-12
            or age > float(max_age_minutes)
        )
        if should_update:
            candidates[int(global_index)] = True
            actions = encode_task_state(
                session.codebook, task_state
            ).decoder_actions
            success_timestamp = timestamp
    return candidates


def simulate_predictive_repetition(
    states,
    timestamps: np.ndarray,
    cluster_ids: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    session,
    protection_candidates: np.ndarray,
    maximum_reservations: int,
    reservation_equivalent_bits: int,
    task_loss_probability: float,
    random_values: np.ndarray,
    max_age_minutes: float,
    outage_penalty_db: float,
) -> PredictiveRepetitionResult:
    """Run one link trajectory with ideal ACK and optional duplicate frames."""

    indices = np.asarray(evaluation_indices, dtype=np.int64)
    candidates = np.asarray(protection_candidates, dtype=bool)
    random = np.asarray(random_values, dtype=np.float64)
    if (
        candidates.shape != (len(states),)
        or random.shape != (indices.size, 2)
        or not 0.0 <= float(task_loss_probability) <= 1.0
        or maximum_reservations < 0
        or reservation_equivalent_bits < 1
        or max_age_minutes <= 0
        or outage_penalty_db <= 0
    ):
        raise ValueError("invalid predictive repetition inputs")

    receiver = PredictiveReceiverState()
    previous_group: str | None = None
    total_bits = 0
    extra_bits = 0
    reserved_bits = 0
    update_count = 0
    reservation_count = 0
    protected_count = 0
    failed_count = 0
    clean_rows = []
    available_rows = []
    regret_rows = []
    update_epoch = 0

    for local_position, global_index in enumerate(indices):
        group = str(cluster_ids[int(global_index)])
        timestamp = str(timestamps[int(global_index)])
        if previous_group is not None and group != previous_group:
            receiver = PredictiveReceiverState()
        previous_group = group
        task_state = states[int(global_index)]
        reserved = (
            bool(candidates[int(global_index)])
            and reservation_count < int(maximum_reservations)
        )
        if reserved:
            reservation_count += 1
            reserved_bits += int(reservation_equivalent_bits)
        current_regret = (
            float("inf")
            if receiver.decoder_actions is None
            else task_state.max_regret_db(receiver.decoder_actions)
        )
        current_age = _age_minutes(
            receiver.success_timestamp, timestamp
        )
        should_update = (
            receiver.decoder_actions is None
            or current_regret > session.codebook.epsilon_db + 1e-12
            or current_age > float(max_age_minutes)
        )
        if should_update:
            update_epoch = (update_epoch + 1) % 256
            bits, decision = encode_compact_update(
                task_state,
                session,
                node_id=1,
                update_epoch=update_epoch,
            )
            packet_bits = int(bits.size)
            protected = reserved
            total_bits += packet_bits
            update_count += 1
            primary_delivered = (
                random[local_position, 0] >= task_loss_probability
            )
            delivered = primary_delivered
            if protected:
                total_bits += packet_bits
                extra_bits += packet_bits
                protected_count += 1
                delivered = delivered or (
                    random[local_position, 1] >= task_loss_probability
                )
            if delivered:
                decoded = decode_compact_update(bits, session)
                if decoded.decoder_actions != decision.decoder_actions:
                    raise ValueError("compact update round trip changed action")
                receiver.decoder_actions = decoded.decoder_actions
                receiver.success_timestamp = timestamp
            else:
                failed_count += 1

        available = receiver.decoder_actions is not None
        available_rows.append(available)
        if available:
            regret = task_state.max_regret_db(
                receiver.decoder_actions
            )
            regret_rows.append(regret)
            clean_rows.append(
                regret <= session.codebook.epsilon_db + 1e-12
            )
        else:
            regret_rows.append(float(outage_penalty_db))
            clean_rows.append(False)

    return PredictiveRepetitionResult(
        total_application_bits=total_bits,
        extra_repetition_bits=extra_bits,
        reserved_capacity_bits=reserved_bits,
        update_count=update_count,
        reservation_count=reservation_count,
        protected_update_count=protected_count,
        failed_update_count=failed_count,
        clean=tuple(clean_rows),
        available=tuple(available_rows),
        effective_regret_db=tuple(regret_rows),
    )
