"""Matched-reliability simulators for the frozen Stage-6 controller.

This module is a new measurement/execution layer. It does not modify the
S6-FC0 implementation. It adds two preregistered session initializations,
paired open-loop packet attempts, and explicit component-level bit accounts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from spectrum_semcom.stage5_cumulative_ack import (
    decode_cumulative_ack,
    encode_cumulative_ack,
)
from spectrum_semcom.stage6_bit_accounting import (
    Stage6BitBreakdown,
    validate_breakdown_identity,
)
from spectrum_semcom.stage6_context_codec import (
    decode_compact_update,
    decode_context_install,
    encode_compact_update,
)
from spectrum_semcom.stage6_context_heartbeat import (
    decode_context_probe_request,
    decode_context_probe_response,
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_context_recovery import (
    ContextBelief,
    ReceiverContextHypothesis,
    belief_maximum_age_minutes,
    belief_requires_context_install,
    belief_worst_case_regret_db,
    compact_update_attempt,
    condition_belief_on_cumulative_ack,
    context_install_attempt,
    initial_context_belief,
)


RANDOM_STREAM_COUNT = 11
RESET_STREAM = 0
INSTALL_STREAM = 1
INSTALL_ACK_STREAM = 2
TASK_ATTEMPT_STREAMS = (3, 4, 5)
UPDATE_ACK_STREAM = 6
DELAYED_ACK_STREAM = 7
HEARTBEAT_REQUEST_STREAM = 8
HEARTBEAT_RESPONSE_STREAM = 9
PROTECTION_ATTEMPT_STREAM = 10

DEPLOYMENT_MODES = (
    "preconfigured_codebook_empty_action_state",
    "online_install_empty_context",
)


@dataclass
class _ReceiverState:
    context_installed: bool = False
    codebook_epoch: int | None = None
    actions: tuple[int, ...] | None = None
    update_epoch: int | None = None
    success_timestamp: str | None = None


@dataclass(frozen=True)
class MatchedTrajectoryResult:
    bit_breakdown: Stage6BitBreakdown
    context_install_count: int
    compact_update_count: int
    duplicate_update_count: int
    update_reservation_count: int
    protected_update_count: int
    ack_frame_count: int
    stale_ack_rejection_count: int
    rejected_compact_count: int
    wrong_codebook_decode_count: int
    maximum_belief_size: int
    heartbeat_probe_count: int
    heartbeat_response_count: int
    heartbeat_failure_count: int
    available: tuple[bool, ...]
    clean: tuple[bool, ...]
    effective_regret_db: tuple[float, ...]
    available_regret_db: tuple[float, ...]


def build_grouped_sessions(
    evaluation_indices: np.ndarray,
    group_ids: np.ndarray,
    *,
    target_scene_count: int,
    minimum_remainder_scenes: int = 2,
) -> tuple[np.ndarray, ...]:
    """Split ordered evaluation rows without crossing group boundaries."""

    indices = np.asarray(evaluation_indices, dtype=np.int64).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    if (
        indices.size < 1
        or int(target_scene_count) < 2
        or int(minimum_remainder_scenes) < 1
        or np.any(indices < 0)
        or np.any(indices >= groups.size)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("invalid grouped-session inputs")
    sessions: list[np.ndarray] = []
    start = 0
    while start < indices.size:
        group = groups[int(indices[start])]
        end = start + 1
        while (
            end < indices.size
            and groups[int(indices[end])] == group
        ):
            end += 1
        group_indices = indices[start:end]
        for chunk_start in range(0, group_indices.size, target_scene_count):
            chunk = group_indices[
                chunk_start : chunk_start + int(target_scene_count)
            ]
            if chunk.size >= int(minimum_remainder_scenes):
                sessions.append(chunk.copy())
        start = end
    if not sessions:
        raise ValueError("session split produced no usable sessions")
    return tuple(sessions)


def combine_matched_trajectory_results(
    values: list[MatchedTrajectoryResult],
) -> MatchedTrajectoryResult:
    """Combine independent sessions from one paired Monte Carlo trajectory."""

    if not values:
        raise ValueError("cannot combine an empty trajectory list")
    bit_fields = tuple(values[0].bit_breakdown.to_dict())
    bit_totals = {
        field: sum(
            int(value.bit_breakdown.to_dict()[field]) for value in values
        )
        for field in bit_fields
    }
    breakdown = Stage6BitBreakdown(**bit_totals)
    validate_breakdown_identity(breakdown)
    sum_fields = (
        "context_install_count",
        "compact_update_count",
        "duplicate_update_count",
        "update_reservation_count",
        "protected_update_count",
        "ack_frame_count",
        "stale_ack_rejection_count",
        "rejected_compact_count",
        "wrong_codebook_decode_count",
        "heartbeat_probe_count",
        "heartbeat_response_count",
        "heartbeat_failure_count",
    )
    sums = {
        field: sum(int(getattr(value, field)) for value in values)
        for field in sum_fields
    }
    return MatchedTrajectoryResult(
        bit_breakdown=breakdown,
        **sums,
        maximum_belief_size=max(
            int(value.maximum_belief_size) for value in values
        ),
        available=tuple(
            item for value in values for item in value.available
        ),
        clean=tuple(item for value in values for item in value.clean),
        effective_regret_db=tuple(
            item for value in values for item in value.effective_regret_db
        ),
        available_regret_db=tuple(
            item for value in values for item in value.available_regret_db
        ),
    )


def exact_query_bundle_bits(
    n_channels: int,
    queries: tuple[Any, ...],
    *,
    header_bits: int,
) -> int:
    """Return the self-contained exact multi-query action packet width."""

    payload = 0
    for query in queries:
        alphabet_size = int(n_channels) - int(query.demand_channels) + 1
        if alphabet_size < 1:
            raise ValueError("query exceeds channel grid")
        payload += int(math.ceil(math.log2(alphabet_size)))
    return int(header_bits) + payload


def _age_minutes(previous: str | None, current: str) -> float:
    if previous is None:
        return float("inf")
    return (
        datetime.fromisoformat(current) - datetime.fromisoformat(previous)
    ).total_seconds() / 60.0


def _validate_common_inputs(
    states: list[Any],
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    random_values: np.ndarray,
    *,
    task_open_loop_attempts: int,
) -> np.ndarray:
    indices = np.asarray(evaluation_indices, dtype=np.int64).reshape(-1)
    random = np.asarray(random_values, dtype=np.float64)
    if (
        not 1 <= int(task_open_loop_attempts) <= len(TASK_ATTEMPT_STREAMS)
        or random.shape != (indices.size, RANDOM_STREAM_COUNT)
        or np.any((random < 0.0) | (random > 1.0))
        or np.any(indices < 0)
        or np.any(indices >= len(states))
        or len(timestamps) < len(states)
    ):
        raise ValueError("invalid matched-reliability trajectory inputs")
    return indices


def _preconfigured_belief(codebook_epoch: int) -> ContextBelief:
    return frozenset(
        {
            ReceiverContextHypothesis(
                context_installed=True,
                codebook_epoch=int(codebook_epoch),
                update_epoch=None,
                decoder_actions=None,
                success_timestamp=None,
            )
        }
    )


def _finish_breakdown(
    *,
    task_frame_bits: int,
    exact_action_frame_bits: int,
    escape_exact_bits: int,
    initial_install_bits: int,
    recovery_install_bits: int,
    compact_update_bits: int,
    duplicate_update_bits: int,
    heartbeat_request_bits: int,
    ack_bits: int,
    heartbeat_response_bits: int,
    reserved_capacity_bits: int,
    used_reserved_capacity_bits: int,
) -> Stage6BitBreakdown:
    actual_forward = (
        int(task_frame_bits)
        + int(initial_install_bits)
        + int(recovery_install_bits)
        + int(heartbeat_request_bits)
    )
    actual_feedback = int(ack_bits) + int(heartbeat_response_bits)
    actual_total = actual_forward + actual_feedback
    unused_reserved = (
        int(reserved_capacity_bits) - int(used_reserved_capacity_bits)
    )
    if unused_reserved < 0:
        raise ValueError("used reserved capacity exceeds reservation")
    value = Stage6BitBreakdown(
        task_frame_bits=int(task_frame_bits),
        exact_action_frame_bits=int(exact_action_frame_bits),
        escape_exact_bits=int(escape_exact_bits),
        initial_install_bits=int(initial_install_bits),
        recovery_install_bits=int(recovery_install_bits),
        compact_update_bits=int(compact_update_bits),
        duplicate_update_bits=int(duplicate_update_bits),
        heartbeat_request_bits=int(heartbeat_request_bits),
        ack_bits=int(ack_bits),
        heartbeat_response_bits=int(heartbeat_response_bits),
        other_feedback_bits=0,
        actual_forward_bits=actual_forward,
        actual_feedback_bits=actual_feedback,
        actual_total_application_bits=actual_total,
        reserved_capacity_bits=int(reserved_capacity_bits),
        used_reserved_capacity_bits=int(used_reserved_capacity_bits),
        unused_reserved_capacity_bits=unused_reserved,
        resource_equivalent_bits=actual_total + unused_reserved,
    )
    validate_breakdown_identity(value)
    return value


def simulate_exact_trajectory(
    states: list[Any],
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    random_values: np.ndarray,
    packet_loss_probability: float,
    receiver_reset_probability: float,
    task_open_loop_attempts: int,
    packet_bits: int,
    epsilon_db: float,
    outage_penalty_db: float,
) -> MatchedTrajectoryResult:
    """Simulate the paired self-contained exact-action reference."""

    indices = _validate_common_inputs(
        states,
        timestamps,
        evaluation_indices,
        random_values,
        task_open_loop_attempts=task_open_loop_attempts,
    )
    if (
        not 0.0 <= float(packet_loss_probability) <= 1.0
        or not 0.0 <= float(receiver_reset_probability) <= 1.0
        or int(packet_bits) < 1
    ):
        raise ValueError("invalid exact-reference parameters")
    random = np.asarray(random_values, dtype=np.float64)
    receiver_actions: tuple[int, ...] | None = None
    available_rows: list[bool] = []
    clean_rows: list[bool] = []
    effective_regret: list[float] = []
    available_regret: list[float] = []
    for local_position, index in enumerate(indices):
        row = random[local_position]
        if row[RESET_STREAM] < receiver_reset_probability:
            receiver_actions = None
        delivered = any(
            row[stream] >= packet_loss_probability
            for stream in TASK_ATTEMPT_STREAMS[:task_open_loop_attempts]
        )
        state = states[int(index)]
        if delivered:
            receiver_actions = state.optimal_actions
        available = receiver_actions is not None
        available_rows.append(available)
        if available:
            regret = float(state.max_regret_db(receiver_actions))
            available_regret.append(regret)
            effective_regret.append(regret)
            clean_rows.append(regret <= float(epsilon_db) + 1e-12)
        else:
            effective_regret.append(float(outage_penalty_db))
            clean_rows.append(False)

    exact_bits = int(packet_bits) * int(task_open_loop_attempts) * indices.size
    breakdown = _finish_breakdown(
        task_frame_bits=exact_bits,
        exact_action_frame_bits=exact_bits,
        escape_exact_bits=0,
        initial_install_bits=0,
        recovery_install_bits=0,
        compact_update_bits=0,
        duplicate_update_bits=0,
        heartbeat_request_bits=0,
        ack_bits=0,
        heartbeat_response_bits=0,
        reserved_capacity_bits=0,
        used_reserved_capacity_bits=0,
    )
    return MatchedTrajectoryResult(
        bit_breakdown=breakdown,
        context_install_count=0,
        compact_update_count=indices.size,
        duplicate_update_count=(
            indices.size * (int(task_open_loop_attempts) - 1)
        ),
        update_reservation_count=0,
        protected_update_count=0,
        ack_frame_count=0,
        stale_ack_rejection_count=0,
        rejected_compact_count=0,
        wrong_codebook_decode_count=0,
        maximum_belief_size=0,
        heartbeat_probe_count=0,
        heartbeat_response_count=0,
        heartbeat_failure_count=0,
        available=tuple(available_rows),
        clean=tuple(clean_rows),
        effective_regret_db=tuple(effective_regret),
        available_regret_db=tuple(available_regret),
    )


def simulate_semantic_trajectory(
    states: list[Any],
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    sender_session: Any,
    receiver_session: Any,
    install_packet: np.ndarray,
    deployment_mode: str,
    condition: dict[str, Any],
    random_values: np.ndarray,
    epsilon_db: float,
    max_age_minutes: float,
    ack_frame_bits: int,
    outage_penalty_db: float,
    heartbeat_interval_scenes: int | None,
    heartbeat_request_frame_bits: int,
    heartbeat_response_frame_bits: int,
    task_open_loop_attempts: int,
    compact_codeword_frame_bits: int,
    update_protection_candidates: np.ndarray,
    maximum_update_reservations: int,
    reservation_equivalent_bits: int,
) -> MatchedTrajectoryResult:
    """Simulate the frozen semantic controller under a fair workpoint."""

    indices = _validate_common_inputs(
        states,
        timestamps,
        evaluation_indices,
        random_values,
        task_open_loop_attempts=task_open_loop_attempts,
    )
    if deployment_mode not in DEPLOYMENT_MODES:
        raise ValueError("unknown deployment mode")
    if heartbeat_interval_scenes is not None and heartbeat_interval_scenes < 1:
        raise ValueError("heartbeat interval must be positive")
    protection = np.asarray(update_protection_candidates, dtype=bool)
    if (
        protection.shape != (len(states),)
        or int(maximum_update_reservations) < 0
        or int(reservation_equivalent_bits) < 1
        or int(compact_codeword_frame_bits) < 1
    ):
        raise ValueError("invalid protection configuration")
    probabilities = {
        key: float(condition[key])
        for key in (
            "install_loss_probability",
            "task_loss_probability",
            "ack_loss_probability",
            "delayed_duplicate_probability",
            "receiver_context_reset_probability",
        )
    }
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("invalid fault probability")
    include_reset = bool(
        condition["include_reset_hypothesis_after_missing_ack"]
    )
    random = np.asarray(random_values, dtype=np.float64)

    if deployment_mode == "preconfigured_codebook_empty_action_state":
        receiver = _ReceiverState(
            context_installed=True,
            codebook_epoch=int(sender_session.epoch),
        )
        belief = _preconfigured_belief(sender_session.epoch)
    else:
        receiver = _ReceiverState()
        belief = initial_context_belief()

    confirmed_update_epoch: int | None = None
    last_sent_epoch = 0
    last_delivered_ack_epoch: int | None = None
    initial_online_install_pending = (
        deployment_mode == "online_install_empty_context"
    )
    scenes_since_successful_feedback = 0

    task_frame_bits = 0
    compact_update_bits = 0
    escape_exact_bits = 0
    duplicate_update_bits = 0
    initial_install_bits = 0
    recovery_install_bits = 0
    heartbeat_request_bits = 0
    ack_bits = 0
    heartbeat_response_bits = 0
    reserved_capacity_bits = 0
    used_reserved_capacity_bits = 0
    install_count = 0
    update_count = 0
    duplicate_update_count = 0
    reservation_count = 0
    protected_update_count = 0
    ack_count = 0
    stale_rejections = 0
    rejected_compact = 0
    wrong_decode = 0
    heartbeat_probe_count = 0
    heartbeat_response_count = 0
    heartbeat_failure_count = 0
    maximum_belief_size = len(belief)
    available_rows: list[bool] = []
    clean_rows: list[bool] = []
    effective_regret: list[float] = []
    available_regret: list[float] = []

    for local_position, index in enumerate(indices):
        state = states[int(index)]
        timestamp = str(timestamps[int(index)])
        row = random[local_position]
        feedback_received = False
        if (
            row[RESET_STREAM]
            < probabilities["receiver_context_reset_probability"]
        ):
            receiver = _ReceiverState()

        if (
            heartbeat_interval_scenes is not None
            and scenes_since_successful_feedback
            >= int(heartbeat_interval_scenes)
            and not belief_requires_context_install(
                belief, codebook_epoch=sender_session.epoch
            )
        ):
            request = encode_context_probe_request(
                node_id=1,
                codebook_epoch=sender_session.epoch,
                expected_update_epoch=last_sent_epoch,
            )
            if int(request.size) != int(heartbeat_request_frame_bits):
                raise ValueError("heartbeat request accounting mismatch")
            heartbeat_request_bits += int(request.size)
            heartbeat_probe_count += 1
            response_delivered = False
            if (
                row[HEARTBEAT_REQUEST_STREAM]
                >= probabilities["task_loss_probability"]
            ):
                decoded_request = decode_context_probe_request(request)
                if (
                    decoded_request.node_id == 1
                    and decoded_request.codebook_epoch
                    == receiver.codebook_epoch
                    and receiver.context_installed
                    and receiver.actions is not None
                    and receiver.update_epoch is not None
                ):
                    response = encode_context_probe_response(
                        node_id=1,
                        codebook_epoch=int(receiver.codebook_epoch),
                        current_update_epoch=int(receiver.update_epoch),
                    )
                    if int(response.size) != int(
                        heartbeat_response_frame_bits
                    ):
                        raise ValueError(
                            "heartbeat response accounting mismatch"
                        )
                    heartbeat_response_bits += int(response.size)
                    heartbeat_response_count += 1
                    response_delivered = (
                        row[HEARTBEAT_RESPONSE_STREAM]
                        >= probabilities["ack_loss_probability"]
                    )
                    if response_delivered:
                        decoded_response = decode_context_probe_response(
                            response
                        )
                        belief, confirmed_update_epoch, accepted = (
                            condition_belief_on_cumulative_ack(
                                belief,
                                ack_epoch=(
                                    decoded_response.current_update_epoch
                                ),
                                confirmed_epoch=confirmed_update_epoch,
                                guard_stale_epochs=False,
                            )
                        )
                        if not accepted:
                            raise ValueError(
                                "heartbeat confirmed impossible belief"
                            )
                        feedback_received = True
            if not response_delivered:
                belief = frozenset(
                    set(belief) | set(initial_context_belief())
                )
                heartbeat_failure_count += 1

        update_reserved = (
            bool(protection[int(index)])
            and reservation_count < int(maximum_update_reservations)
        )
        if update_reserved:
            reservation_count += 1
            reserved_capacity_bits += int(reservation_equivalent_bits)

        needs_install = belief_requires_context_install(
            belief, codebook_epoch=sender_session.epoch
        )
        if needs_install:
            install_size = int(install_packet.size)
            install_count += 1
            if initial_online_install_pending:
                initial_install_bits += install_size
                initial_online_install_pending = False
            else:
                recovery_install_bits += install_size
            install_delivered = (
                row[INSTALL_STREAM]
                >= probabilities["install_loss_probability"]
            )
            install_ack_delivered = False
            if install_delivered:
                decoded_install = decode_context_install(install_packet)
                same_context = (
                    receiver.context_installed
                    and receiver.codebook_epoch
                    == decoded_install.session.epoch
                )
                receiver.context_installed = True
                receiver.codebook_epoch = decoded_install.session.epoch
                if not same_context:
                    receiver.actions = None
                    receiver.update_epoch = None
                    receiver.success_timestamp = None
                ack_bits += int(ack_frame_bits)
                ack_count += 1
                install_ack_delivered = (
                    row[INSTALL_ACK_STREAM]
                    >= probabilities["ack_loss_probability"]
                )
                feedback_received = (
                    feedback_received or install_ack_delivered
                )
            belief = context_install_attempt(
                belief,
                codebook_epoch=sender_session.epoch,
                ack_received=install_ack_delivered,
            )

        worst_regret = belief_worst_case_regret_db(
            state,
            belief,
            codebook_epoch=sender_session.epoch,
        )
        worst_age = belief_maximum_age_minutes(
            belief,
            timestamp_local=timestamp,
            codebook_epoch=sender_session.epoch,
        )
        should_update = (
            worst_regret > float(epsilon_db) + 1e-12
            or worst_age > float(max_age_minutes)
        )

        current_ack_delivered = False
        current_ack_epoch: int | None = None
        if should_update:
            last_sent_epoch = (last_sent_epoch + 1) % 256
            update_packet, decision = encode_compact_update(
                state,
                sender_session,
                node_id=1,
                update_epoch=last_sent_epoch,
            )
            packet_size = int(update_packet.size)
            if packet_size < int(compact_codeword_frame_bits):
                raise ValueError("compact packet is shorter than base frame")
            base_escape_bits = packet_size - int(compact_codeword_frame_bits)
            compact_update_bits += int(compact_codeword_frame_bits)
            escape_exact_bits += base_escape_bits
            base_duplicates = int(task_open_loop_attempts) - 1
            duplicate_update_bits += base_duplicates * packet_size
            duplicate_update_count += base_duplicates
            update_count += 1
            delivered = any(
                row[stream] >= probabilities["task_loss_probability"]
                for stream in TASK_ATTEMPT_STREAMS[
                    : int(task_open_loop_attempts)
                ]
            )
            if update_reserved:
                duplicate_update_bits += packet_size
                duplicate_update_count += 1
                protected_update_count += 1
                used_reserved_capacity_bits += int(
                    reservation_equivalent_bits
                )
                delivered = delivered or (
                    row[PROTECTION_ATTEMPT_STREAM]
                    >= probabilities["task_loss_probability"]
                )

            accepted_update = False
            if delivered:
                if (
                    receiver.context_installed
                    and receiver.codebook_epoch == sender_session.epoch
                ):
                    decoded = decode_compact_update(
                        update_packet, receiver_session
                    )
                    if decoded.decoder_actions != decision.decoder_actions:
                        wrong_decode += 1
                    else:
                        receiver.actions = decoded.decoder_actions
                        receiver.update_epoch = decoded.update_epoch
                        receiver.success_timestamp = timestamp
                        accepted_update = True
                else:
                    rejected_compact += 1
            if accepted_update:
                ack_frame = encode_cumulative_ack(
                    node_id=1,
                    epoch=last_sent_epoch,
                )
                ack = decode_cumulative_ack(ack_frame)
                if int(ack_frame.size) != int(ack_frame_bits):
                    raise ValueError("ACK accounting mismatch")
                ack_bits += int(ack_frame.size)
                ack_count += 1
                current_ack_epoch = int(ack.epoch)
                current_ack_delivered = (
                    row[UPDATE_ACK_STREAM]
                    >= probabilities["ack_loss_probability"]
                )
                feedback_received = (
                    feedback_received or current_ack_delivered
                )

            belief = compact_update_attempt(
                belief,
                codebook_epoch=sender_session.epoch,
                update_epoch=last_sent_epoch,
                decoder_actions=decision.decoder_actions,
                success_timestamp=timestamp,
                ack_received=False,
                include_context_loss_on_no_ack=(
                    include_reset and not current_ack_delivered
                ),
            )
            if current_ack_delivered and current_ack_epoch is not None:
                belief, confirmed_update_epoch, accepted = (
                    condition_belief_on_cumulative_ack(
                        belief,
                        ack_epoch=current_ack_epoch,
                        confirmed_epoch=confirmed_update_epoch,
                        guard_stale_epochs=True,
                    )
                )
                if not accepted:
                    stale_rejections += 1

        duplicate_epoch = last_delivered_ack_epoch
        if (
            duplicate_epoch is not None
            and row[DELAYED_ACK_STREAM]
            < probabilities["delayed_duplicate_probability"]
        ):
            ack_bits += int(ack_frame_bits)
            ack_count += 1
            belief, confirmed_update_epoch, accepted = (
                condition_belief_on_cumulative_ack(
                    belief,
                    ack_epoch=duplicate_epoch,
                    confirmed_epoch=confirmed_update_epoch,
                    guard_stale_epochs=True,
                )
            )
            if not accepted:
                stale_rejections += 1
        if current_ack_delivered and current_ack_epoch is not None:
            last_delivered_ack_epoch = current_ack_epoch

        if feedback_received:
            scenes_since_successful_feedback = 0
        else:
            scenes_since_successful_feedback += 1
        maximum_belief_size = max(maximum_belief_size, len(belief))
        available = receiver.actions is not None
        available_rows.append(available)
        if available:
            regret = float(state.max_regret_db(receiver.actions))
            available_regret.append(regret)
            effective_regret.append(regret)
            clean_rows.append(regret <= float(epsilon_db) + 1e-12)
        else:
            effective_regret.append(float(outage_penalty_db))
            clean_rows.append(False)

    task_frame_bits = (
        compact_update_bits + escape_exact_bits + duplicate_update_bits
    )
    breakdown = _finish_breakdown(
        task_frame_bits=task_frame_bits,
        exact_action_frame_bits=0,
        escape_exact_bits=escape_exact_bits,
        initial_install_bits=initial_install_bits,
        recovery_install_bits=recovery_install_bits,
        compact_update_bits=compact_update_bits,
        duplicate_update_bits=duplicate_update_bits,
        heartbeat_request_bits=heartbeat_request_bits,
        ack_bits=ack_bits,
        heartbeat_response_bits=heartbeat_response_bits,
        reserved_capacity_bits=reserved_capacity_bits,
        used_reserved_capacity_bits=used_reserved_capacity_bits,
    )
    return MatchedTrajectoryResult(
        bit_breakdown=breakdown,
        context_install_count=install_count,
        compact_update_count=update_count,
        duplicate_update_count=duplicate_update_count,
        update_reservation_count=reservation_count,
        protected_update_count=protected_update_count,
        ack_frame_count=ack_count,
        stale_ack_rejection_count=stale_rejections,
        rejected_compact_count=rejected_compact,
        wrong_codebook_decode_count=wrong_decode,
        maximum_belief_size=maximum_belief_size,
        heartbeat_probe_count=heartbeat_probe_count,
        heartbeat_response_count=heartbeat_response_count,
        heartbeat_failure_count=heartbeat_failure_count,
        available=tuple(available_rows),
        clean=tuple(clean_rows),
        effective_regret_db=tuple(effective_regret),
        available_regret_db=tuple(available_regret),
    )
