"""Development-only Stage-6 R1 semi-stateful reliability protocols.

The frozen S6-FC0 implementation is not modified.  This module isolates three
post-negative-result candidates:

* durable-codebook compact refresh;
* compact updates with context-independent exact-action recovery;
* event-triggered exact-action caching without a semantic codebook.
"""

from __future__ import annotations

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
from spectrum_semcom.stage6_matched_reliability import (
    DELAYED_ACK_STREAM,
    HEARTBEAT_REQUEST_STREAM,
    HEARTBEAT_RESPONSE_STREAM,
    INSTALL_ACK_STREAM,
    INSTALL_STREAM,
    RANDOM_STREAM_COUNT,
    RESET_STREAM,
    TASK_ATTEMPT_STREAMS,
    UPDATE_ACK_STREAM,
)


R1_VARIANTS = (
    "v2_durable_codebook_compact_refresh",
    "v3_context_independent_exact_refresh",
    "v4_event_triggered_exact_action_cache",
)


@dataclass(frozen=True)
class R1ContextHypothesis:
    codebook_epoch: int | None
    actions: tuple[int, ...] | None
    update_epoch: int | None
    success_timestamp: str | None

    def __post_init__(self) -> None:
        if self.codebook_epoch is not None and not 0 <= int(
            self.codebook_epoch
        ) <= 255:
            raise ValueError("codebook epoch must fit eight bits")
        if self.update_epoch is not None and not 0 <= int(
            self.update_epoch
        ) <= 255:
            raise ValueError("update epoch must fit eight bits")
        if (self.actions is None) != (self.success_timestamp is None):
            raise ValueError("actions and timestamp must coexist")
        if self.success_timestamp is not None:
            datetime.fromisoformat(self.success_timestamp)


@dataclass
class _R1Receiver:
    codebook_epoch: int | None = None
    actions: tuple[int, ...] | None = None
    update_epoch: int | None = None
    success_timestamp: str | None = None


@dataclass(frozen=True)
class R1TrajectoryResult:
    bit_breakdown: Stage6BitBreakdown
    initial_install_count: int
    recovery_install_count: int
    compact_refresh_count: int
    exact_refresh_count: int
    duplicate_update_count: int
    ack_frame_count: int
    heartbeat_probe_count: int
    heartbeat_response_count: int
    heartbeat_failure_count: int
    actual_action_reset_count: int
    actual_codebook_reset_count: int
    missing_update_ack_uncertainty_count: int
    failed_heartbeat_uncertainty_count: int
    rejected_compact_count: int
    wrong_codebook_decode_count: int
    maximum_belief_size: int
    available: tuple[bool, ...]
    clean: tuple[bool, ...]
    effective_regret_db: tuple[float, ...]
    available_regret_db: tuple[float, ...]


def _empty_hypothesis(
    variant: str,
    codebook_epoch: int,
) -> R1ContextHypothesis:
    return R1ContextHypothesis(
        codebook_epoch=(
            int(codebook_epoch)
            if variant == "v2_durable_codebook_compact_refresh"
            else None
        ),
        actions=None,
        update_epoch=None,
        success_timestamp=None,
    )


def _age_minutes(previous: str | None, current: str) -> float:
    if previous is None:
        return float("inf")
    return (
        datetime.fromisoformat(current) - datetime.fromisoformat(previous)
    ).total_seconds() / 60.0


def _belief_worst_regret(
    state: Any,
    belief: frozenset[R1ContextHypothesis],
) -> float:
    values = []
    for hypothesis in belief:
        if hypothesis.actions is None:
            return float("inf")
        values.append(float(state.max_regret_db(hypothesis.actions)))
    return float(max(values, default=float("inf")))


def _belief_worst_age(
    belief: frozenset[R1ContextHypothesis],
    timestamp: str,
) -> float:
    return float(
        max(
            (
                _age_minutes(hypothesis.success_timestamp, timestamp)
                for hypothesis in belief
            ),
            default=float("inf"),
        )
    )


def _all_have_codebook(
    belief: frozenset[R1ContextHypothesis],
    codebook_epoch: int,
) -> bool:
    return bool(belief) and all(
        hypothesis.codebook_epoch == int(codebook_epoch)
        for hypothesis in belief
    )


def _finish_breakdown(
    *,
    exact_action_frame_bits: int,
    escape_exact_bits: int,
    initial_install_bits: int,
    recovery_install_bits: int,
    compact_update_bits: int,
    duplicate_update_bits: int,
    heartbeat_request_bits: int,
    ack_bits: int,
    heartbeat_response_bits: int,
) -> Stage6BitBreakdown:
    task_frame_bits = (
        int(exact_action_frame_bits)
        + int(escape_exact_bits)
        + int(compact_update_bits)
        + int(duplicate_update_bits)
    )
    forward = (
        task_frame_bits
        + int(initial_install_bits)
        + int(recovery_install_bits)
        + int(heartbeat_request_bits)
    )
    feedback = int(ack_bits) + int(heartbeat_response_bits)
    total = forward + feedback
    value = Stage6BitBreakdown(
        task_frame_bits=task_frame_bits,
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
        actual_forward_bits=forward,
        actual_feedback_bits=feedback,
        actual_total_application_bits=total,
        reserved_capacity_bits=0,
        used_reserved_capacity_bits=0,
        unused_reserved_capacity_bits=0,
        resource_equivalent_bits=total,
    )
    validate_breakdown_identity(value)
    return value


def combine_r1_trajectory_results(
    values: list[R1TrajectoryResult],
) -> R1TrajectoryResult:
    if not values:
        raise ValueError("cannot combine empty R1 results")
    bit_fields = tuple(values[0].bit_breakdown.to_dict())
    breakdown = Stage6BitBreakdown(
        **{
            field: sum(
                int(value.bit_breakdown.to_dict()[field])
                for value in values
            )
            for field in bit_fields
        }
    )
    validate_breakdown_identity(breakdown)
    count_fields = (
        "initial_install_count",
        "recovery_install_count",
        "compact_refresh_count",
        "exact_refresh_count",
        "duplicate_update_count",
        "ack_frame_count",
        "heartbeat_probe_count",
        "heartbeat_response_count",
        "heartbeat_failure_count",
        "actual_action_reset_count",
        "actual_codebook_reset_count",
        "missing_update_ack_uncertainty_count",
        "failed_heartbeat_uncertainty_count",
        "rejected_compact_count",
        "wrong_codebook_decode_count",
    )
    counts = {
        field: sum(int(getattr(value, field)) for value in values)
        for field in count_fields
    }
    return R1TrajectoryResult(
        bit_breakdown=breakdown,
        **counts,
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


def simulate_r1_trajectory(
    states: list[Any],
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    variant: str,
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
    exact_action_frame_bits: int,
) -> R1TrajectoryResult:
    """Simulate one paired R1 link trajectory."""

    if variant not in R1_VARIANTS:
        raise ValueError("unknown R1 variant")
    indices = np.asarray(evaluation_indices, dtype=np.int64).reshape(-1)
    random = np.asarray(random_values, dtype=np.float64)
    if (
        indices.size < 1
        or random.shape != (indices.size, RANDOM_STREAM_COUNT)
        or np.any(indices < 0)
        or np.any(indices >= len(states))
        or not 1 <= int(task_open_loop_attempts) <= len(
            TASK_ATTEMPT_STREAMS
        )
        or int(exact_action_frame_bits) < 1
        or int(compact_codeword_frame_bits) < 1
        or int(ack_frame_bits) < 1
    ):
        raise ValueError("invalid R1 trajectory inputs")
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
        raise ValueError("invalid R1 fault probability")

    codebook_epoch = int(sender_session.epoch)
    preconfigured = (
        deployment_mode == "preconfigured_codebook_empty_action_state"
    )
    if deployment_mode not in (
        "preconfigured_codebook_empty_action_state",
        "online_install_empty_context",
    ):
        raise ValueError("unknown deployment mode")
    initial_codebook = (
        codebook_epoch
        if preconfigured
        and variant != "v4_event_triggered_exact_action_cache"
        else None
    )
    receiver = _R1Receiver(codebook_epoch=initial_codebook)
    belief = frozenset(
        {
            R1ContextHypothesis(
                codebook_epoch=initial_codebook,
                actions=None,
                update_epoch=None,
                success_timestamp=None,
            )
        }
    )
    initial_install_pending = (
        not preconfigured
        and variant == "v2_durable_codebook_compact_refresh"
    )
    last_sent_epoch = 0
    last_delivered_ack_epoch: int | None = None
    scenes_since_feedback = 0

    exact_bits = 0
    escape_bits = 0
    initial_install_bits = 0
    recovery_install_bits = 0
    compact_bits = 0
    duplicate_bits = 0
    heartbeat_request_bits = 0
    heartbeat_response_bits = 0
    ack_bits = 0
    initial_install_count = 0
    recovery_install_count = 0
    compact_count = 0
    exact_count = 0
    duplicate_count = 0
    ack_count = 0
    probe_count = 0
    response_count = 0
    heartbeat_failure_count = 0
    action_reset_count = 0
    codebook_reset_count = 0
    missing_ack_uncertainty_count = 0
    heartbeat_uncertainty_count = 0
    rejected_compact_count = 0
    wrong_decode_count = 0
    maximum_belief_size = 1
    available_rows: list[bool] = []
    clean_rows: list[bool] = []
    effective_regret: list[float] = []
    available_regret: list[float] = []

    for local_position, index in enumerate(indices):
        state = states[int(index)]
        timestamp = str(timestamps[int(index)])
        row = random[local_position]
        feedback_received = False
        if row[RESET_STREAM] < probabilities[
            "receiver_context_reset_probability"
        ]:
            receiver.actions = None
            receiver.update_epoch = None
            receiver.success_timestamp = None
            action_reset_count += 1
            if variant == "v3_context_independent_exact_refresh":
                receiver.codebook_epoch = None
                codebook_reset_count += 1

        if (
            heartbeat_interval_scenes is not None
            and scenes_since_feedback >= int(heartbeat_interval_scenes)
        ):
            heartbeat_request_bits += int(heartbeat_request_frame_bits)
            probe_count += 1
            response_delivered = False
            if (
                row[HEARTBEAT_REQUEST_STREAM]
                >= probabilities["task_loss_probability"]
                and receiver.actions is not None
                and receiver.update_epoch is not None
            ):
                heartbeat_response_bits += int(
                    heartbeat_response_frame_bits
                )
                response_count += 1
                response_delivered = (
                    row[HEARTBEAT_RESPONSE_STREAM]
                    >= probabilities["ack_loss_probability"]
                )
                if response_delivered:
                    compatible = frozenset(
                        hypothesis
                        for hypothesis in belief
                        if hypothesis.actions is not None
                        and hypothesis.update_epoch
                        == receiver.update_epoch
                        and (
                            variant
                            == "v4_event_triggered_exact_action_cache"
                            or hypothesis.codebook_epoch
                            == receiver.codebook_epoch
                        )
                    )
                    if not compatible:
                        compatible = frozenset(
                            {
                                R1ContextHypothesis(
                                    codebook_epoch=(
                                        None
                                        if variant
                                        == "v4_event_triggered_exact_action_cache"
                                        else receiver.codebook_epoch
                                    ),
                                    actions=receiver.actions,
                                    update_epoch=receiver.update_epoch,
                                    success_timestamp=(
                                        receiver.success_timestamp
                                    ),
                                )
                            }
                        )
                    belief = compatible
                    feedback_received = True
            if not response_delivered:
                belief = frozenset(
                    set(belief)
                    | {
                        _empty_hypothesis(variant, codebook_epoch)
                    }
                )
                heartbeat_failure_count += 1
                heartbeat_uncertainty_count += 1

        needs_install = (
            variant == "v2_durable_codebook_compact_refresh"
            and not _all_have_codebook(belief, codebook_epoch)
        )
        if needs_install:
            install_size = int(install_packet.size)
            if initial_install_pending:
                initial_install_bits += install_size
                initial_install_count += 1
                initial_install_pending = False
            else:
                recovery_install_bits += install_size
                recovery_install_count += 1
            delivered = (
                row[INSTALL_STREAM]
                >= probabilities["install_loss_probability"]
            )
            ack_delivered = False
            delivered_hypotheses = set()
            if delivered:
                decoded = decode_context_install(install_packet)
                same = receiver.codebook_epoch == decoded.session.epoch
                receiver.codebook_epoch = decoded.session.epoch
                if not same:
                    receiver.actions = None
                    receiver.update_epoch = None
                    receiver.success_timestamp = None
                ack_bits += int(ack_frame_bits)
                ack_count += 1
                ack_delivered = (
                    row[INSTALL_ACK_STREAM]
                    >= probabilities["ack_loss_probability"]
                )
                for hypothesis in belief:
                    same_hypothesis = (
                        hypothesis.codebook_epoch == codebook_epoch
                    )
                    delivered_hypotheses.add(
                        R1ContextHypothesis(
                            codebook_epoch=codebook_epoch,
                            actions=(
                                hypothesis.actions
                                if same_hypothesis
                                else None
                            ),
                            update_epoch=(
                                hypothesis.update_epoch
                                if same_hypothesis
                                else None
                            ),
                            success_timestamp=(
                                hypothesis.success_timestamp
                                if same_hypothesis
                                else None
                            ),
                        )
                    )
            if ack_delivered:
                belief = frozenset(delivered_hypotheses)
                feedback_received = True
            else:
                belief = frozenset(
                    set(belief) | delivered_hypotheses
                )

        worst_regret = _belief_worst_regret(state, belief)
        worst_age = _belief_worst_age(belief, timestamp)
        should_update = (
            worst_regret > float(epsilon_db) + 1e-12
            or worst_age > float(max_age_minutes)
        )
        if (
            variant == "v2_durable_codebook_compact_refresh"
            and not _all_have_codebook(belief, codebook_epoch)
        ):
            should_update = False
        current_ack_delivered = False
        current_ack_epoch: int | None = None
        if should_update:
            last_sent_epoch = (last_sent_epoch + 1) % 256
            use_compact = (
                variant != "v4_event_triggered_exact_action_cache"
                and _all_have_codebook(belief, codebook_epoch)
            )
            decoder_actions = tuple(state.optimal_actions)
            packet_size = int(exact_action_frame_bits)
            base_escape = 0
            if use_compact:
                packet, decision = encode_compact_update(
                    state,
                    sender_session,
                    node_id=1,
                    update_epoch=last_sent_epoch,
                )
                packet_size = int(packet.size)
                base_escape = (
                    packet_size - int(compact_codeword_frame_bits)
                )
                compact_bits += int(compact_codeword_frame_bits)
                escape_bits += int(base_escape)
                compact_count += 1
                decoder_actions = tuple(decision.decoder_actions)
            else:
                exact_bits += int(exact_action_frame_bits)
                exact_count += 1
            additional_attempts = int(task_open_loop_attempts) - 1
            duplicate_bits += additional_attempts * packet_size
            duplicate_count += additional_attempts
            delivered = any(
                row[stream] >= probabilities["task_loss_probability"]
                for stream in TASK_ATTEMPT_STREAMS[
                    : int(task_open_loop_attempts)
                ]
            )
            accepted = False
            if delivered:
                if use_compact:
                    if receiver.codebook_epoch == codebook_epoch:
                        decoded = decode_compact_update(
                            packet, receiver_session
                        )
                        if (
                            tuple(decoded.decoder_actions)
                            != decoder_actions
                        ):
                            wrong_decode_count += 1
                        else:
                            receiver.actions = decoder_actions
                            receiver.update_epoch = int(
                                decoded.update_epoch
                            )
                            receiver.success_timestamp = timestamp
                            accepted = True
                    else:
                        rejected_compact_count += 1
                else:
                    receiver.actions = decoder_actions
                    receiver.update_epoch = last_sent_epoch
                    receiver.success_timestamp = timestamp
                    accepted = True

            delivered_hypotheses = set()
            if use_compact:
                for hypothesis in belief:
                    if hypothesis.codebook_epoch == codebook_epoch:
                        delivered_hypotheses.add(
                            R1ContextHypothesis(
                                codebook_epoch=codebook_epoch,
                                actions=decoder_actions,
                                update_epoch=last_sent_epoch,
                                success_timestamp=timestamp,
                            )
                        )
            else:
                for hypothesis in belief:
                    delivered_hypotheses.add(
                        R1ContextHypothesis(
                            codebook_epoch=(
                                None
                                if variant
                                == "v4_event_triggered_exact_action_cache"
                                else hypothesis.codebook_epoch
                            ),
                            actions=decoder_actions,
                            update_epoch=last_sent_epoch,
                            success_timestamp=timestamp,
                        )
                    )
            if accepted:
                ack_frame = encode_cumulative_ack(
                    node_id=1, epoch=last_sent_epoch
                )
                ack = decode_cumulative_ack(ack_frame)
                if int(ack_frame.size) != int(ack_frame_bits):
                    raise ValueError("R1 ACK accounting mismatch")
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
            if current_ack_delivered:
                compatible = frozenset(
                    hypothesis
                    for hypothesis in delivered_hypotheses
                    if hypothesis.update_epoch == current_ack_epoch
                )
                if not compatible:
                    raise ValueError("R1 ACK confirmed no hypothesis")
                belief = compatible
            else:
                belief = frozenset(
                    set(belief)
                    | delivered_hypotheses
                    | {_empty_hypothesis(variant, codebook_epoch)}
                )
                missing_ack_uncertainty_count += 1

        if (
            last_delivered_ack_epoch is not None
            and row[DELAYED_ACK_STREAM]
            < probabilities["delayed_duplicate_probability"]
        ):
            ack_bits += int(ack_frame_bits)
            ack_count += 1
        if current_ack_delivered and current_ack_epoch is not None:
            last_delivered_ack_epoch = current_ack_epoch

        scenes_since_feedback = (
            0 if feedback_received else scenes_since_feedback + 1
        )
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

    breakdown = _finish_breakdown(
        exact_action_frame_bits=exact_bits,
        escape_exact_bits=escape_bits,
        initial_install_bits=initial_install_bits,
        recovery_install_bits=recovery_install_bits,
        compact_update_bits=compact_bits,
        duplicate_update_bits=duplicate_bits,
        heartbeat_request_bits=heartbeat_request_bits,
        ack_bits=ack_bits,
        heartbeat_response_bits=heartbeat_response_bits,
    )
    return R1TrajectoryResult(
        bit_breakdown=breakdown,
        initial_install_count=initial_install_count,
        recovery_install_count=recovery_install_count,
        compact_refresh_count=compact_count,
        exact_refresh_count=exact_count,
        duplicate_update_count=duplicate_count,
        ack_frame_count=ack_count,
        heartbeat_probe_count=probe_count,
        heartbeat_response_count=response_count,
        heartbeat_failure_count=heartbeat_failure_count,
        actual_action_reset_count=action_reset_count,
        actual_codebook_reset_count=codebook_reset_count,
        missing_update_ack_uncertainty_count=(
            missing_ack_uncertainty_count
        ),
        failed_heartbeat_uncertainty_count=heartbeat_uncertainty_count,
        rejected_compact_count=rejected_compact_count,
        wrong_codebook_decode_count=wrong_decode_count,
        maximum_belief_size=maximum_belief_size,
        available=tuple(available_rows),
        clean=tuple(clean_rows),
        effective_regret_db=tuple(effective_regret),
        available_regret_db=tuple(available_regret),
    )
