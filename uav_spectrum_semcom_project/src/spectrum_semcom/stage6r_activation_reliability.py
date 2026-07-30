"""Full reliability simulator with Stage-6R preinstalled activation.

The frozen Stage-6 simulator remains unchanged.  This sibling implementation
adds a versioned preinstalled-codebook activation path and a mandatory full
manifest fallback after a bounded number of unconfirmed activation attempts.
Risk-reservation protection is intentionally outside this v1 path because the
selected clean-0.90 workpoints do not use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectrum_semcom.stage5_cumulative_ack import (
    decode_cumulative_ack,
    encode_cumulative_ack,
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
    belief_maximum_age_minutes,
    belief_requires_context_install,
    belief_worst_case_regret_db,
    compact_update_attempt,
    condition_belief_on_cumulative_ack,
    context_install_attempt,
    initial_context_belief,
)
from spectrum_semcom.stage6_matched_reliability import (
    DELAYED_ACK_STREAM,
    HEARTBEAT_REQUEST_STREAM,
    HEARTBEAT_RESPONSE_STREAM,
    INSTALL_ACK_STREAM,
    INSTALL_STREAM,
    RESET_STREAM,
    TASK_ATTEMPT_STREAMS,
    UPDATE_ACK_STREAM,
    MatchedTrajectoryResult,
    _finish_breakdown,
    _ReceiverState,
    _validate_common_inputs,
)
from spectrum_semcom.stage6r_codebook_activation import (
    PreinstalledCodebookCatalog,
    decode_codebook_activation,
    encode_codebook_activation,
)


@dataclass(frozen=True)
class ActivationMatchedTrajectoryResult:
    matched: MatchedTrajectoryResult
    activation_attempt_count: int
    activation_rejection_count: int
    activation_ack_count: int
    activation_bits: int
    initial_activation_bits: int
    recovery_activation_bits: int
    full_install_attempt_count: int
    full_install_fallback_count: int
    full_initial_install_bits: int
    full_recovery_install_bits: int
    forced_update_request_count: int
    forced_update_redundant_count: int


def simulate_activation_semantic_trajectory(
    states: list[Any],
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    sender_session: Any,
    receiver_session: Any,
    full_install_packet: np.ndarray,
    sender_catalog: PreinstalledCodebookCatalog | None,
    receiver_catalog: PreinstalledCodebookCatalog | None,
    bank_id: int | None,
    catalog_node_id: int,
    maximum_activation_attempts: int,
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
    heartbeat_trigger_mask: np.ndarray | None = None,
    forced_update_trigger_mask: np.ndarray | None = None,
) -> ActivationMatchedTrajectoryResult:
    """Simulate known-bank activation or OOD full-install fallback.

    ``heartbeat_trigger_mask`` is a Stage-7 research hook.  When omitted, the
    frozen Stage-6R fixed-interval behavior is unchanged.  When supplied, the
    aligned Boolean mask replaces the fixed interval and requests a heartbeat
    at selected evaluation scenes.  It does not bypass any context, identity,
    regret, escape, or fail-closed check.

    ``forced_update_trigger_mask`` is a separate Stage-7 diagnostic hook.  A
    true entry requests a normal compact semantic update even when the
    analytical regret and age triggers are not yet active.  The request still
    uses the current measured state and passes through the frozen packet,
    context, ACK, duplicate, escape, and bit-accounting paths.
    """

    indices = _validate_common_inputs(
        states,
        timestamps,
        evaluation_indices,
        random_values,
        task_open_loop_attempts=task_open_loop_attempts,
    )
    activation_eligible = (
        sender_catalog is not None
        and receiver_catalog is not None
        and bank_id is not None
    )
    if (
        (activation_eligible and int(maximum_activation_attempts) < 1)
        or heartbeat_interval_scenes is not None
        and int(heartbeat_interval_scenes) < 1
        or int(compact_codeword_frame_bits) < 1
    ):
        raise ValueError("invalid activation reliability configuration")
    heartbeat_triggers = (
        None
        if heartbeat_trigger_mask is None
        else np.asarray(heartbeat_trigger_mask, dtype=bool).reshape(-1)
    )
    forced_update_triggers = (
        None
        if forced_update_trigger_mask is None
        else np.asarray(forced_update_trigger_mask, dtype=bool).reshape(-1)
    )
    if (
        heartbeat_triggers is not None
        and heartbeat_triggers.shape != (indices.size,)
    ):
        raise ValueError("heartbeat trigger mask must align with evaluation indices")
    if (
        heartbeat_triggers is not None
        and heartbeat_interval_scenes is not None
    ):
        raise ValueError(
            "fixed heartbeat interval and trigger mask are mutually exclusive"
        )
    if (
        forced_update_triggers is not None
        and forced_update_triggers.shape != (indices.size,)
    ):
        raise ValueError(
            "forced update trigger mask must align with evaluation indices"
        )
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

    receiver = _ReceiverState()
    active_receiver_session = receiver_session
    belief = initial_context_belief()
    confirmed_update_epoch: int | None = None
    last_sent_epoch = 0
    last_delivered_ack_epoch: int | None = None
    scenes_since_successful_feedback = 0
    establishment_attempts = 0
    activation_epoch: int | None = None
    full_fallback_active = not activation_eligible
    first_context_control_pending = True

    compact_update_bits = 0
    escape_exact_bits = 0
    duplicate_update_bits = 0
    initial_context_bits = 0
    recovery_context_bits = 0
    heartbeat_request_bits = 0
    ack_bits = 0
    heartbeat_response_bits = 0
    context_attempt_count = 0
    update_count = 0
    duplicate_update_count = 0
    ack_count = 0
    stale_rejections = 0
    rejected_compact = 0
    wrong_decode = 0
    heartbeat_probe_count = 0
    heartbeat_response_count = 0
    heartbeat_failure_count = 0
    maximum_belief_size = len(belief)
    activation_attempt_count = 0
    activation_rejection_count = 0
    activation_ack_count = 0
    activation_bits = 0
    initial_activation_bits = 0
    recovery_activation_bits = 0
    full_install_attempt_count = 0
    full_install_fallback_count = 0
    full_initial_install_bits = 0
    full_recovery_install_bits = 0
    forced_update_request_count = 0
    forced_update_redundant_count = 0
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

        heartbeat_due = (
            bool(heartbeat_triggers[local_position])
            if heartbeat_triggers is not None
            else (
                heartbeat_interval_scenes is not None
                and scenes_since_successful_feedback
                >= int(heartbeat_interval_scenes)
            )
        )
        if (
            heartbeat_due
            and not belief_requires_context_install(
                belief, codebook_epoch=sender_session.epoch
            )
        ):
            request = encode_context_probe_request(
                node_id=catalog_node_id,
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
                    decoded_request.node_id == catalog_node_id
                    and decoded_request.codebook_epoch
                    == receiver.codebook_epoch
                    and receiver.context_installed
                    and receiver.actions is not None
                    and receiver.update_epoch is not None
                ):
                    response = encode_context_probe_response(
                        node_id=catalog_node_id,
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

        needs_install = belief_requires_context_install(
            belief, codebook_epoch=sender_session.epoch
        )
        if needs_install:
            context_attempt_count += 1
            attempt_is_initial = first_context_control_pending
            first_context_control_pending = False
            establishment_ack_delivered = False
            if activation_eligible and not full_fallback_active:
                if activation_epoch is None:
                    last_sent_epoch = (last_sent_epoch + 1) % 256
                    activation_epoch = last_sent_epoch
                activation_frame = encode_codebook_activation(
                    sender_catalog,
                    node_id=catalog_node_id,
                    bank_id=int(bank_id),
                    codebook_epoch=sender_session.epoch,
                    activation_epoch=activation_epoch,
                )
                frame_size = int(activation_frame.size)
                activation_bits += frame_size
                activation_attempt_count += 1
                establishment_attempts += 1
                if attempt_is_initial:
                    initial_context_bits += frame_size
                    initial_activation_bits += frame_size
                else:
                    recovery_context_bits += frame_size
                    recovery_activation_bits += frame_size
                activation_delivered = (
                    row[INSTALL_STREAM]
                    >= probabilities["install_loss_probability"]
                )
                activation_resolved = False
                if activation_delivered:
                    try:
                        decoded_activation = decode_codebook_activation(
                            activation_frame,
                            receiver_catalog,
                        )
                        activation_resolved = (
                            decoded_activation.session.manifest_sha256
                            == sender_session.manifest_sha256
                        )
                    except ValueError:
                        activation_resolved = False
                    if activation_resolved:
                        same_context = (
                            receiver.context_installed
                            and receiver.codebook_epoch
                            == sender_session.epoch
                        )
                        receiver.context_installed = True
                        receiver.codebook_epoch = sender_session.epoch
                        active_receiver_session = (
                            decoded_activation.session
                        )
                        if not same_context:
                            receiver.actions = None
                            receiver.update_epoch = None
                            receiver.success_timestamp = None
                        ack_frame = encode_cumulative_ack(
                            node_id=catalog_node_id,
                            epoch=int(activation_epoch),
                        )
                        if int(ack_frame.size) != int(ack_frame_bits):
                            raise ValueError(
                                "activation ACK accounting mismatch"
                            )
                        ack_bits += int(ack_frame.size)
                        ack_count += 1
                        activation_ack_count += 1
                        establishment_ack_delivered = (
                            row[INSTALL_ACK_STREAM]
                            >= probabilities["ack_loss_probability"]
                        )
                        feedback_received = (
                            feedback_received
                            or establishment_ack_delivered
                        )
                        if establishment_ack_delivered:
                            confirmed_update_epoch = activation_epoch
                    else:
                        activation_rejection_count += 1
                belief = context_install_attempt(
                    belief,
                    codebook_epoch=sender_session.epoch,
                    ack_received=establishment_ack_delivered,
                )
                if (
                    not establishment_ack_delivered
                    and establishment_attempts
                    >= int(maximum_activation_attempts)
                ):
                    full_fallback_active = True
                    full_install_fallback_count += 1
            else:
                install_size = int(full_install_packet.size)
                full_install_attempt_count += 1
                if attempt_is_initial:
                    initial_context_bits += install_size
                    full_initial_install_bits += install_size
                else:
                    recovery_context_bits += install_size
                    full_recovery_install_bits += install_size
                install_delivered = (
                    row[INSTALL_STREAM]
                    >= probabilities["install_loss_probability"]
                )
                if install_delivered:
                    decoded_install = decode_context_install(
                        full_install_packet
                    )
                    same_context = (
                        receiver.context_installed
                        and receiver.codebook_epoch
                        == decoded_install.session.epoch
                    )
                    receiver.context_installed = True
                    receiver.codebook_epoch = decoded_install.session.epoch
                    active_receiver_session = decoded_install.session
                    if not same_context:
                        receiver.actions = None
                        receiver.update_epoch = None
                        receiver.success_timestamp = None
                    ack_bits += int(ack_frame_bits)
                    ack_count += 1
                    establishment_ack_delivered = (
                        row[INSTALL_ACK_STREAM]
                        >= probabilities["ack_loss_probability"]
                    )
                    feedback_received = (
                        feedback_received
                        or establishment_ack_delivered
                    )
                belief = context_install_attempt(
                    belief,
                    codebook_epoch=sender_session.epoch,
                    ack_received=establishment_ack_delivered,
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
        analytic_update = (
            worst_regret > float(epsilon_db) + 1e-12
            or worst_age > float(max_age_minutes)
        )
        forced_update = (
            forced_update_triggers is not None
            and bool(forced_update_triggers[local_position])
        )
        if forced_update:
            forced_update_request_count += 1
            forced_update_redundant_count += int(analytic_update)
        should_update = analytic_update or forced_update

        current_ack_delivered = False
        current_ack_epoch: int | None = None
        if should_update:
            last_sent_epoch = (last_sent_epoch + 1) % 256
            update_packet, decision = encode_compact_update(
                state,
                sender_session,
                node_id=catalog_node_id,
                update_epoch=last_sent_epoch,
            )
            packet_size = int(update_packet.size)
            if packet_size < int(compact_codeword_frame_bits):
                raise ValueError("compact packet is shorter than base frame")
            base_escape_bits = (
                packet_size - int(compact_codeword_frame_bits)
            )
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

            accepted_update = False
            if delivered:
                if (
                    receiver.context_installed
                    and receiver.codebook_epoch == sender_session.epoch
                ):
                    decoded = decode_compact_update(
                        update_packet, active_receiver_session
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
                    node_id=catalog_node_id,
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
        if not belief_requires_context_install(
            belief, codebook_epoch=sender_session.epoch
        ):
            establishment_attempts = 0
            activation_epoch = None
            full_fallback_active = not activation_eligible
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
        initial_install_bits=initial_context_bits,
        recovery_install_bits=recovery_context_bits,
        compact_update_bits=compact_update_bits,
        duplicate_update_bits=duplicate_update_bits,
        heartbeat_request_bits=heartbeat_request_bits,
        ack_bits=ack_bits,
        heartbeat_response_bits=heartbeat_response_bits,
        reserved_capacity_bits=0,
        used_reserved_capacity_bits=0,
    )
    matched = MatchedTrajectoryResult(
        bit_breakdown=breakdown,
        context_install_count=context_attempt_count,
        compact_update_count=update_count,
        duplicate_update_count=duplicate_update_count,
        update_reservation_count=0,
        protected_update_count=0,
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
    return ActivationMatchedTrajectoryResult(
        matched=matched,
        activation_attempt_count=activation_attempt_count,
        activation_rejection_count=activation_rejection_count,
        activation_ack_count=activation_ack_count,
        activation_bits=activation_bits,
        initial_activation_bits=initial_activation_bits,
        recovery_activation_bits=recovery_activation_bits,
        full_install_attempt_count=full_install_attempt_count,
        full_install_fallback_count=full_install_fallback_count,
        full_initial_install_bits=full_initial_install_bits,
        full_recovery_install_bits=full_recovery_install_bits,
        forced_update_request_count=forced_update_request_count,
        forced_update_redundant_count=forced_update_redundant_count,
    )
