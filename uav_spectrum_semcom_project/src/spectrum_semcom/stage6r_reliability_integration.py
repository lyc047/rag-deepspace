"""Stage-6R calibration and full reliability-protocol integration.

This module is deliberately additive.  It reuses the frozen Stage-6 matched
reliability simulator without modifying its state machine or bit accounting.
The Stage-6R session first sends self-contained exact actions during a
chronological calibration prefix, installs a hash-verified selected codebook,
and then runs the existing event-triggered semantic protocol on the suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spectrum_semcom.stage6_context_codec import (
    decode_context_install,
    encode_compact_update,
    encode_context_install,
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_matched_reliability import (
    MatchedTrajectoryResult,
    combine_matched_trajectory_results,
    simulate_exact_trajectory,
    simulate_semantic_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskState,
    TaskCodeword,
)
from spectrum_semcom.stage6_task_codec import install_codebook


@dataclass(frozen=True)
class CalibratedSemanticTrajectory:
    """One complete Stage-6R session and its auditable sub-trajectories."""

    combined: MatchedTrajectoryResult
    calibration: MatchedTrajectoryResult
    semantic_suffix: MatchedTrajectoryResult
    codebook: GreedyTaskCodebook
    calibration_scene_count: int
    future_scene_count: int
    install_packet_bits: int
    compact_frame_bits: int
    maximum_compact_update_bits: int


def codebook_from_selected_actions(
    calibration_states: Sequence[SpectrumTaskState],
    selected_actions: Sequence[Sequence[int]],
) -> GreedyTaskCodebook:
    """Construct a deterministic codebook from a frozen action selection.

    The selected actions may come from a known activity bank or from the
    calibration-only analytic greedy selector.  This function does not learn
    from the future suffix.  It validates every action against the task schema
    and records calibration coverage for audit purposes.
    """

    states = tuple(calibration_states)
    actions = tuple(tuple(int(value) for value in row) for row in selected_actions)
    if not states or not actions or len(set(actions)) != len(actions):
        raise ValueError("calibration states and unique selected actions are required")
    reference = states[0]
    for state in states[1:]:
        if (
            state.n_channels != reference.n_channels
            or state.schema != reference.schema
            or abs(state.epsilon_db - reference.epsilon_db) > 1e-12
        ):
            raise ValueError("calibration states do not share one task schema")
    for action_tuple in actions:
        if len(action_tuple) != len(reference.profiles):
            raise ValueError("selected action does not match the query count")
        for profile, action in zip(reference.profiles, action_tuple):
            if int(action) not in profile.allowed_starts:
                raise ValueError("selected action is outside the query alphabet")

    covered_union: set[int] = set()
    codewords = []
    for codeword_id, action_tuple in enumerate(actions):
        covered = {
            index
            for index, state in enumerate(states)
            if state.accepts(action_tuple)
        }
        covered_union.update(covered)
        codewords.append(
            TaskCodeword(
                codeword_id=codeword_id,
                decoder_actions=action_tuple,
                training_coverage_count=len(covered),
            )
        )
    return GreedyTaskCodebook(
        n_channels=reference.n_channels,
        queries=reference.queries,
        epsilon_db=reference.epsilon_db,
        codewords=tuple(codewords),
        training_scene_count=len(states),
        exact_action_tuple_count=len(
            {state.optimal_actions for state in states}
        ),
        training_covered_count=len(covered_union),
    )


def _compact_frame_bits(
    states: Sequence[SpectrumTaskState],
    sender_session: Any,
) -> int:
    for state in states:
        packet, decision = encode_compact_update(
            state,
            sender_session,
            node_id=1,
            update_epoch=1,
        )
        if not decision.uses_fallback:
            return int(packet.size)
    raise ValueError("selected codebook has no compactly encodable state")


def simulate_calibrated_semantic_trajectory(
    states: list[SpectrumTaskState],
    timestamps: np.ndarray,
    session_indices: np.ndarray,
    *,
    calibration_scene_count: int,
    selected_actions: Sequence[Sequence[int]],
    random_values: np.ndarray,
    condition: dict[str, Any],
    exact_packet_bits: int,
    epsilon_db: float,
    max_age_minutes: float,
    ack_frame_bits: int,
    outage_penalty_db: float,
    heartbeat_interval_scenes: int | None,
    heartbeat_request_frame_bits: int,
    heartbeat_response_frame_bits: int,
    calibration_open_loop_attempts: int,
    task_open_loop_attempts: int,
    codebook_epoch: int,
    update_protection_candidates: np.ndarray | None = None,
    maximum_update_reservations: int = 0,
) -> CalibratedSemanticTrajectory:
    """Run one conservative full-protocol Stage-6R session.

    A full context-install packet is charged after calibration for both a
    bank-derived and an adapted codebook.  This avoids claiming an unimplemented
    short bank-activation protocol.  The receiver starts the semantic suffix
    with no installed context, so installation loss, ACK loss, later reset,
    heartbeat, and recovery are all exercised by the frozen simulator.
    """

    indices = np.asarray(session_indices, dtype=np.int64).reshape(-1)
    random = np.asarray(random_values, dtype=np.float64)
    calibration_count = int(calibration_scene_count)
    if (
        indices.size < 2
        or not 1 <= calibration_count < indices.size
        or random.shape[0] != indices.size
        or np.any(indices < 0)
        or np.any(indices >= len(states))
    ):
        raise ValueError("invalid calibrated Stage-6R session")
    calibration_indices = indices[:calibration_count]
    future_indices = indices[calibration_count:]
    calibration_states = [states[int(index)] for index in calibration_indices]
    codebook = codebook_from_selected_actions(
        calibration_states,
        selected_actions,
    )
    sender = install_codebook(codebook, epoch=int(codebook_epoch))
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    compact_bits = _compact_frame_bits(
        [states[int(index)] for index in indices],
        sender,
    )
    reservation_bits = int(maximum_compact_update_bits(sender))
    protection = (
        np.zeros(len(states), dtype=bool)
        if update_protection_candidates is None
        else np.asarray(update_protection_candidates, dtype=bool)
    )

    calibration = simulate_exact_trajectory(
        states,
        timestamps,
        calibration_indices,
        random_values=random[:calibration_count],
        packet_loss_probability=float(condition["task_loss_probability"]),
        receiver_reset_probability=float(
            condition["receiver_context_reset_probability"]
        ),
        task_open_loop_attempts=int(calibration_open_loop_attempts),
        packet_bits=int(exact_packet_bits),
        epsilon_db=float(epsilon_db),
        outage_penalty_db=float(outage_penalty_db),
    )
    semantic = simulate_semantic_trajectory(
        states,
        timestamps,
        future_indices,
        sender_session=sender,
        receiver_session=receiver,
        install_packet=install_packet,
        deployment_mode="online_install_empty_context",
        condition=condition,
        random_values=random[calibration_count:],
        epsilon_db=float(epsilon_db),
        max_age_minutes=float(max_age_minutes),
        ack_frame_bits=int(ack_frame_bits),
        outage_penalty_db=float(outage_penalty_db),
        heartbeat_interval_scenes=heartbeat_interval_scenes,
        heartbeat_request_frame_bits=int(heartbeat_request_frame_bits),
        heartbeat_response_frame_bits=int(heartbeat_response_frame_bits),
        task_open_loop_attempts=int(task_open_loop_attempts),
        compact_codeword_frame_bits=compact_bits,
        update_protection_candidates=protection,
        maximum_update_reservations=int(maximum_update_reservations),
        reservation_equivalent_bits=reservation_bits,
    )
    combined = combine_matched_trajectory_results([calibration, semantic])
    return CalibratedSemanticTrajectory(
        combined=combined,
        calibration=calibration,
        semantic_suffix=semantic,
        codebook=codebook,
        calibration_scene_count=calibration_count,
        future_scene_count=int(future_indices.size),
        install_packet_bits=int(install_packet.size),
        compact_frame_bits=compact_bits,
        maximum_compact_update_bits=reservation_bits,
    )
