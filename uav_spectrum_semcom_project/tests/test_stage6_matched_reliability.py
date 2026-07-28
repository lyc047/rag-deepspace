from __future__ import annotations

import numpy as np

from spectrum_semcom.stage5_cumulative_ack import (
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (
    decode_context_install,
    encode_compact_update,
    encode_context_install,
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_context_heartbeat import (
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_matched_reliability import (
    RANDOM_STREAM_COUNT,
    build_grouped_sessions,
    combine_matched_trajectory_results,
    simulate_exact_trajectory,
    simulate_semantic_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook


def _fixture():
    queries = (SpectrumTaskQuery(2),)
    states = [
        build_task_state(
            np.asarray([-100.0, -99.0, -80.0, -70.0]),
            queries,
            epsilon_db=0.2,
        ),
        build_task_state(
            np.asarray([-70.0, -80.0, -99.0, -100.0]),
            queries,
            epsilon_db=0.2,
        ),
    ]
    codebook = fit_greedy_task_codebook(states)
    sender = install_codebook(codebook, epoch=1)
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    update, decision = encode_compact_update(
        states[0],
        sender,
        node_id=1,
        update_epoch=1,
    )
    assert not decision.uses_fallback
    timestamps = np.asarray(
        ["2023-01-01T00:00:00", "2023-01-01T00:00:30"]
    )
    return states, timestamps, sender, receiver, install_packet, update.size


def _condition(task_loss: float = 0.0) -> dict[str, object]:
    return {
        "install_loss_probability": 0.0,
        "task_loss_probability": task_loss,
        "ack_loss_probability": 0.0,
        "delayed_duplicate_probability": 0.0,
        "receiver_context_reset_probability": 0.0,
        "include_reset_hypothesis_after_missing_ack": True,
    }


def _semantic(
    *,
    attempts: int,
    random_values: np.ndarray,
    deployment_mode: str,
    task_loss: float = 0.0,
):
    states, timestamps, sender, receiver, install_packet, compact_bits = (
        _fixture()
    )
    request_bits = encode_context_probe_request(
        node_id=1,
        codebook_epoch=1,
        expected_update_epoch=0,
    ).size
    response_bits = encode_context_probe_response(
        node_id=1,
        codebook_epoch=1,
        current_update_epoch=0,
    ).size
    return simulate_semantic_trajectory(
        states,
        timestamps,
        np.asarray([0]),
        sender_session=sender,
        receiver_session=receiver,
        install_packet=install_packet,
        deployment_mode=deployment_mode,
        condition=_condition(task_loss),
        random_values=random_values,
        epsilon_db=0.2,
        max_age_minutes=120.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=20,
        heartbeat_request_frame_bits=int(request_bits),
        heartbeat_response_frame_bits=int(response_bits),
        task_open_loop_attempts=attempts,
        compact_codeword_frame_bits=int(compact_bits),
        update_protection_candidates=np.zeros(len(states), dtype=bool),
        maximum_update_reservations=0,
        reservation_equivalent_bits=maximum_compact_update_bits(sender),
    )


def test_exact_attempts_share_nested_random_streams() -> None:
    states, timestamps, *_ = _fixture()
    random = np.ones((1, RANDOM_STREAM_COUNT), dtype=np.float64)
    random[0, 3] = 0.1
    random[0, 4] = 0.9
    one = simulate_exact_trajectory(
        states,
        timestamps,
        np.asarray([0]),
        random_values=random,
        packet_loss_probability=0.5,
        receiver_reset_probability=0.0,
        task_open_loop_attempts=1,
        packet_bits=50,
        epsilon_db=0.2,
        outage_penalty_db=10.0,
    )
    two = simulate_exact_trajectory(
        states,
        timestamps,
        np.asarray([0]),
        random_values=random,
        packet_loss_probability=0.5,
        receiver_reset_probability=0.0,
        task_open_loop_attempts=2,
        packet_bits=50,
        epsilon_db=0.2,
        outage_penalty_db=10.0,
    )
    assert one.clean == (False,)
    assert two.clean == (True,)
    assert one.bit_breakdown.actual_total_application_bits == 50
    assert two.bit_breakdown.actual_total_application_bits == 100


def test_preconfigured_session_omits_only_initial_install() -> None:
    random = np.ones((1, RANDOM_STREAM_COUNT), dtype=np.float64)
    online = _semantic(
        attempts=1,
        random_values=random,
        deployment_mode="online_install_empty_context",
    )
    preconfigured = _semantic(
        attempts=1,
        random_values=random,
        deployment_mode="preconfigured_codebook_empty_action_state",
    )
    assert online.clean == preconfigured.clean == (True,)
    assert online.context_install_count == 1
    assert preconfigured.context_install_count == 0
    assert online.bit_breakdown.initial_install_bits > 0
    assert preconfigured.bit_breakdown.initial_install_bits == 0
    assert (
        online.bit_breakdown.actual_total_application_bits
        - preconfigured.bit_breakdown.actual_total_application_bits
        == online.bit_breakdown.initial_install_bits
        + cumulative_ack_payload_bits()
    )


def test_semantic_attempts_share_nested_task_random_streams() -> None:
    random = np.ones((1, RANDOM_STREAM_COUNT), dtype=np.float64)
    random[0, 3] = 0.1
    random[0, 4] = 0.9
    one = _semantic(
        attempts=1,
        random_values=random,
        deployment_mode="preconfigured_codebook_empty_action_state",
        task_loss=0.5,
    )
    two = _semantic(
        attempts=2,
        random_values=random,
        deployment_mode="preconfigured_codebook_empty_action_state",
        task_loss=0.5,
    )
    assert one.clean == (False,)
    assert two.clean == (True,)
    assert two.duplicate_update_count == 1
    assert (
        two.bit_breakdown.task_frame_bits
        == 2 * one.bit_breakdown.task_frame_bits
    )


def test_risk_protection_adds_one_attempt_and_consumes_reservation() -> None:
    states, timestamps, sender, receiver, install_packet, compact_bits = (
        _fixture()
    )
    random = np.ones((1, RANDOM_STREAM_COUNT), dtype=np.float64)
    reservation_bits = maximum_compact_update_bits(sender)
    value = simulate_semantic_trajectory(
        states,
        timestamps,
        np.asarray([0]),
        sender_session=sender,
        receiver_session=receiver,
        install_packet=install_packet,
        deployment_mode="preconfigured_codebook_empty_action_state",
        condition=_condition(0.0),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=120.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
        update_protection_candidates=np.ones(len(states), dtype=bool),
        maximum_update_reservations=1,
        reservation_equivalent_bits=reservation_bits,
    )
    assert value.protected_update_count == 1
    assert value.duplicate_update_count == 1
    assert value.bit_breakdown.duplicate_update_bits == compact_bits
    assert value.bit_breakdown.used_reserved_capacity_bits == reservation_bits
    assert value.bit_breakdown.unused_reserved_capacity_bits == 0


def test_grouped_sessions_never_cross_boundaries() -> None:
    groups = np.asarray(["a"] * 5 + ["b"] * 4)
    sessions = build_grouped_sessions(
        np.arange(9),
        groups,
        target_scene_count=3,
        minimum_remainder_scenes=2,
    )
    assert [value.tolist() for value in sessions] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, 7],
    ]
    assert all(len(set(groups[value])) == 1 for value in sessions)


def test_combining_sessions_preserves_bit_identities() -> None:
    random = np.ones((1, RANDOM_STREAM_COUNT), dtype=np.float64)
    first = _semantic(
        attempts=1,
        random_values=random,
        deployment_mode="online_install_empty_context",
    )
    second = _semantic(
        attempts=1,
        random_values=random,
        deployment_mode="online_install_empty_context",
    )
    combined = combine_matched_trajectory_results([first, second])
    assert len(combined.clean) == 2
    assert combined.context_install_count == 2
    assert (
        combined.bit_breakdown.actual_total_application_bits
        == 2 * first.bit_breakdown.actual_total_application_bits
    )
