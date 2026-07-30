from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage5_cumulative_ack import (
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_bit_accounting import validate_breakdown_identity
from spectrum_semcom.stage6_context_codec import (
    decode_context_install,
    encode_compact_update,
    encode_context_install,
)
from spectrum_semcom.stage6_context_heartbeat import (
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_matched_reliability import (
    RANDOM_STREAM_COUNT,
    RESET_STREAM,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook
from spectrum_semcom.stage6r_activation_reliability import (
    simulate_activation_semantic_trajectory,
)
from spectrum_semcom.stage6r_codebook_activation import (
    build_preinstalled_catalog,
)


def _fixture():
    queries = (SpectrumTaskQuery(1), SpectrumTaskQuery(2))
    rows = np.asarray(
        [
            [-90.0, -89.0, -80.0, -79.0],
            [-91.0, -90.0, -80.0, -79.0],
            [-90.0, -89.0, -81.0, -80.0],
            [-89.0, -88.0, -80.0, -79.0],
        ]
    )
    states = [
        build_task_state(row, queries, epsilon_db=0.2) for row in rows
    ]
    codebook = fit_greedy_task_codebook(states)
    sender = install_codebook(codebook, epoch=2)
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    catalog = build_preinstalled_catalog([(7, codebook)], catalog_epoch=3)
    compact_packet, decision = encode_compact_update(
        states[0], sender, node_id=1, update_epoch=1
    )
    assert not decision.uses_fallback
    return states, sender, receiver, install_packet, catalog, compact_packet.size


def _condition(**overrides):
    value = {
        "install_loss_probability": 0.0,
        "task_loss_probability": 0.0,
        "ack_loss_probability": 0.0,
        "delayed_duplicate_probability": 0.0,
        "receiver_context_reset_probability": 0.0,
        "include_reset_hypothesis_after_missing_ack": True,
    }
    value.update(overrides)
    return value


def _run(*, receiver_catalog=None, condition=None):
    states, sender, receiver, install_packet, catalog, compact_bits = _fixture()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    result = simulate_activation_semantic_trajectory(
        states,
        np.asarray(
            [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
        ),
        np.arange(len(states)),
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=install_packet,
        sender_catalog=catalog,
        receiver_catalog=(
            catalog if receiver_catalog is None else receiver_catalog
        ),
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition() if condition is None else condition,
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
    )
    return result, catalog


def test_no_fault_activation_uses_64_bits_and_stays_clean() -> None:
    result, _ = _run()
    validate_breakdown_identity(result.matched.bit_breakdown)
    assert result.activation_attempt_count == 1
    assert result.activation_bits == 64
    assert result.full_install_attempt_count == 0
    assert result.matched.bit_breakdown.initial_install_bits == 64
    assert result.matched.wrong_codebook_decode_count == 0
    assert result.checkpoint_count == 0
    assert result.checkpoint_bits == 0
    assert result.event_context_update_count == 0
    assert result.event_context_update_bits == 0
    assert all(result.matched.clean)


def test_stage7_trigger_mask_replaces_fixed_heartbeat_without_bypassing_checks() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    timestamps = np.asarray(
        [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
    )
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    trigger = np.asarray([False, True, False, False], dtype=bool)
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1, codebook_epoch=2, expected_update_epoch=0
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1, codebook_epoch=2, current_update_epoch=0
        ).size
    )
    result = simulate_activation_semantic_trajectory(
        states,
        timestamps,
        np.arange(len(states)),
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=packet,
        sender_catalog=catalog,
        receiver_catalog=catalog,
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=heartbeat_request_bits,
        heartbeat_response_frame_bits=heartbeat_response_bits,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
        heartbeat_trigger_mask=trigger,
    )
    assert result.matched.heartbeat_probe_count == 1
    assert result.matched.heartbeat_response_count == 1
    assert result.matched.wrong_codebook_decode_count == 0


def test_stage7_trigger_mask_rejects_misalignment_and_mixed_control() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    timestamps = np.asarray(
        [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
    )
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1, codebook_epoch=2, expected_update_epoch=0
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1, codebook_epoch=2, current_update_epoch=0
        ).size
    )
    common = dict(
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=packet,
        sender_catalog=catalog,
        receiver_catalog=catalog,
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_request_frame_bits=heartbeat_request_bits,
        heartbeat_response_frame_bits=heartbeat_response_bits,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
    )
    with pytest.raises(ValueError, match="align"):
        simulate_activation_semantic_trajectory(
            states,
            timestamps,
            np.arange(len(states)),
            heartbeat_interval_scenes=None,
            heartbeat_trigger_mask=np.asarray([True]),
            **common,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate_activation_semantic_trajectory(
            states,
            timestamps,
            np.arange(len(states)),
            heartbeat_interval_scenes=10,
            heartbeat_trigger_mask=np.zeros(len(states), dtype=bool),
            **common,
        )


def test_stage7_forced_update_uses_normal_reliability_and_bit_paths() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    timestamps = np.asarray(
        [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
    )
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    common = dict(
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=packet,
        sender_catalog=catalog,
        receiver_catalog=catalog,
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
    )
    baseline = simulate_activation_semantic_trajectory(
        states,
        timestamps,
        np.arange(len(states)),
        **common,
    )
    forced = simulate_activation_semantic_trajectory(
        states,
        timestamps,
        np.arange(len(states)),
        forced_update_trigger_mask=np.asarray(
            [False, False, True, False], dtype=bool
        ),
        **common,
    )
    assert (
        forced.matched.compact_update_count
        == baseline.matched.compact_update_count + 1
    )
    assert (
        forced.matched.bit_breakdown.compact_update_bits
        == baseline.matched.bit_breakdown.compact_update_bits + compact_bits
    )
    assert forced.forced_update_request_count == 1
    assert forced.forced_update_redundant_count == 0
    assert forced.matched.wrong_codebook_decode_count == 0


def test_stage7_forced_update_rejects_misaligned_mask() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    with pytest.raises(ValueError, match="forced update trigger mask"):
        simulate_activation_semantic_trajectory(
            states,
            np.asarray(
                [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
            ),
            np.arange(len(states)),
            sender_session=sender,
            receiver_session=receiver,
            full_install_packet=packet,
            sender_catalog=catalog,
            receiver_catalog=catalog,
            bank_id=7,
            catalog_node_id=1,
            maximum_activation_attempts=2,
            condition=_condition(),
            random_values=random,
            epsilon_db=0.2,
            max_age_minutes=60.0,
            ack_frame_bits=cumulative_ack_payload_bits(),
            outage_penalty_db=10.0,
            heartbeat_interval_scenes=None,
            heartbeat_request_frame_bits=32,
            heartbeat_response_frame_bits=32,
            task_open_loop_attempts=1,
            compact_codeword_frame_bits=int(compact_bits),
            forced_update_trigger_mask=np.asarray([True], dtype=bool),
        )


def test_stage7_checkpoint_restores_context_through_normal_ack_path() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    result = simulate_activation_semantic_trajectory(
        states,
        np.asarray(
            [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
        ),
        np.arange(len(states)),
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=packet,
        sender_catalog=catalog,
        receiver_catalog=catalog,
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
        checkpoint_interval_scenes=1,
    )
    validate_breakdown_identity(result.matched.bit_breakdown)
    assert result.checkpoint_count >= 1
    assert result.checkpoint_ack_count == result.checkpoint_count
    assert result.checkpoint_rejection_count == 0
    assert result.checkpoint_bits > 0
    assert result.matched.wrong_codebook_decode_count == 0
    assert all(result.matched.clean)


def test_stage7_checkpoint_and_heartbeat_are_mutually_exclusive() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate_activation_semantic_trajectory(
            states,
            np.asarray(
                [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
            ),
            np.arange(len(states)),
            sender_session=sender,
            receiver_session=receiver,
            full_install_packet=packet,
            sender_catalog=catalog,
            receiver_catalog=catalog,
            bank_id=7,
            catalog_node_id=1,
            maximum_activation_attempts=2,
            condition=_condition(),
            random_values=random,
            epsilon_db=0.2,
            max_age_minutes=60.0,
            ack_frame_bits=cumulative_ack_payload_bits(),
            outage_penalty_db=10.0,
            heartbeat_interval_scenes=10,
            heartbeat_request_frame_bits=32,
            heartbeat_response_frame_bits=32,
            task_open_loop_attempts=1,
            compact_codeword_frame_bits=int(compact_bits),
            checkpoint_interval_scenes=10,
        )


def test_stage7_event_context_update_rescues_hidden_reset() -> None:
    states, sender, receiver, packet, catalog, compact_bits = _fixture()
    timestamps = np.asarray(
        [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
    )
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    random[2, RESET_STREAM] = 0.0
    common = dict(
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=packet,
        sender_catalog=catalog,
        receiver_catalog=catalog,
        bank_id=7,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(receiver_context_reset_probability=0.5),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
        forced_update_trigger_mask=np.asarray(
            [False, False, True, False], dtype=bool
        ),
    )
    compact = simulate_activation_semantic_trajectory(
        states,
        timestamps,
        np.arange(len(states)),
        **common,
    )
    piggyback = simulate_activation_semantic_trajectory(
        states,
        timestamps,
        np.arange(len(states)),
        self_contained_event_updates=True,
        **common,
    )
    validate_breakdown_identity(piggyback.matched.bit_breakdown)
    assert compact.matched.clean[2] is False
    assert piggyback.matched.clean[2] is True
    assert piggyback.event_context_update_count >= 1
    assert piggyback.event_context_reset_rescue_count == 1
    assert piggyback.event_context_incremental_bits > 0
    assert piggyback.event_context_rejection_count == 0
    assert piggyback.matched.wrong_codebook_decode_count == 0


def test_catalog_mismatch_falls_back_to_full_install() -> None:
    states, _, _, _, _, _ = _fixture()
    wrong_codebook = fit_greedy_task_codebook(
        [
            build_task_state(
                np.asarray([-70.0, -90.0, -80.0, -79.0]),
                states[0].queries,
                epsilon_db=0.2,
            )
        ]
    )
    wrong_catalog = build_preinstalled_catalog(
        [(7, wrong_codebook)],
        catalog_epoch=3,
    )
    result, _ = _run(receiver_catalog=wrong_catalog)
    validate_breakdown_identity(result.matched.bit_breakdown)
    assert result.activation_attempt_count == 2
    assert result.activation_rejection_count == 2
    assert result.full_install_fallback_count == 1
    assert result.full_install_attempt_count >= 1
    assert result.matched.wrong_codebook_decode_count == 0
    assert result.matched.available[-1]


def test_ood_path_uses_full_install_only() -> None:
    states, sender, receiver, install_packet, _, compact_bits = _fixture()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    result = simulate_activation_semantic_trajectory(
        states,
        np.asarray(
            [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
        ),
        np.arange(len(states)),
        sender_session=sender,
        receiver_session=receiver,
        full_install_packet=install_packet,
        sender_catalog=None,
        receiver_catalog=None,
        bank_id=None,
        catalog_node_id=1,
        maximum_activation_attempts=2,
        condition=_condition(),
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=int(compact_bits),
    )
    assert result.activation_attempt_count == 0
    assert result.full_install_attempt_count == 1
    assert result.full_initial_install_bits == install_packet.size
    assert all(result.matched.clean)
