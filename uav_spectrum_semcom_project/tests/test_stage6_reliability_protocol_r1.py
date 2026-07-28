from __future__ import annotations

import numpy as np

from spectrum_semcom.stage5_cumulative_ack import (
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (
    decode_context_install,
    encode_compact_update,
    encode_context_install,
)
from spectrum_semcom.stage6_reliability_protocol_r1 import (
    R1_VARIANTS,
    combine_r1_trajectory_results,
    simulate_r1_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook


def _fixture():
    query = (SpectrumTaskQuery(2),)
    states = [
        build_task_state(
            np.asarray([0.0, 0.0, 10.0, 10.0]),
            query,
            epsilon_db=0.2,
        ),
        build_task_state(
            np.asarray([10.0, 10.0, 0.0, 0.0]),
            query,
            epsilon_db=0.2,
        ),
        build_task_state(
            np.asarray([10.0, 10.0, 0.0, 0.0]),
            query,
            epsilon_db=0.2,
        ),
    ]
    codebook = fit_greedy_task_codebook(states)
    sender = install_codebook(codebook, epoch=1)
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    compact, decision = encode_compact_update(
        states[0], sender, node_id=1, update_epoch=1
    )
    assert not decision.uses_fallback
    condition = {
        "install_loss_probability": 0.0,
        "task_loss_probability": 0.0,
        "ack_loss_probability": 0.0,
        "delayed_duplicate_probability": 0.0,
        "receiver_context_reset_probability": 0.0,
    }
    return (
        states,
        sender,
        receiver,
        install_packet,
        int(compact.size),
        condition,
    )


def _run(variant: str, random: np.ndarray, condition: dict, deployment: str):
    (
        states,
        sender,
        receiver,
        install_packet,
        compact_bits,
        _,
    ) = _fixture()
    return simulate_r1_trajectory(
        states,
        np.asarray(
            [
                "2022-01-01T00:00:00",
                "2022-01-01T00:00:30",
                "2022-01-01T00:01:00",
            ]
        ),
        np.arange(3),
        variant=variant,
        sender_session=sender,
        receiver_session=receiver,
        install_packet=install_packet,
        deployment_mode=deployment,
        condition=condition,
        random_values=random,
        epsilon_db=0.2,
        max_age_minutes=30.0,
        ack_frame_bits=cumulative_ack_payload_bits(),
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=10,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        task_open_loop_attempts=1,
        compact_codeword_frame_bits=compact_bits,
        exact_action_frame_bits=49,
    )


def test_all_r1_variants_are_clean_without_faults() -> None:
    random = np.ones((3, 11), dtype=np.float64)
    condition = _fixture()[-1]
    for variant in R1_VARIANTS:
        value = _run(
            variant,
            random,
            condition,
            "preconfigured_codebook_empty_action_state",
        )
        assert value.clean == (True, True, True)
        assert value.wrong_codebook_decode_count == 0


def test_durable_codebook_recovers_reset_with_compact_frame() -> None:
    random = np.ones((3, 11), dtype=np.float64)
    random[1, 0] = 0.0
    condition = dict(_fixture()[-1])
    condition["receiver_context_reset_probability"] = 0.5
    value = _run(
        "v2_durable_codebook_compact_refresh",
        random,
        condition,
        "preconfigured_codebook_empty_action_state",
    )
    assert value.clean == (True, True, True)
    assert value.actual_action_reset_count == 1
    assert value.actual_codebook_reset_count == 0
    assert value.exact_refresh_count == 0
    assert value.compact_refresh_count == 2


def test_hybrid_uses_exact_refresh_after_full_context_reset() -> None:
    random = np.ones((3, 11), dtype=np.float64)
    random[1, 0] = 0.0
    condition = dict(_fixture()[-1])
    condition["receiver_context_reset_probability"] = 0.5
    value = _run(
        "v3_context_independent_exact_refresh",
        random,
        condition,
        "preconfigured_codebook_empty_action_state",
    )
    assert value.clean == (True, False, True)
    assert value.actual_codebook_reset_count == 1
    assert value.rejected_compact_count == 1
    assert value.exact_refresh_count == 1
    assert value.wrong_codebook_decode_count == 0


def test_online_durable_codebook_installs_once_when_ack_arrives() -> None:
    random = np.ones((3, 11), dtype=np.float64)
    condition = _fixture()[-1]
    value = _run(
        "v2_durable_codebook_compact_refresh",
        random,
        condition,
        "online_install_empty_context",
    )
    assert value.initial_install_count == 1
    assert value.recovery_install_count == 0
    assert value.clean == (True, True, True)


def test_combination_preserves_bit_identity_and_reason_counts() -> None:
    random = np.ones((3, 11), dtype=np.float64)
    condition = _fixture()[-1]
    value = _run(
        "v4_event_triggered_exact_action_cache",
        random,
        condition,
        "preconfigured_codebook_empty_action_state",
    )
    combined = combine_r1_trajectory_results([value, value])
    assert combined.exact_refresh_count == 2 * value.exact_refresh_count
    assert (
        combined.bit_breakdown.actual_total_application_bits
        == 2 * value.bit_breakdown.actual_total_application_bits
    )
