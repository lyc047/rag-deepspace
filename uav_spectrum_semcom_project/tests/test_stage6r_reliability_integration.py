from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage6_bit_accounting import validate_breakdown_identity
from spectrum_semcom.stage6_matched_reliability import RANDOM_STREAM_COUNT
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
)
from spectrum_semcom.stage6r_reliability_integration import (
    codebook_from_selected_actions,
    simulate_calibrated_semantic_trajectory,
)


def _states() -> list:
    rows = np.asarray(
        [
            [-90.0, -89.0, -80.0, -79.0],
            [-91.0, -90.0, -80.0, -79.0],
            [-90.0, -89.0, -81.0, -80.0],
            [-80.0, -79.0, -91.0, -90.0],
            [-81.0, -80.0, -92.0, -91.0],
            [-82.0, -81.0, -93.0, -92.0],
        ]
    )
    queries = (SpectrumTaskQuery(1), SpectrumTaskQuery(2))
    return [
        build_task_state(row, queries, epsilon_db=0.2) for row in rows
    ]


def _condition() -> dict:
    return {
        "install_loss_probability": 0.0,
        "task_loss_probability": 0.0,
        "ack_loss_probability": 0.0,
        "delayed_duplicate_probability": 0.0,
        "receiver_context_reset_probability": 0.0,
        "include_reset_hypothesis_after_missing_ack": True,
    }


def test_selected_action_codebook_records_calibration_coverage() -> None:
    states = _states()
    codebook = codebook_from_selected_actions(
        states[:3],
        [states[0].optimal_actions],
    )
    assert codebook.training_scene_count == 3
    assert codebook.training_covered_count == 3
    assert codebook.codewords[0].training_coverage_count == 3
    assert codebook.symbol_width_bits == 1


def test_selected_action_codebook_rejects_duplicate_actions() -> None:
    states = _states()
    actions = states[0].optimal_actions
    with pytest.raises(ValueError, match="unique selected actions"):
        codebook_from_selected_actions(states[:2], [actions, actions])


def test_full_protocol_session_charges_calibration_and_install() -> None:
    states = _states()
    random = np.ones((len(states), RANDOM_STREAM_COUNT), dtype=np.float64)
    result = simulate_calibrated_semantic_trajectory(
        states,
        np.asarray(
            [f"2026-01-01T00:0{i}:00" for i in range(len(states))]
        ),
        np.arange(len(states)),
        calibration_scene_count=2,
        selected_actions=[states[0].optimal_actions],
        random_values=random,
        condition=_condition(),
        exact_packet_bits=56,
        epsilon_db=0.2,
        max_age_minutes=60.0,
        ack_frame_bits=24,
        outage_penalty_db=10.0,
        heartbeat_interval_scenes=None,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        calibration_open_loop_attempts=1,
        task_open_loop_attempts=1,
        codebook_epoch=1,
    )
    bits = result.combined.bit_breakdown
    validate_breakdown_identity(bits)
    assert result.calibration.bit_breakdown.exact_action_frame_bits == 112
    assert result.semantic_suffix.bit_breakdown.initial_install_bits == (
        result.install_packet_bits
    )
    assert bits.exact_action_frame_bits == 112
    assert bits.initial_install_bits == result.install_packet_bits
    assert result.combined.wrong_codebook_decode_count == 0
    assert len(result.combined.clean) == len(states)
    assert all(result.combined.available)
