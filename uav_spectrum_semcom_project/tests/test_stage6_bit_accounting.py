from types import SimpleNamespace

import pytest

from spectrum_semcom.stage6_bit_accounting import (
    reconstruct_frozen_trajectory_bits,
    validate_breakdown_identity,
)


def _trajectory(**overrides):
    values = {
        "context_install_count": 2,
        "compact_update_count": 3,
        "protected_update_count": 1,
        "heartbeat_probe_count": 2,
        "heartbeat_response_count": 1,
        "ack_frame_count": 4,
        "forward_application_bits": 2 * 100 + 3 * 40 + 1 * 40 + 2 * 32,
        "ack_application_bits": 4 * 24 + 1 * 32,
        "reserved_capacity_bits": 2 * 50,
    }
    values.update(overrides)
    values["total_application_bits"] = (
        values["forward_application_bits"]
        + values["ack_application_bits"]
    )
    return SimpleNamespace(**values)


def test_reconstructs_all_bit_identities():
    value = reconstruct_frozen_trajectory_bits(
        _trajectory(),
        context_install_bits=100,
        compact_codeword_update_bits=40,
        ack_frame_bits=24,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        reservation_equivalent_bits=50,
        starts_with_online_install=True,
    )
    validate_breakdown_identity(value)
    assert value.task_frame_bits == (
        value.compact_update_bits
        + value.escape_exact_bits
        + value.duplicate_update_bits
    )
    assert value.exact_action_frame_bits == 0
    assert value.initial_install_bits == 100
    assert value.recovery_install_bits == 100
    assert value.used_reserved_capacity_bits == 50
    assert value.unused_reserved_capacity_bits == 50
    assert value.resource_equivalent_bits == (
        value.actual_total_application_bits + 50
    )


def test_rejects_feedback_components_larger_than_aggregate():
    with pytest.raises(ValueError, match="feedback"):
        reconstruct_frozen_trajectory_bits(
            _trajectory(ack_application_bits=1),
            context_install_bits=100,
            compact_codeword_update_bits=40,
            ack_frame_bits=24,
            heartbeat_request_frame_bits=32,
            heartbeat_response_frame_bits=32,
            reservation_equivalent_bits=50,
            starts_with_online_install=True,
        )


def test_labels_unexplained_forward_remainder_as_escape_bits():
    trajectory = _trajectory(
        forward_application_bits=2 * 100 + 3 * 40 + 1 * 40 + 2 * 32 + 17
    )
    value = reconstruct_frozen_trajectory_bits(
        trajectory,
        context_install_bits=100,
        compact_codeword_update_bits=40,
        ack_frame_bits=24,
        heartbeat_request_frame_bits=32,
        heartbeat_response_frame_bits=32,
        reservation_equivalent_bits=50,
        starts_with_online_install=True,
    )
    assert value.escape_exact_bits == 17
    validate_breakdown_identity(value)
