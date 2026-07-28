"""Versioned bit accounting for Stage-6 matched-reliability experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage6BitBreakdown:
    task_frame_bits: int
    exact_action_frame_bits: int
    escape_exact_bits: int
    initial_install_bits: int
    recovery_install_bits: int
    compact_update_bits: int
    duplicate_update_bits: int
    heartbeat_request_bits: int
    ack_bits: int
    heartbeat_response_bits: int
    other_feedback_bits: int
    actual_forward_bits: int
    actual_feedback_bits: int
    actual_total_application_bits: int
    reserved_capacity_bits: int
    used_reserved_capacity_bits: int
    unused_reserved_capacity_bits: int
    resource_equivalent_bits: int

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def reconstruct_frozen_trajectory_bits(
    trajectory: Any,
    *,
    context_install_bits: int,
    compact_codeword_update_bits: int,
    ack_frame_bits: int,
    heartbeat_request_frame_bits: int,
    heartbeat_response_frame_bits: int,
    reservation_equivalent_bits: int,
    starts_with_online_install: bool,
) -> Stage6BitBreakdown:
    """Reconstruct the frozen simulator's accounting for a no-escape run.

    The function does not change S6-FC0. It derives a detailed ledger from the
    frozen aggregate counters and fails if known components exceed aggregates.
    Any unexplained forward remainder is conservatively labeled
    ``escape_exact_bits``; the excluded-pilot dry-run requires this remainder
    to be zero.
    """

    install_count = int(trajectory.context_install_count)
    update_count = int(trajectory.compact_update_count)
    protected_count = int(trajectory.protected_update_count)
    heartbeat_probe_count = int(trajectory.heartbeat_probe_count)
    heartbeat_response_count = int(trajectory.heartbeat_response_count)
    ack_count = int(trajectory.ack_frame_count)

    initial_install_count = (
        1 if starts_with_online_install and install_count > 0 else 0
    )
    recovery_install_count = install_count - initial_install_count
    initial_install_bits = initial_install_count * int(context_install_bits)
    recovery_install_bits = recovery_install_count * int(context_install_bits)
    compact_update_bits = update_count * int(compact_codeword_update_bits)
    duplicate_update_bits = protected_count * int(
        compact_codeword_update_bits
    )
    heartbeat_request_bits = heartbeat_probe_count * int(
        heartbeat_request_frame_bits
    )
    known_forward = (
        initial_install_bits
        + recovery_install_bits
        + compact_update_bits
        + duplicate_update_bits
        + heartbeat_request_bits
    )
    actual_forward = int(trajectory.forward_application_bits)
    escape_exact_bits = actual_forward - known_forward
    if escape_exact_bits < 0:
        raise ValueError("known forward components exceed aggregate bits")

    ack_bits = ack_count * int(ack_frame_bits)
    heartbeat_response_bits = heartbeat_response_count * int(
        heartbeat_response_frame_bits
    )
    actual_feedback = int(trajectory.ack_application_bits)
    other_feedback = actual_feedback - ack_bits - heartbeat_response_bits
    if other_feedback < 0:
        raise ValueError("known feedback components exceed aggregate bits")

    actual_total = int(trajectory.total_application_bits)
    if actual_total != actual_forward + actual_feedback:
        raise ValueError("actual application-bit identity failed")

    reserved = int(trajectory.reserved_capacity_bits)
    used_reserved = protected_count * int(reservation_equivalent_bits)
    if used_reserved > reserved:
        raise ValueError("used reserved capacity exceeds total reservation")
    unused_reserved = reserved - used_reserved
    resource_equivalent = actual_total + unused_reserved

    task_frame_bits = (
        compact_update_bits + escape_exact_bits + duplicate_update_bits
    )
    return Stage6BitBreakdown(
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
        other_feedback_bits=other_feedback,
        actual_forward_bits=actual_forward,
        actual_feedback_bits=actual_feedback,
        actual_total_application_bits=actual_total,
        reserved_capacity_bits=reserved,
        used_reserved_capacity_bits=used_reserved,
        unused_reserved_capacity_bits=unused_reserved,
        resource_equivalent_bits=resource_equivalent,
    )


def validate_breakdown_identity(value: Stage6BitBreakdown) -> None:
    task_components = (
        value.exact_action_frame_bits
        + value.compact_update_bits
        + value.escape_exact_bits
        + value.duplicate_update_bits
    )
    if task_components != value.task_frame_bits:
        raise ValueError("task-frame component identity failed")
    forward_components = (
        value.initial_install_bits
        + value.recovery_install_bits
        + value.task_frame_bits
        + value.heartbeat_request_bits
    )
    feedback_components = (
        value.ack_bits
        + value.heartbeat_response_bits
        + value.other_feedback_bits
    )
    if forward_components != value.actual_forward_bits:
        raise ValueError("forward component identity failed")
    if feedback_components != value.actual_feedback_bits:
        raise ValueError("feedback component identity failed")
    if (
        value.actual_forward_bits + value.actual_feedback_bits
        != value.actual_total_application_bits
    ):
        raise ValueError("total application-bit identity failed")
    if (
        value.reserved_capacity_bits - value.used_reserved_capacity_bits
        != value.unused_reserved_capacity_bits
    ):
        raise ValueError("reservation identity failed")
    if (
        value.actual_total_application_bits
        + value.unused_reserved_capacity_bits
        != value.resource_equivalent_bits
    ):
        raise ValueError("resource-equivalent identity failed")
