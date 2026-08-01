from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage9_1_hybrid_recovery import HybridFaultSchedule
from spectrum_semcom.stage9_5_causal_feedback import (
    POLICIES,
    ResetNack,
    decode_reset_nack,
    encode_reset_nack,
    simulate_causal_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import FaultSchedule, SilProtocolBits, TaskTrace


def _bits() -> SilProtocolBits:
    return SilProtocolBits(42, 24, 32, 32, 64, 64, 62, 403)


def _trace(scenes: int, change: bool = True) -> TaskTrace:
    desired = np.arange(scenes, dtype=np.int64) % 3 if change else np.zeros(scenes, dtype=np.int64)
    return TaskTrace(desired, tuple(frozenset({int(value)}) for value in desired))


def _faults(scenes: int, *, emitter: bool, restart_at: int = 2) -> HybridFaultSchedule:
    restart = np.zeros(scenes, dtype=bool)
    restart[restart_at] = True
    return HybridFaultSchedule(
        FaultSchedule(
            restart,
            np.zeros(scenes, dtype=bool),
            np.ones(scenes, dtype=bool),
            np.ones(scenes, dtype=bool),
        ),
        np.full(scenes, emitter, dtype=bool),
    )


def test_reset_nack_codec_is_exactly_32_bits():
    message = ResetNack(1, 65530, 7)
    bits = encode_reset_nack(message)
    assert bits.size == 32
    assert decode_reset_nack(bits) == message
    corrupt = bits.copy()
    corrupt[0] ^= 1
    with pytest.raises(ValueError):
        decode_reset_nack(corrupt)


def test_reset_nack_recovers_before_watchdog_when_boot_announcement_fails():
    scenes = 70
    nack = simulate_causal_trajectory(
        _trace(scenes),
        _faults(scenes, emitter=False),
        policy="causal_hbw30_reset_nack",
        bits=_bits(),
    )
    no_nack = simulate_causal_trajectory(
        _trace(scenes),
        _faults(scenes, emitter=False),
        policy="causal_hbw30_no_nack",
        bits=_bits(),
    )
    assert nack.reset_nack_frame_count > 0
    assert nack.clean_count > no_nack.clean_count
    assert max(nack.recovery_latencies) < max(no_nack.recovery_latencies)


@pytest.mark.parametrize("policy", POLICIES)
def test_all_policies_are_causally_closed_and_bit_accounted(policy):
    result = simulate_causal_trajectory(
        _trace(80, change=False),
        _faults(80, emitter=False),
        policy=policy,
        bits=_bits(),
    )
    assert result.sender_internal_state_read_count == 0
    assert result.uncharged_feedback_transition_count == 0
    assert result.wrong_context_execution_count == 0
    assert result.stale_restore_acceptance_count == 0
    assert result.total_bits == (
        result.heartbeat_bits
        + result.boot_status_bits
        + result.reset_nack_bits
        + result.install_bits
        + result.update_bits
        + result.ack_bits
    )

