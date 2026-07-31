from __future__ import annotations

import numpy as np

from spectrum_semcom.stage9_1_hybrid_recovery import HybridFaultSchedule
from spectrum_semcom.stage9_4_tagged_hybrid_integration import (
    simulate_tagged_hbw30_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import (
    FaultSchedule,
    SilProtocolBits,
    TaskTrace,
)


def _bits() -> SilProtocolBits:
    return SilProtocolBits(42, 24, 32, 32, 64, 64, 62, 403)


def _trace(scenes: int) -> TaskTrace:
    desired = np.arange(scenes, dtype=np.int64) % 3
    return TaskTrace(
        desired_symbol=desired,
        acceptable_symbols=tuple(frozenset({int(value)}) for value in desired),
    )


def test_tag_cost_is_exact_and_task_outcome_matches_zero_tag_shadow():
    scenes = 100
    faults = HybridFaultSchedule(
        base=FaultSchedule(
            restart=np.zeros(scenes, dtype=bool),
            cold_restart=np.zeros(scenes, dtype=bool),
            forward_delivered=np.ones(scenes, dtype=bool),
            feedback_delivered=np.ones(scenes, dtype=bool),
        ),
        boot_emitter_available=np.ones(scenes, dtype=bool),
    )
    tagged = simulate_tagged_hbw30_trajectory(
        _trace(scenes), faults, bits=_bits(), restore_tag_bits=16
    )
    shadow = simulate_tagged_hbw30_trajectory(
        _trace(scenes), faults, bits=_bits(), restore_tag_bits=0
    )
    assert tagged.total_bits - shadow.total_bits == 16 * tagged.restore_frame_count
    assert tagged.recovery_tag_bits == 16 * tagged.restore_frame_count
    assert tagged.update_bits == shadow.update_bits
    assert tagged.ack_bits == shadow.ack_bits
    assert tagged.clean_count == shadow.clean_count
    assert tagged.available_count == shadow.available_count


def test_missed_boot_event_waits_fail_closed_until_watchdog():
    scenes = 70
    restart = np.zeros(scenes, dtype=bool)
    restart[2] = True
    faults = HybridFaultSchedule(
        base=FaultSchedule(
            restart=restart,
            cold_restart=np.zeros(scenes, dtype=bool),
            forward_delivered=np.ones(scenes, dtype=bool),
            feedback_delivered=np.ones(scenes, dtype=bool),
        ),
        boot_emitter_available=np.zeros(scenes, dtype=bool),
    )
    result = simulate_tagged_hbw30_trajectory(
        _trace(scenes), faults, bits=_bits(), restore_tag_bits=16
    )
    assert result.identity_sync_wait_scenes > 0
    assert result.wrong_context_execution_count == 0
    assert result.stale_restore_acceptance_count == 0
    assert result.recovery_latencies
    assert max(result.recovery_latencies) <= 30


def test_lost_ordinary_update_does_not_invalidate_cached_action():
    scenes = 20
    forward = np.ones(scenes, dtype=bool)
    forward[5] = False
    faults = HybridFaultSchedule(
        base=FaultSchedule(
            restart=np.zeros(scenes, dtype=bool),
            cold_restart=np.zeros(scenes, dtype=bool),
            forward_delivered=forward,
            feedback_delivered=np.ones(scenes, dtype=bool),
        ),
        boot_emitter_available=np.ones(scenes, dtype=bool),
    )
    result = simulate_tagged_hbw30_trajectory(
        _trace(scenes), faults, bits=_bits(), restore_tag_bits=16
    )
    assert result.available_count == scenes
    assert result.wrong_context_execution_count == 0
