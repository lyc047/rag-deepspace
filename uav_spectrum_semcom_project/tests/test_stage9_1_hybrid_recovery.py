from __future__ import annotations

import numpy as np

from spectrum_semcom.stage9_1_hybrid_recovery import (
    HardenedPersistentReceiver,
    HybridFaultSchedule,
    simulate_hybrid_trajectory,
)
from spectrum_semcom.stage9_sil_runtime import (
    FaultSchedule,
    SilProtocolBits,
    TaskTrace,
)


def _bits() -> SilProtocolBits:
    return SilProtocolBits(42, 24, 32, 32, 64, 64, 62, 403)


def test_hardened_receiver_rejects_stale_and_wrong_epoch(tmp_path):
    receiver = HardenedPersistentReceiver(tmp_path / "state.json")
    receiver.boot(boot_id=1, cold=True)
    assert receiver.restore(
        catalog_epoch=3,
        update_epoch=10,
        actions=(1,),
        full_install=True,
    )["accepted"]
    assert receiver.update(
        catalog_epoch=3, update_epoch=11, actions=(2,)
    )["accepted"]
    assert not receiver.update(
        catalog_epoch=3, update_epoch=10, actions=(0,)
    )["accepted"]
    assert not receiver.update(
        catalog_epoch=4, update_epoch=12, actions=(0,)
    )["accepted"]
    assert receiver.execute()["actions"] == [2]


def test_watchdog_recovers_when_boot_emitter_is_unavailable():
    scenes = 70
    trace = TaskTrace(
        desired_symbol=np.zeros(scenes, dtype=np.int64),
        acceptable_symbols=tuple(frozenset({0}) for _ in range(scenes)),
    )
    restart = np.zeros(scenes, dtype=bool)
    restart[2] = True
    base = FaultSchedule(
        restart=restart,
        cold_restart=np.zeros(scenes, dtype=bool),
        forward_delivered=np.ones(scenes, dtype=bool),
        feedback_delivered=np.ones(scenes, dtype=bool),
    )
    faults = HybridFaultSchedule(
        base=base,
        boot_emitter_available=np.zeros(scenes, dtype=bool),
    )
    event = simulate_hybrid_trajectory(
        trace, faults, policy="boot_event", bits=_bits()
    )
    watchdog = simulate_hybrid_trajectory(
        trace, faults, policy="boot_event_watchdog30", bits=_bits()
    )
    assert watchdog.clean_count > event.clean_count
    assert watchdog.wrong_context_execution_count == 0


def test_watchdog30_has_no_slower_recovery_than_watchdog60():
    scenes = 100
    trace = TaskTrace(
        desired_symbol=np.zeros(scenes, dtype=np.int64),
        acceptable_symbols=tuple(frozenset({0}) for _ in range(scenes)),
    )
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
    w30 = simulate_hybrid_trajectory(
        trace, faults, policy="boot_event_watchdog30", bits=_bits()
    )
    w60 = simulate_hybrid_trajectory(
        trace, faults, policy="boot_event_watchdog60", bits=_bits()
    )
    assert max(w30.recovery_latencies) <= max(w60.recovery_latencies)
