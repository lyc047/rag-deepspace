from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from spectrum_semcom.stage9_sil_runtime import (
    FaultSchedule,
    PersistentReceiver,
    SilProtocolBits,
    TaskTrace,
    simulate_sil_trajectory,
)


def _bits() -> SilProtocolBits:
    return SilProtocolBits(42, 24, 32, 32, 64, 64, 62, 403)


def test_receiver_is_fail_closed_across_warm_and_cold_boot(tmp_path):
    path = tmp_path / "receiver.json"
    receiver = PersistentReceiver(path)
    assert not receiver.boot(boot_id=1, cold=True)["active"]
    assert receiver.restore(
        catalog_epoch=2,
        update_epoch=1,
        actions=(1,),
        full_install=True,
    )["accepted"]
    assert receiver.execute()["actions"] == [1]
    warm = receiver.boot(boot_id=2, cold=False)
    assert warm["catalog_present"] and not warm["active"]
    assert receiver.restore(
        catalog_epoch=2,
        update_epoch=2,
        actions=(2,),
        full_install=False,
    )["accepted"]
    cold = receiver.boot(boot_id=3, cold=True)
    assert not cold["catalog_present"] and not cold["active"]
    assert not receiver.restore(
        catalog_epoch=2,
        update_epoch=3,
        actions=(0,),
        full_install=False,
    )["accepted"]


def test_boot_event_recovers_before_fixed_heartbeat():
    scenes = 12
    trace = TaskTrace(
        desired_symbol=np.zeros(scenes, dtype=np.int64),
        acceptable_symbols=tuple(frozenset({0}) for _ in range(scenes)),
    )
    restart = np.zeros(scenes, dtype=bool)
    restart[2] = True
    faults = FaultSchedule(
        restart=restart,
        cold_restart=np.zeros(scenes, dtype=bool),
        forward_delivered=np.ones(scenes, dtype=bool),
        feedback_delivered=np.ones(scenes, dtype=bool),
    )
    event = simulate_sil_trajectory(trace, faults, policy="boot_event", bits=_bits())
    fixed = simulate_sil_trajectory(trace, faults, policy="fixed10", bits=_bits())
    assert event.clean_count > fixed.clean_count
    assert event.wrong_context_execution_count == 0
    assert fixed.wrong_context_execution_count == 0


def test_subprocess_worker_matches_receiver_lifecycle(tmp_path):
    worker = subprocess.Popen(
        [
            sys.executable,
            "scripts/stage9_receiver_worker.py",
            "--state-path",
            str(tmp_path / "worker.json"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert worker.stdin is not None and worker.stdout is not None

    def exchange(payload):
        worker.stdin.write(json.dumps(payload) + "\n")
        worker.stdin.flush()
        return json.loads(worker.stdout.readline())

    assert not exchange({"op": "boot", "boot_id": 1, "cold": True})[
        "catalog_present"
    ]
    restored = exchange(
        {
            "op": "restore",
            "catalog_epoch": 1,
            "update_epoch": 1,
            "actions": [2],
            "full_install": True,
        }
    )
    assert restored["accepted"]
    assert exchange({"op": "execute"})["actions"] == [2]
    warm = exchange({"op": "boot", "boot_id": 2, "cold": False})
    assert warm["catalog_present"] and not warm["active"]
    worker.terminate()
    worker.wait(timeout=5)
