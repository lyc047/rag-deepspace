#!/usr/bin/env python
"""Run real OS-process lifecycle conformance for Stage-9.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class WorkerClient:
    def __init__(self, state_path: Path) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/stage9_1_receiver_worker.py"),
                "--state-path",
                str(state_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.pid = int(self.process.pid)

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = (
                self.process.stderr.read()
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(f"worker stopped without response: {error}")
        return json.loads(line)

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=5)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--state-path",
        default="results/stage9_1/process_fault_conformance_v1/receiver_state.json",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    state_path = ROOT / args.state_path
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)

    events: list[dict[str, Any]] = []
    pids: list[int] = []
    safety_violations = 0
    started = time.perf_counter()

    def record(name: str, response: dict[str, Any] | None) -> None:
        events.append({"event": name, "response": response})

    worker = WorkerClient(state_path)
    pids.append(worker.pid)
    cold = worker.exchange({"op": "boot", "boot_id": 1, "cold": True})
    record("cold_process_start", cold)
    safety_violations += int(cold["active"] or cold["catalog_present"])
    installed = worker.exchange(
        {
            "op": "restore",
            "catalog_epoch": 3,
            "update_epoch": 1,
            "actions": [0],
            "full_install": True,
        }
    )
    record("full_install", installed)
    safety_violations += int(not installed["accepted"])
    update = worker.exchange(
        {
            "op": "update",
            "catalog_epoch": 3,
            "update_epoch": 2,
            "actions": [1],
        }
    )
    record("ordinary_update", update)
    safety_violations += int(not update["accepted"])

    record("network_proxy_dropped_update_epoch_3", None)
    unchanged = worker.exchange({"op": "execute"})
    record("execute_after_drop", unchanged)
    safety_violations += int(unchanged["actions"] != [1])
    update4 = worker.exchange(
        {
            "op": "update",
            "catalog_epoch": 3,
            "update_epoch": 4,
            "actions": [2],
        }
    )
    record("update_epoch_4", update4)
    delayed = worker.exchange(
        {
            "op": "update",
            "catalog_epoch": 3,
            "update_epoch": 3,
            "actions": [0],
        }
    )
    record("delayed_update_epoch_3", delayed)
    safety_violations += int(delayed["accepted"])
    wrong_epoch = worker.exchange(
        {
            "op": "update",
            "catalog_epoch": 4,
            "update_epoch": 5,
            "actions": [0],
        }
    )
    record("wrong_catalog_epoch", wrong_epoch)
    safety_violations += int(wrong_epoch["accepted"])
    worker.stop()
    record("warm_process_kill", {"pid": pids[-1]})

    worker = WorkerClient(state_path)
    pids.append(worker.pid)
    warm = worker.exchange({"op": "boot", "boot_id": 2, "cold": False})
    record("warm_process_restart", warm)
    safety_violations += int(
        not warm["catalog_present"] or warm["active"]
    )
    unavailable = worker.exchange({"op": "execute"})
    record("fail_closed_after_warm_restart", unavailable)
    safety_violations += int(unavailable["available"])
    activated = worker.exchange(
        {
            "op": "restore",
            "catalog_epoch": 3,
            "update_epoch": 5,
            "actions": [2],
            "full_install": False,
        }
    )
    record("activation_restore", activated)
    safety_violations += int(not activated["accepted"])
    worker.stop()
    record("cold_process_kill", {"pid": pids[-1]})

    worker = WorkerClient(state_path)
    pids.append(worker.pid)
    cold_again = worker.exchange({"op": "boot", "boot_id": 3, "cold": True})
    record("cold_process_restart", cold_again)
    safety_violations += int(
        cold_again["catalog_present"] or cold_again["active"]
    )
    rejected_activation = worker.exchange(
        {
            "op": "restore",
            "catalog_epoch": 3,
            "update_epoch": 6,
            "actions": [1],
            "full_install": False,
        }
    )
    record("activation_without_catalog", rejected_activation)
    safety_violations += int(rejected_activation["accepted"])
    reinstalled = worker.exchange(
        {
            "op": "restore",
            "catalog_epoch": 3,
            "update_epoch": 6,
            "actions": [1],
            "full_install": True,
        }
    )
    record("full_reinstall", reinstalled)
    safety_violations += int(not reinstalled["accepted"])
    worker.stop()

    result = {
        "version": "1.0",
        "experiment_id": "stage9_1_process_fault_conformance_v1",
        "status": "PASS" if safety_violations == 0 else "FAIL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "real_os_processes_used": True,
        "process_ids": pids,
        "distinct_process_ids": len(set(pids)),
        "network_proxy_drop_count": 1,
        "safety_violation_count": safety_violations,
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
        "events": events,
        "input_hashes": {
            "worker": _sha256(ROOT / "scripts/stage9_1_receiver_worker.py"),
            "runtime": _sha256(
                ROOT / "src/spectrum_semcom/stage9_1_hybrid_recovery.py"
            ),
            "runner": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": "OS-process lifecycle conformance only; not a real board, RF link, or failure-rate estimate.",
    }
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "pids": pids}))


if __name__ == "__main__":
    main()
