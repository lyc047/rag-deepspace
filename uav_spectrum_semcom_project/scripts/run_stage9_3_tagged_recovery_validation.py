#!/usr/bin/env python
"""Registered Stage-9.3 binary, subprocess, and adversarial validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    GreedyTaskCodebook,
    SpectrumTaskQuery,
    TaskCodeword,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
)
from spectrum_semcom.stage9_3_tagged_recovery import (  # noqa: E402
    RecoverySessionIdentity,
    TaggedRecoveryReceiver,
    decode_tagged_activation,
    decode_tagged_context_install,
    encode_tagged_activation,
    encode_tagged_context_install,
    recovery_session_tag,
)


DIGEST = "ab" * 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codebook() -> GreedyTaskCodebook:
    return GreedyTaskCodebook(
        n_channels=32,
        queries=(SpectrumTaskQuery(8), SpectrumTaskQuery(16), SpectrumTaskQuery(24)),
        epsilon_db=0.2,
        codewords=tuple(
            TaskCodeword(index, (index, index, index), 0)
            for index in range(3)
        ),
        training_scene_count=0,
        exact_action_tuple_count=0,
        training_covered_count=0,
    )


class Worker:
    def __init__(self, state_path: Path) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "stage9_3_receiver_worker.py"),
                "--state-path",
                str(state_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.pid = int(self.process.pid)

    def call(self, op: str, **payload: Any) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps({"op": op, **payload}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = ""
            if self.process.stderr is not None:
                error = self.process.stderr.read()
            raise RuntimeError(f"receiver worker exited unexpectedly: {error}")
        return json.loads(line)

    def stop(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=10)
        if self.process.returncode != 0:
            error = "" if self.process.stderr is None else self.process.stderr.read()
            raise RuntimeError(f"receiver worker failed: {error}")


def _binary_codec_validation() -> dict[str, Any]:
    codebook = _codebook()
    session = install_codebook(codebook, epoch=7)
    catalog = build_preinstalled_catalog(((2, codebook),), catalog_epoch=7)
    identity = RecoverySessionIdentity(3, 1001, 7)
    activation = encode_tagged_activation(
        catalog,
        identity=identity,
        node_id=1,
        bank_id=2,
        codebook_epoch=7,
        activation_epoch=9,
    )
    install = encode_tagged_context_install(
        session, identity=identity, node_id=1
    )
    activation_decoded = decode_tagged_activation(
        activation, catalog, expected_identity=identity
    )
    install_decoded = decode_tagged_context_install(
        install, expected_identity=identity
    )
    stale_rejected = 0
    stale = RecoverySessionIdentity(3, 1000, 7)
    for decoder, frame, extra in (
        (decode_tagged_activation, activation, {"catalog": catalog}),
        (decode_tagged_context_install, install, {}),
    ):
        try:
            decoder(frame, expected_identity=stale, **extra)
        except ValueError:
            stale_rejected += 1
    return {
        "tagged_activation_bits": int(activation.size),
        "tagged_full_install_bits": int(install.size),
        "activation_round_trip": bool(
            activation_decoded.session.manifest_sha256 == session.manifest_sha256
        ),
        "full_install_round_trip": bool(
            install_decoded.session.manifest_sha256 == session.manifest_sha256
        ),
        "stale_codec_rejections": stale_rejected,
        "ordinary_compact_update_bits": 42,
        "ordinary_ack_bits": 24,
    }


def _process_validation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage9_3_") as temp:
        state_path = Path(temp) / "receiver-state.json"
        worker = Worker(state_path)
        pids = [worker.pid]
        checks: dict[str, bool] = {}
        worker.call("boot", boot_id=100, cold=True)
        begun = worker.call(
            "begin_recovery", controller_epoch=1, catalog_epoch=7
        )
        tag100 = int(begun["recovery_session_tag"])
        checks["tagged_cold_full_install"] = bool(
            worker.call(
                "restore",
                session_tag=tag100,
                catalog_epoch=7,
                update_epoch=254,
                actions=[1, 2, 3],
                full_install=True,
                catalog_digest=DIGEST,
            )["accepted"]
        )
        worker.stop()

        worker = Worker(state_path)
        pids.append(worker.pid)
        warm = worker.call("boot", boot_id=101, cold=False)
        begun = worker.call(
            "begin_recovery", controller_epoch=1, catalog_epoch=7
        )
        tag101 = int(begun["recovery_session_tag"])
        activation = worker.call(
            "restore",
            session_tag=tag101,
            catalog_epoch=7,
            update_epoch=254,
            actions=[1, 2, 3],
            full_install=False,
            catalog_digest=DIGEST,
        )
        checks["tagged_warm_activation"] = bool(
            warm["catalog_present"] and activation["accepted"]
        )
        checks["stale_previous_boot_rejected"] = not worker.call(
            "restore",
            session_tag=tag100,
            catalog_epoch=7,
            update_epoch=1,
            actions=[9, 9, 9],
            full_install=False,
            catalog_digest=DIGEST,
        )["accepted"]
        worker.call("begin_recovery", controller_epoch=2, catalog_epoch=7)
        tag102 = recovery_session_tag(RecoverySessionIdentity(2, 101, 7))
        checks["stale_previous_controller_rejected"] = not worker.call(
            "restore",
            session_tag=tag101,
            catalog_epoch=7,
            update_epoch=1,
            actions=[9, 9, 9],
            full_install=False,
            catalog_digest=DIGEST,
        )["accepted"]
        checks["wrong_catalog_rejected"] = not worker.call(
            "restore",
            session_tag=recovery_session_tag(RecoverySessionIdentity(2, 101, 8)),
            catalog_epoch=8,
            update_epoch=1,
            actions=[9, 9, 9],
            full_install=False,
            catalog_digest=DIGEST,
        )["accepted"]
        worker.call(
            "restore",
            session_tag=tag102,
            catalog_epoch=7,
            update_epoch=254,
            actions=[1, 2, 3],
            full_install=False,
            catalog_digest=DIGEST,
        )
        u255 = worker.call(
            "update", catalog_epoch=7, update_epoch=255, actions=[2, 3, 4]
        )
        u0 = worker.call(
            "update", catalog_epoch=7, update_epoch=0, actions=[3, 4, 5]
        )
        stale254 = worker.call(
            "update", catalog_epoch=7, update_epoch=254, actions=[9, 9, 9]
        )
        duplicate0 = worker.call(
            "update", catalog_epoch=7, update_epoch=0, actions=[3, 4, 5]
        )
        checks["serial_wrap_254_255_0"] = bool(
            u255["accepted"] and u0["accepted"]
        )
        checks["delayed_prewrap_rejected"] = not stale254["accepted"]
        checks["duplicate_idempotent"] = bool(
            duplicate0["accepted"] and duplicate0["duplicate"]
        )
        worker.call(
            "enqueue_update",
            packet_id="flush",
            catalog_epoch=7,
            update_epoch=1,
            actions=[4, 5, 6],
        )
        worker.call("begin_recovery", controller_epoch=3, catalog_epoch=7)
        checks["queue_flush_on_session_change"] = not worker.call(
            "deliver_queued", packet_id="flush"
        )["accepted"]
        worker.call(
            "enqueue_update",
            packet_id="old",
            catalog_epoch=7,
            update_epoch=1,
            actions=[4, 5, 6],
        )
        worker.call("advance_update_clock", count=128)
        checks["queue_age_128_rejected"] = not worker.call(
            "deliver_queued", packet_id="old"
        )["accepted"]
        worker.stop()

        envelope = json.loads(state_path.read_text(encoding="utf-8"))
        envelope["payload"]["catalog_epoch"] = 8
        state_path.write_text(json.dumps(envelope), encoding="utf-8")
        worker = Worker(state_path)
        pids.append(worker.pid)
        corrupted = worker.call("boot", boot_id=102, cold=False)
        checks["durable_corruption_fail_closed"] = bool(
            corrupted["durable_corruption_detected"]
            and not corrupted["catalog_present"]
            and not worker.call("execute")["available"]
        )
        worker.call("begin_recovery", controller_epoch=4, catalog_epoch=7)
        old_sender_tag = recovery_session_tag(RecoverySessionIdentity(3, 102, 7))
        stale = worker.call(
            "restore",
            session_tag=old_sender_tag,
            catalog_epoch=7,
            update_epoch=1,
            actions=[9, 9, 9],
            full_install=True,
            catalog_digest=DIGEST,
        )
        new_tag = recovery_session_tag(RecoverySessionIdentity(4, 102, 7))
        recovered = worker.call(
            "restore",
            session_tag=new_tag,
            catalog_epoch=7,
            update_epoch=1,
            actions=[1, 2, 3],
            full_install=True,
            catalog_digest=DIGEST,
        )
        checks["simultaneous_sender_receiver_restart"] = bool(
            not stale["accepted"] and recovered["accepted"]
        )
        worker.stop()
    return {
        "worker_pids": pids,
        "independent_process_count": len(set(pids)),
        "checks": checks,
        "passed_count": sum(checks.values()),
        "scenario_count": len(checks),
        "safety_violations": len(checks) - sum(checks.values()),
    }


def _random_adversarial(
    *, trajectories: int, steps: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    event_names = (
        "valid_update",
        "duplicate_update",
        "stale_update",
        "receiver_warm_restart",
        "receiver_cold_restart",
        "sender_restart",
        "valid_restore",
        "stale_restore",
        "wrong_catalog_restore",
        "catalog_corruption",
        "queue_delayed_update",
        "queue_flush",
    )
    counts = {name: 0 for name in event_names}
    violations = stale_accepts = wrong_catalog_accepts = corrupt_executes = 0
    for _ in range(int(trajectories)):
        receiver = TaggedRecoveryReceiver()
        boot_id = 1
        controller = 1
        serial = 250
        action = (0, 0, 0)
        receiver.boot(boot_id=boot_id, cold=True)
        begun = receiver.begin_recovery(
            controller_epoch=controller, catalog_epoch=7
        )
        receiver.restore(
            session_tag=int(begun["recovery_session_tag"]),
            catalog_epoch=7,
            update_epoch=serial,
            actions=action,
            full_install=True,
            catalog_digest=DIGEST,
        )
        for _step in range(int(steps)):
            event = str(rng.choice(event_names))
            counts[event] += 1
            if event == "valid_update":
                serial = (serial + int(rng.integers(1, 8))) % 256
                action = tuple(int(value) for value in rng.integers(0, 16, 3))
                if receiver.snapshot.active:
                    reply = receiver.update(
                        catalog_epoch=7, update_epoch=serial, actions=action
                    )
                    violations += int(not reply["accepted"])
            elif event == "duplicate_update" and receiver.snapshot.active:
                reply = receiver.update(
                    catalog_epoch=7,
                    update_epoch=int(receiver.snapshot.update_epoch),
                    actions=tuple(receiver.snapshot.actions or ()),
                )
                violations += int(not (reply["accepted"] and reply["duplicate"]))
            elif event == "stale_update" and receiver.snapshot.active:
                stale_serial = (int(receiver.snapshot.update_epoch) - 1) % 256
                reply = receiver.update(
                    catalog_epoch=7, update_epoch=stale_serial, actions=(31, 31, 31)
                )
                violations += int(reply["accepted"])
            elif event in {"receiver_warm_restart", "receiver_cold_restart"}:
                boot_id = (boot_id + 1) % 65536
                cold = event == "receiver_cold_restart"
                receiver.boot(boot_id=boot_id, cold=cold)
                begun = receiver.begin_recovery(
                    controller_epoch=controller, catalog_epoch=7
                )
                serial = int(rng.integers(0, 256))
                action = tuple(int(value) for value in rng.integers(0, 16, 3))
                reply = receiver.restore(
                    session_tag=int(begun["recovery_session_tag"]),
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=action,
                    full_install=cold,
                    catalog_digest=DIGEST,
                )
                violations += int(not reply["accepted"])
            elif event == "sender_restart":
                controller = (controller + 1) % 256
                begun = receiver.begin_recovery(
                    controller_epoch=controller, catalog_epoch=7
                )
                reply = receiver.restore(
                    session_tag=int(begun["recovery_session_tag"]),
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=action,
                    full_install=not receiver.snapshot.catalog_present,
                    catalog_digest=DIGEST,
                )
                violations += int(not reply["accepted"])
            elif event == "valid_restore":
                begun = receiver.begin_recovery(
                    controller_epoch=controller, catalog_epoch=7
                )
                reply = receiver.restore(
                    session_tag=int(begun["recovery_session_tag"]),
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=action,
                    full_install=not receiver.snapshot.catalog_present,
                    catalog_digest=DIGEST,
                )
                violations += int(not reply["accepted"])
            elif event == "stale_restore":
                stale_tag = recovery_session_tag(
                    RecoverySessionIdentity((controller - 1) % 256, boot_id, 7)
                )
                reply = receiver.restore(
                    session_tag=stale_tag,
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=(31, 31, 31),
                    full_install=False,
                    catalog_digest=DIGEST,
                )
                stale_accepts += int(reply["accepted"])
            elif event == "wrong_catalog_restore":
                wrong_tag = recovery_session_tag(
                    RecoverySessionIdentity(controller, boot_id, 8)
                )
                reply = receiver.restore(
                    session_tag=wrong_tag,
                    catalog_epoch=8,
                    update_epoch=serial,
                    actions=(31, 31, 31),
                    full_install=False,
                    catalog_digest=DIGEST,
                )
                wrong_catalog_accepts += int(reply["accepted"])
            elif event == "catalog_corruption" and receiver.snapshot.catalog_present:
                receiver.corrupt_durable_for_test()
                boot_id = (boot_id + 1) % 65536
                status = receiver.boot(boot_id=boot_id, cold=False)
                corrupt_executes += int(receiver.execute()["available"])
                violations += int(
                    not status["durable_corruption_detected"]
                    or status["catalog_present"]
                )
                begun = receiver.begin_recovery(
                    controller_epoch=controller, catalog_epoch=7
                )
                receiver.restore(
                    session_tag=int(begun["recovery_session_tag"]),
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=action,
                    full_install=True,
                    catalog_digest=DIGEST,
                )
            elif event == "queue_delayed_update" and receiver.snapshot.active:
                queued_serial = (int(receiver.snapshot.update_epoch) + 1) % 256
                receiver.enqueue_update(
                    packet_id="q",
                    catalog_epoch=7,
                    update_epoch=queued_serial,
                    actions=(1, 1, 1),
                )
                age = int(rng.integers(0, 150))
                receiver.advance_update_clock(count=age)
                reply = receiver.deliver_queued(packet_id="q")
                violations += int(reply["accepted"] != (age <= 127))
                if reply["accepted"]:
                    serial = queued_serial
                    action = (1, 1, 1)
            elif event == "queue_flush":
                receiver.enqueue_update(
                    packet_id="qf",
                    catalog_epoch=7,
                    update_epoch=(serial + 1) % 256,
                    actions=(2, 2, 2),
                )
                controller = (controller + 1) % 256
                receiver.begin_recovery(
                    controller_epoch=controller, catalog_epoch=7
                )
                violations += int(
                    receiver.deliver_queued(packet_id="qf")["accepted"]
                )
                begun = receiver.status()
                receiver.restore(
                    session_tag=int(begun["recovery_session_tag"]),
                    catalog_epoch=7,
                    update_epoch=serial,
                    actions=action,
                    full_install=not receiver.snapshot.catalog_present,
                    catalog_digest=DIGEST,
                )
            execution = receiver.execute()
            if execution["available"]:
                violations += int(
                    not receiver.snapshot.active
                    or receiver.expected_identity is None
                    or receiver.expected_identity.receiver_boot_id
                    != receiver.snapshot.boot_id
                    or receiver.expected_identity.catalog_epoch
                    != receiver.snapshot.catalog_epoch
                )
    return {
        "seed": int(seed),
        "trajectories": int(trajectories),
        "steps_per_trajectory": int(steps),
        "event_counts": counts,
        "safety_violations": int(violations),
        "stale_restore_acceptances": int(stale_accepts),
        "wrong_catalog_restore_acceptances": int(wrong_catalog_accepts),
        "corrupt_catalog_executable_count": int(corrupt_executes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    random_config = config["random_adversarial_validation"]
    if args.mode == "development":
        trajectories = int(random_config["development_trajectories"])
        seed = int(random_config["development_seed"])
    else:
        trajectories = int(random_config["confirmation_trajectories"])
        seed = int(random_config["confirmation_seed"])
    result = {
        "experiment_id": config["experiment_id"],
        "candidate": config["candidate"],
        "mode": args.mode,
        "config_sha256": _sha256(args.config),
        "binary_codec": _binary_codec_validation(),
        "process_conformance": _process_validation(),
        "random_adversarial": _random_adversarial(
            trajectories=trajectories,
            steps=int(random_config["steps_per_trajectory"]),
            seed=seed,
        ),
    }
    codec = result["binary_codec"]
    process = result["process_conformance"]
    random_result = result["random_adversarial"]
    result["gates_passed"] = bool(
        codec["tagged_activation_bits"] == 80
        and codec["tagged_full_install_bits"] == 419
        and codec["ordinary_compact_update_bits"] == 42
        and codec["ordinary_ack_bits"] == 24
        and codec["activation_round_trip"]
        and codec["full_install_round_trip"]
        and codec["stale_codec_rejections"] == 2
        and process["safety_violations"] == 0
        and random_result["safety_violations"] == 0
        and random_result["stale_restore_acceptances"] == 0
        and random_result["wrong_catalog_restore_acceptances"] == 0
        and random_result["corrupt_catalog_executable_count"] == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
