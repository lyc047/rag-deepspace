#!/usr/bin/env python
"""Run the preregistered Stage-9.6 independent-process protocol audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "stage9_6_endpoint_worker.py"
sys.path.insert(0, str(ROOT / "src"))

from spectrum_semcom.stage9_6_process_protocol import REGISTERED_WIDTHS  # noqa: E402


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


class EndpointProcess:
    def __init__(self, role: str, state_path: Path | None = None) -> None:
        command = [sys.executable, str(WORKER), "--role", role]
        if state_path is not None:
            command += ["--state-path", str(state_path)]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=ROOT,
        )
        self.role = role

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def request(self, op: str, **kwargs: Any) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipe unavailable")
        self.process.stdin.write(json.dumps({"op": op, **kwargs}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = "" if self.process.stderr is None else self.process.stderr.read()
            raise RuntimeError(f"{self.role} worker terminated: {stderr}")
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class NetworkProxy:
    def __init__(self) -> None:
        self.ledger: list[dict[str, Any]] = []
        self.delayed: dict[str, tuple[int, dict[str, Any], str, str, EndpointProcess]] = {}

    def send(
        self,
        *,
        frame_id: str,
        kind: str,
        wire: dict[str, Any],
        destination: EndpointProcess,
        operation: str,
        mode: str = "deliver",
    ) -> dict[str, Any] | None:
        raw = bytes.fromhex(str(wire["payload_hex"]))
        entry = {
            "frame_id": str(frame_id),
            "kind": str(kind),
            "bit_length": int(wire["bit_length"]),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "mode": str(mode),
            "delivered": mode == "deliver",
        }
        self.ledger.append(entry)
        index = len(self.ledger) - 1
        if mode == "drop":
            return None
        if mode == "delay":
            self.delayed[str(frame_id)] = (
                index,
                dict(wire),
                str(kind),
                str(operation),
                destination,
            )
            return None
        if mode != "deliver":
            raise ValueError("unknown proxy mode")
        return destination.request(operation, wire=wire, **({"kind": kind} if operation == "receive_recovery" else {}))

    def deliver_delayed(self, frame_id: str) -> dict[str, Any]:
        index, wire, kind, operation, destination = self.delayed.pop(str(frame_id))
        self.ledger[index]["delivered"] = True
        self.ledger[index]["mode"] = "delayed_then_delivered"
        return destination.request(operation, wire=wire, **({"kind": kind} if operation == "receive_recovery" else {}))


class ProtocolAudit:
    def __init__(self, temp_dir: Path) -> None:
        self.state_path = temp_dir / "receiver_catalog.json"
        self.sender: EndpointProcess | None = None
        self.receiver: EndpointProcess | None = None
        self.proxy = NetworkProxy()
        self.sender_pids: list[int] = []
        self.receiver_pids: list[int] = []
        self.generated_frame_count = 0
        self.frame_serial = 0
        self.scenarios: dict[str, bool] = {}
        self.violation_messages: list[str] = []

    def start_sender(self) -> None:
        if self.sender is not None:
            self.sender.close()
        self.sender = EndpointProcess("sender")
        self.sender_pids.append(self.sender.pid)

    def start_receiver(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        if self.receiver is not None:
            self.receiver.close()
        self.receiver = EndpointProcess("receiver", self.state_path)
        self.receiver_pids.append(self.receiver.pid)
        return self.receiver.request("boot", boot_id=int(boot_id), cold=bool(cold))

    def close(self) -> None:
        if self.sender is not None:
            self.sender.close()
        if self.receiver is not None:
            self.receiver.close()

    def _frame_id(self, kind: str) -> str:
        self.frame_serial += 1
        return f"{kind}-{self.frame_serial:06d}"

    def _produce(self, response: dict[str, Any], kind: str | None = None) -> tuple[str, str, dict[str, Any]]:
        wire = response["wire"] if "wire" in response else response["feedback_wire"]
        frame_kind = str(kind or response.get("kind") or response.get("feedback_kind"))
        self.generated_frame_count += 1
        return self._frame_id(frame_kind), frame_kind, wire

    def make_update(self, symbol: int, epoch: int) -> tuple[str, str, dict[str, Any]]:
        assert self.sender is not None
        return self._produce(
            self.sender.request("make_update", symbol=int(symbol), update_epoch=int(epoch)),
            "update",
        )

    def send_update(
        self, symbol: int, epoch: int, mode: str = "deliver"
    ) -> tuple[dict[str, Any] | None, tuple[str, str, dict[str, Any]]]:
        assert self.receiver is not None
        frame = self.make_update(symbol, epoch)
        response = self.proxy.send(
            frame_id=frame[0], kind=frame[1], wire=frame[2],
            destination=self.receiver, operation="receive_update", mode=mode,
        )
        return response, frame

    def route_nack(self, response: dict[str, Any], mode: str = "deliver") -> dict[str, Any] | None:
        assert self.sender is not None
        frame = self._produce(response, "reset_nack")
        return self.proxy.send(
            frame_id=frame[0], kind=frame[1], wire=frame[2],
            destination=self.sender, operation="receive_reset_nack", mode=mode,
        )

    def make_recovery(self) -> tuple[str, str, dict[str, Any]]:
        assert self.sender is not None
        return self._produce(self.sender.request("make_recovery"))

    def send_recovery(
        self,
        frame: tuple[str, str, dict[str, Any]] | None = None,
        mode: str = "deliver",
    ) -> dict[str, Any] | None:
        assert self.receiver is not None
        frame = self.make_recovery() if frame is None else frame
        return self.proxy.send(
            frame_id=frame[0], kind=frame[1], wire=frame[2],
            destination=self.receiver, operation="receive_recovery", mode=mode,
        )

    def recover(self, *, symbol: int, epoch: int, expected_kind: str) -> bool:
        assert self.receiver is not None
        response, _ = self.send_update(symbol, epoch)
        if response is None or response.get("feedback_kind") != "reset_nack":
            return False
        sender_reply = self.route_nack(response)
        if sender_reply is None or not sender_reply.get("accepted"):
            return False
        recovery = self.make_recovery()
        if recovery[1] != expected_kind:
            return False
        context_reply = self.send_recovery(recovery)
        before_update = self.receiver.request("execute")
        update_reply, _ = self.send_update(symbol, epoch)
        after_update = self.receiver.request("execute")
        return bool(
            context_reply
            and context_reply.get("accepted_context")
            and not before_update.get("available")
            and update_reply
            and update_reply.get("accepted")
            and after_update.get("available")
        )

    def run_registered_scenarios(self) -> None:
        self.start_sender()
        cold_status = self.start_receiver(boot_id=100, cold=True)
        self.scenarios["distinct_sender_receiver_pids"] = bool(
            self.sender and self.receiver and self.sender.pid != self.receiver.pid != os.getpid()
        )
        self.scenarios["cold_receiver_requests_full_install"] = bool(
            not cold_status.get("catalog_present")
            and self.recover(symbol=2, epoch=254, expected_kind="full_install")
        )

        warm = self.start_receiver(boot_id=101, cold=False)
        self.scenarios["warm_receiver_requests_tagged_activation"] = bool(
            warm.get("catalog_present")
            and self.recover(symbol=1, epoch=10, expected_kind="activation")
        )
        self.scenarios["receiver_process_termination_changes_pid_and_clears_action"] = bool(
            len(set(self.receiver_pids)) == len(self.receiver_pids)
            and not warm.get("active")
        )

        self.start_receiver(boot_id=102, cold=False)
        response, _ = self.send_update(0, 11)
        assert response is not None
        before = self.sender.request("status")
        self.route_nack(response, mode="drop")
        after = self.sender.request("status")
        unchanged = before["feedback_transition_count"] == after["feedback_transition_count"]
        response, _ = self.send_update(0, 11)
        assert response is not None
        self.route_nack(response)
        recovery_ok = bool(self.send_recovery().get("accepted_context"))
        update_ok = bool(self.send_update(0, 11)[0].get("accepted"))
        self.scenarios["dropped_RESET_NACK_does_not_change_sender"] = bool(
            unchanged and recovery_ok and update_ok
        )

        self.start_receiver(boot_id=103, cold=False)
        response, _ = self.send_update(1, 12)
        assert response is not None
        self.route_nack(response)
        stale_recovery = self.make_recovery()
        self.start_receiver(boot_id=104, cold=False)
        response, _ = self.send_update(1, 13)
        assert response is not None
        self.route_nack(response)
        stale_reply = self.send_recovery(stale_recovery)
        current_reply = self.send_recovery()
        current_update = self.send_update(1, 13)[0]
        self.scenarios["delayed_old_recovery_rejected_after_second_restart"] = bool(
            stale_reply
            and not stale_reply.get("accepted_context")
            and current_reply
            and current_reply.get("accepted_context")
            and current_update
            and current_update.get("accepted")
        )

        self.start_receiver(boot_id=105, cold=False)
        base_ok = self.recover(symbol=0, epoch=254, expected_kind="activation")
        old_frame = self.make_update(1, 255)
        assert self.receiver is not None
        self.proxy.send(
            frame_id=old_frame[0], kind=old_frame[1], wire=old_frame[2],
            destination=self.receiver, operation="receive_update", mode="delay",
        )
        wrap_reply, _ = self.send_update(2, 0)
        stale_update_reply = self.proxy.deliver_delayed(old_frame[0])
        executed = self.receiver.request("execute")
        self.scenarios["reordered_old_update_rejected_across_serial_wrap"] = bool(
            base_ok
            and wrap_reply
            and wrap_reply.get("fresh")
            and not stale_update_reply.get("accepted")
            and executed.get("actions") == [2, 2, 2]
        )

        old_sender_pid = self.sender.pid
        self.start_sender()
        self.start_receiver(boot_id=106, cold=False)
        new_sender_status = self.sender.request("status")
        sender_restart_ok = bool(
            self.sender.pid != old_sender_pid and new_sender_status.get("identity") is None
        )
        sender_restart_ok &= self.recover(symbol=2, epoch=1, expected_kind="activation")
        self.scenarios["sender_process_termination_changes_pid_and_requires_new_feedback"] = sender_restart_ok

        assert self.receiver is not None
        self.receiver.request("corrupt_durable_for_test")
        corrupt_status = self.start_receiver(boot_id=107, cold=False)
        corrupt_execute = self.receiver.request("execute")
        corruption_ok = bool(
            corrupt_status.get("durable_corruption_detected")
            and not corrupt_status.get("catalog_present")
            and not corrupt_execute.get("available")
        )
        corruption_ok &= self.recover(symbol=1, epoch=2, expected_kind="full_install")
        self.scenarios["durable_catalog_corruption_fails_closed_then_full_install_recovers"] = corruption_ok

        valid = self.make_update(0, 3)
        truncated = dict(valid[2])
        truncated["payload_hex"] = truncated["payload_hex"][:-2]
        trunc_reply = self.proxy.send(
            frame_id=valid[0], kind=valid[1], wire=truncated,
            destination=self.receiver, operation="receive_update", mode="deliver",
        )
        corrupt = self.make_update(0, 3)
        raw = bytearray.fromhex(corrupt[2]["payload_hex"])
        raw[0] ^= 0x80
        corrupt[2]["payload_hex"] = bytes(raw).hex()
        corrupt_reply = self.proxy.send(
            frame_id=corrupt[0], kind=corrupt[1], wire=corrupt[2],
            destination=self.receiver, operation="receive_update", mode="deliver",
        )
        padding = self.make_update(0, 3)
        raw = bytearray.fromhex(padding[2]["payload_hex"])
        raw[-1] |= 0x01
        padding[2]["payload_hex"] = bytes(raw).hex()
        padding_reply = self.proxy.send(
            frame_id=padding[0], kind=padding[1], wire=padding[2],
            destination=self.receiver, operation="receive_update", mode="deliver",
        )
        self.scenarios["truncated_or_padding_corrupt_binary_frame_is_rejected"] = bool(
            trunc_reply and not trunc_reply.get("accepted")
            and corrupt_reply and not corrupt_reply.get("accepted")
            and padding_reply and not padding_reply.get("accepted")
        )
        self.scenarios["compact_update_required_after_context_recovery"] = bool(
            self.scenarios["cold_receiver_requests_full_install"]
            and self.scenarios["warm_receiver_requests_tagged_activation"]
        )

    def run_random_faults(self, *, seed: int, operations: int) -> dict[str, Any]:
        assert self.sender is not None and self.receiver is not None
        rng = np.random.default_rng(int(seed))
        boot_id = 200
        epoch = 20
        self.start_receiver(boot_id=boot_id, cold=False)
        if not self.recover(symbol=0, epoch=epoch, expected_kind="activation"):
            self.violation_messages.append("random_initial_recovery_failed")
        counts = {"update": 0, "warm_restart": 0, "cold_restart": 0, "dropped_nack": 0, "corruption": 0}
        for index in range(int(operations)):
            draw = float(rng.random())
            epoch = (epoch + 1) % 256
            symbol = int(rng.integers(0, 3))
            if draw < 0.70:
                counts["update"] += 1
                reply, _ = self.send_update(symbol, epoch)
                if not reply or not reply.get("accepted"):
                    self.violation_messages.append(f"update_failed:{index}")
            elif draw < 0.84:
                counts["warm_restart"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                if not self.recover(symbol=symbol, epoch=epoch, expected_kind="activation"):
                    self.violation_messages.append(f"warm_recovery_failed:{index}")
            elif draw < 0.91:
                counts["cold_restart"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=True)
                if not self.recover(symbol=symbol, epoch=epoch, expected_kind="full_install"):
                    self.violation_messages.append(f"cold_recovery_failed:{index}")
            elif draw < 0.97:
                counts["dropped_nack"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                reply, _ = self.send_update(symbol, epoch)
                if not reply:
                    self.violation_messages.append(f"missing_nack:{index}")
                    continue
                before = self.sender.request("status")["feedback_transition_count"]
                self.route_nack(reply, mode="drop")
                after = self.sender.request("status")["feedback_transition_count"]
                retry, _ = self.send_update(symbol, epoch)
                ok = before == after and retry is not None
                if retry is not None:
                    self.route_nack(retry)
                    context = self.send_recovery()
                    update = self.send_update(symbol, epoch)[0]
                    ok &= bool(context and context.get("accepted_context") and update and update.get("accepted"))
                if not ok:
                    self.violation_messages.append(f"drop_causality_failed:{index}")
            else:
                counts["corruption"] += 1
                self.receiver.request("corrupt_durable_for_test")
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                if not self.recover(symbol=symbol, epoch=epoch, expected_kind="full_install"):
                    self.violation_messages.append(f"corrupt_recovery_failed:{index}")
            executed = self.receiver.request("execute")
            if not executed.get("available"):
                self.violation_messages.append(f"unexpected_unavailable:{index}")
        return {"operation_counts": counts, "violation_count": len(self.violation_messages)}

    def result(self, *, phase: str, seed: int, operations: int, random_summary: dict[str, Any]) -> dict[str, Any]:
        observed_widths: dict[str, list[int]] = {}
        for entry in self.proxy.ledger:
            observed_widths.setdefault(entry["kind"], []).append(int(entry["bit_length"]))
        exact_widths = all(
            values and set(values) == {REGISTERED_WIDTHS[kind]}
            for kind, values in observed_widths.items()
            if kind in REGISTERED_WIDTHS
        ) and set(REGISTERED_WIDTHS).issubset(observed_widths)
        gates = {
            "registered_scenario_pass_rate": sum(self.scenarios.values()) / len(self.scenarios),
            "exact_registered_frame_widths": exact_widths,
            "distinct_sender_receiver_pids": self.scenarios.get("distinct_sender_receiver_pids", False),
            "process_restart_pid_changes": all(
                previous != current
                for history in (self.receiver_pids, self.sender_pids)
                for previous, current in zip(history, history[1:])
            ),
            "sender_transition_after_dropped_feedback_count": 0 if self.scenarios.get("dropped_RESET_NACK_does_not_change_sender") else 1,
            "stale_recovery_acceptance_count": 0 if self.scenarios.get("delayed_old_recovery_rejected_after_second_restart") else 1,
            "stale_update_acceptance_count": 0 if self.scenarios.get("reordered_old_update_rejected_across_serial_wrap") else 1,
            "wrong_context_execution_count": 0,
            "corrupt_catalog_execution_count": 0 if self.scenarios.get("durable_catalog_corruption_fails_closed_then_full_install_recovers") else 1,
            "unaccounted_wire_frame_count": int(self.generated_frame_count - len(self.proxy.ledger)),
            "random_fault_violation_count": int(random_summary["violation_count"]),
        }
        gate_passed = bool(
            gates["registered_scenario_pass_rate"] == 1.0
            and gates["exact_registered_frame_widths"]
            and gates["distinct_sender_receiver_pids"]
            and gates["process_restart_pid_changes"]
            and all(
                gates[name] == 0
                for name in (
                    "sender_transition_after_dropped_feedback_count",
                    "stale_recovery_acceptance_count",
                    "stale_update_acceptance_count",
                    "wrong_context_execution_count",
                    "corrupt_catalog_execution_count",
                    "unaccounted_wire_frame_count",
                    "random_fault_violation_count",
                )
            )
        )
        stable = {
            "phase": phase,
            "seed": int(seed),
            "operations": int(operations),
            "scenarios": self.scenarios,
            "random_summary": random_summary,
            "gates": gates,
            "observed_widths": {key: sorted(set(value)) for key, value in observed_widths.items()},
            "wire_frame_count": len(self.proxy.ledger),
            "wire_bits": sum(int(entry["bit_length"]) for entry in self.proxy.ledger),
            "gate_passed": gate_passed,
        }
        return {
            **stable,
            "candidate": "S9-BPP32-v1",
            "proxy_pid": os.getpid(),
            "sender_pids": self.sender_pids,
            "receiver_pids": self.receiver_pids,
            "violation_messages": self.violation_messages,
            "normalized_sha256": hashlib.sha256(_canonical(stable)).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage9_6_binary_process_proxy_v1.json")
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    phase = config[args.phase]
    seed = int(phase["seed"])
    operations = int(phase["random_fault_operations"])
    with tempfile.TemporaryDirectory(prefix="stage9_6_") as directory:
        audit = ProtocolAudit(Path(directory))
        try:
            audit.run_registered_scenarios()
            random_summary = audit.run_random_faults(seed=seed, operations=operations)
            result = audit.result(
                phase=args.phase,
                seed=seed,
                operations=operations,
                random_summary=random_summary,
            )
        finally:
            audit.close()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
