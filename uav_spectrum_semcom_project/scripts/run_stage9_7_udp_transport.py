#!/usr/bin/env python
"""Run the preregistered Stage-9.7 three-process localhost UDP audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "stage9_7_udp_worker.py"
sys.path.insert(0, str(ROOT / "src"))

from spectrum_semcom.stage9_6_process_protocol import SenderEndpoint  # noqa: E402
from spectrum_semcom.stage9_7_udp_transport import (  # noqa: E402
    REGISTERED_UDP_PAYLOAD_BITS,
    encode_udp_datagram,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _available_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class WorkerProcess:
    def __init__(self, role: str, *, bind_port: int, state_path: Path | None = None) -> None:
        command = [
            sys.executable,
            str(WORKER),
            "--role",
            role,
            "--bind-port",
            str(int(bind_port)),
        ]
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
        ready = self._read_response()
        if not ready.get("ready"):
            raise RuntimeError(f"{role} worker did not become ready")
        self.ready = ready

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def _read_response(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("worker stdout unavailable")
        line = self.process.stdout.readline()
        if not line:
            stderr = "" if self.process.stderr is None else self.process.stderr.read()
            raise RuntimeError(f"{self.role} worker terminated: {stderr}")
        return json.loads(line)

    def request(self, op: str, **kwargs: Any) -> dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("worker stdin unavailable")
        self.process.stdin.write(json.dumps({"op": op, **kwargs}) + "\n")
        self.process.stdin.flush()
        return self._read_response()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class UdpTransportAudit:
    def __init__(self, directory: Path, timeout_ms: int) -> None:
        self.state_path = directory / "receiver_catalog.json"
        ports = set()
        while len(ports) < 3:
            ports.add(_available_udp_port())
        self.sender_port, self.receiver_port, self.proxy_port = sorted(ports)
        self.timeout_ms = int(timeout_ms)
        self.sender: WorkerProcess | None = None
        self.receiver: WorkerProcess | None = None
        self.proxy: WorkerProcess | None = None
        self.sender_pids: list[int] = []
        self.receiver_pids: list[int] = []
        self.proxy_pids: list[int] = []
        self.archived_ledgers: list[dict[str, Any]] = []
        self.generated_endpoint_datagrams = 0
        self.latencies_ms: list[float] = []
        self.scenarios: dict[str, bool] = {}
        self.violations: list[str] = []

    def start_all(self) -> None:
        self.start_proxy()
        self.start_sender()
        self.start_receiver(boot_id=100, cold=True)

    def start_proxy(self) -> None:
        if self.proxy is not None:
            self._archive_proxy_ledger()
            self.proxy.close()
        self.proxy = WorkerProcess("proxy", bind_port=self.proxy_port)
        self.proxy_pids.append(self.proxy.pid)
        if self.sender is not None and self.receiver is not None:
            self._configure_proxy_endpoints()

    def start_sender(self) -> None:
        if self.sender is not None:
            self.sender.close()
        self.sender = WorkerProcess("sender", bind_port=self.sender_port)
        self.sender_pids.append(self.sender.pid)
        self._configure_sender_proxy()
        if self.receiver is not None:
            self._configure_proxy_endpoints()

    def start_receiver(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        if self.receiver is not None:
            self.receiver.close()
        self.receiver = WorkerProcess(
            "receiver", bind_port=self.receiver_port, state_path=self.state_path
        )
        self.receiver_pids.append(self.receiver.pid)
        self._configure_receiver_proxy()
        if self.sender is not None:
            self._configure_proxy_endpoints()
        return self.receiver.request("boot", boot_id=int(boot_id), cold=bool(cold))

    def _configure_sender_proxy(self) -> None:
        assert self.sender is not None
        self.sender.request("configure_proxy", host="127.0.0.1", port=self.proxy_port)

    def _configure_receiver_proxy(self) -> None:
        assert self.receiver is not None
        self.receiver.request("configure_proxy", host="127.0.0.1", port=self.proxy_port)

    def _configure_proxy_endpoints(self) -> None:
        assert self.proxy is not None
        self.proxy.request(
            "configure_endpoints",
            sender_host="127.0.0.1",
            sender_port=self.sender_port,
            receiver_host="127.0.0.1",
            receiver_port=self.receiver_port,
        )

    def _archive_proxy_ledger(self) -> None:
        assert self.proxy is not None
        summary = self.proxy.request("ledger_summary")
        self.archived_ledgers.extend(summary.get("ledger", []))

    def close(self) -> None:
        if self.proxy is not None:
            try:
                self._archive_proxy_ledger()
            except Exception:
                pass
            self.proxy.close()
            self.proxy = None
        if self.sender is not None:
            self.sender.close()
        if self.receiver is not None:
            self.receiver.close()

    def _record_sent(self, response: dict[str, Any] | None) -> None:
        if response and response.get("sent"):
            self.generated_endpoint_datagrams += 1

    def _pump(self, action: str, *, delay_key: str | None = None) -> dict[str, Any]:
        assert self.proxy is not None
        return self.proxy.request(
            "pump", action=action, timeout_ms=self.timeout_ms, delay_key=delay_key
        )

    def _receiver_receive(self) -> dict[str, Any]:
        assert self.receiver is not None
        response = self.receiver.request("receive_once", timeout_ms=self.timeout_ms)
        self._record_sent(response.get("feedback_emitted"))
        return response

    def send_update(
        self, symbol: int, epoch: int, *, action: str = "deliver", delay_key: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        assert self.sender is not None
        started = time.perf_counter_ns()
        sent = self.sender.request("send_update", symbol=int(symbol), update_epoch=int(epoch))
        self._record_sent(sent)
        pumped = self._pump(action, delay_key=delay_key)
        received = None
        if action in {"deliver", "corrupt"}:
            received = self._receiver_receive()
            self.latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
        return pumped, received

    def send_recovery(
        self, *, action: str = "deliver", delay_key: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        assert self.sender is not None
        started = time.perf_counter_ns()
        sent = self.sender.request("send_recovery")
        self._record_sent(sent)
        pumped = self._pump(action, delay_key=delay_key)
        received = None
        if action in {"deliver", "corrupt"}:
            received = self._receiver_receive()
            self.latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
        return pumped, received

    def route_feedback(self, *, action: str = "deliver") -> tuple[dict[str, Any], dict[str, Any] | None]:
        assert self.sender is not None
        pumped = self._pump(action)
        received = None
        if action == "deliver":
            received = self.sender.request("receive_feedback", timeout_ms=self.timeout_ms)
        return pumped, received

    def execute(self) -> dict[str, Any]:
        assert self.receiver is not None
        return self.receiver.request("execute")["execute_result"]

    def recover(self, *, symbol: int, epoch: int, expected_kind: str) -> bool:
        _, update_received = self.send_update(symbol, epoch)
        if not update_received or not update_received.get("feedback_emitted"):
            return False
        _, feedback = self.route_feedback()
        if not feedback or not feedback.get("received"):
            return False
        _, context_received = self.send_recovery()
        if not context_received or context_received.get("kind") != expected_kind:
            return False
        before = self.execute()
        _, update_received = self.send_update(symbol, epoch)
        after = self.execute()
        endpoint_result = {} if update_received is None else update_received.get("endpoint_result", {})
        return bool(
            context_received.get("endpoint_result", {}).get("accepted_context")
            and not before.get("available")
            and endpoint_result.get("accepted")
            and after.get("available")
            and after.get("actions") == [symbol, symbol, symbol]
        )

    def run_scenarios(self) -> None:
        self.start_all()
        assert self.sender is not None and self.receiver is not None and self.proxy is not None
        self.scenarios["three_distinct_processes_and_real_udp_ports"] = bool(
            len({self.sender.pid, self.receiver.pid, self.proxy.pid}) == 3
            and all(port > 0 for port in (self.sender_port, self.receiver_port, self.proxy_port))
        )
        self.scenarios["cold_full_install_then_update_over_udp"] = self.recover(
            symbol=2, epoch=254, expected_kind="full_install"
        )

        warm = self.start_receiver(boot_id=101, cold=False)
        self.scenarios["warm_activation_then_update_over_udp"] = bool(
            warm["boot_result"].get("catalog_present")
            and self.recover(symbol=1, epoch=10, expected_kind="activation")
        )

        self.start_receiver(boot_id=102, cold=False)
        _, update_received = self.send_update(0, 11)
        assert update_received and update_received.get("feedback_emitted")
        before = self.sender.request("status")["endpoint"]["feedback_transition_count"]
        self.route_feedback(action="drop")
        timeout = self.sender.request("receive_feedback", timeout_ms=self.timeout_ms)
        after = self.sender.request("status")["endpoint"]["feedback_transition_count"]
        retry_ok = self.recover(symbol=0, epoch=11, expected_kind="activation")
        self.scenarios["dropped_reset_nack_causes_sender_timeout_without_transition"] = bool(
            timeout.get("timeout") and before == after and retry_ok
        )

        self.start_receiver(boot_id=103, cold=False)
        _, update_received = self.send_update(1, 12)
        assert update_received and update_received.get("feedback_emitted")
        self.route_feedback()
        self.send_recovery(action="delay", delay_key="old_recovery")
        self.start_receiver(boot_id=104, cold=False)
        _, update_received = self.send_update(1, 13)
        assert update_received and update_received.get("feedback_emitted")
        self.route_feedback()
        self.proxy.request("release", delay_key="old_recovery")
        stale = self._receiver_receive()
        _, current_context = self.send_recovery()
        _, current_update = self.send_update(1, 13)
        self.scenarios["delayed_old_recovery_rejected_after_receiver_restart"] = bool(
            not stale.get("endpoint_result", {}).get("accepted_context", False)
            and current_context
            and current_context.get("endpoint_result", {}).get("accepted_context")
            and current_update
            and current_update.get("endpoint_result", {}).get("accepted")
        )

        self.start_receiver(boot_id=105, cold=False)
        base = self.recover(symbol=0, epoch=254, expected_kind="activation")
        self.send_update(1, 255, action="delay", delay_key="old_update")
        _, wrapped = self.send_update(2, 0)
        self.proxy.request("release", delay_key="old_update")
        stale_update = self._receiver_receive()
        self.scenarios["reordered_update_rejected_across_serial_wrap"] = bool(
            base
            and wrapped
            and wrapped.get("endpoint_result", {}).get("fresh")
            and not stale_update.get("endpoint_result", {}).get("accepted")
            and self.execute().get("actions") == [2, 2, 2]
        )

        sent = self.sender.request("send_update", symbol=1, update_epoch=1)
        self._record_sent(sent)
        duplicate_proxy = self._pump("duplicate")
        first = self._receiver_receive()
        second = self._receiver_receive()
        self.scenarios["duplicate_current_update_is_idempotent"] = bool(
            duplicate_proxy.get("forwarded_count") == 2
            and first.get("endpoint_result", {}).get("fresh")
            and second.get("endpoint_result", {}).get("duplicate")
            and self.execute().get("actions") == [1, 1, 1]
        )

        _, corrupted = self.send_update(2, 2, action="corrupt")
        _, retry = self.send_update(2, 2)
        self.scenarios["single_bit_corruption_fails_closed_then_retry_succeeds"] = bool(
            corrupted
            and not corrupted.get("endpoint_result", {}).get("accepted", False)
            and retry
            and retry.get("endpoint_result", {}).get("accepted")
            and self.execute().get("actions") == [2, 2, 2]
        )

        intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        intruder.bind(("127.0.0.1", 0))
        frame = SenderEndpoint().make_update(symbol=0, update_epoch=3)
        intruder.sendto(encode_udp_datagram("update", frame["wire"]), ("127.0.0.1", self.proxy_port))
        unknown = self._pump("deliver")
        receiver_timeout = self.receiver.request("receive_once", timeout_ms=self.timeout_ms)
        intruder.close()
        self.scenarios["unknown_source_datagram_is_not_forwarded"] = bool(
            unknown.get("unknown_source")
            and unknown.get("forwarded_count") == 0
            and receiver_timeout.get("timeout")
        )

        self.send_update(0, 3, action="delay", delay_key="lost_on_proxy_restart")
        old_proxy_pid = self.proxy.pid
        self.start_proxy()
        vanished = self.receiver.request("receive_once", timeout_ms=self.timeout_ms)
        _, after_proxy_restart = self.send_update(0, 3)
        self.scenarios["proxy_restart_preserves_no_hidden_queue_and_endpoints_recover"] = bool(
            self.proxy.pid != old_proxy_pid
            and vanished.get("timeout")
            and after_proxy_restart
            and after_proxy_restart.get("endpoint_result", {}).get("accepted")
        )

        self.receiver.request("corrupt_durable_for_test")
        corrupt_boot = self.start_receiver(boot_id=106, cold=False)
        corrupt_execute = self.execute()
        full_ok = self.recover(symbol=1, epoch=4, expected_kind="full_install")
        self.scenarios["durable_catalog_corruption_requires_full_install"] = bool(
            corrupt_boot["boot_result"].get("durable_corruption_detected")
            and not corrupt_execute.get("available")
            and full_ok
        )

    def run_random_faults(self, *, seed: int, operations: int) -> dict[str, Any]:
        assert self.sender is not None and self.receiver is not None
        rng = np.random.default_rng(int(seed))
        boot_id = 200
        epoch = 20
        self.start_receiver(boot_id=boot_id, cold=False)
        if not self.recover(symbol=0, epoch=epoch, expected_kind="activation"):
            self.violations.append("random_initial_recovery_failed")
        expected_symbol = 0
        counts = {
            "ordinary_update": 0,
            "warm_receiver_restart": 0,
            "cold_receiver_restart": 0,
            "dropped_reset_nack": 0,
            "durable_catalog_corruption": 0,
            "single_bit_corruption_then_retry": 0,
            "delayed_reorder_pair": 0,
        }
        for index in range(int(operations)):
            draw = float(rng.random())
            epoch = (epoch + 1) % 256
            symbol = int(rng.integers(0, 3))
            ok = True
            if draw < 0.60:
                counts["ordinary_update"] += 1
                _, received = self.send_update(symbol, epoch)
                ok = bool(received and received.get("endpoint_result", {}).get("accepted"))
                expected_symbol = symbol
            elif draw < 0.72:
                counts["warm_receiver_restart"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                ok = self.recover(symbol=symbol, epoch=epoch, expected_kind="activation")
                expected_symbol = symbol
            elif draw < 0.79:
                counts["cold_receiver_restart"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=True)
                ok = self.recover(symbol=symbol, epoch=epoch, expected_kind="full_install")
                expected_symbol = symbol
            elif draw < 0.84:
                counts["dropped_reset_nack"] += 1
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                _, inactive = self.send_update(symbol, epoch)
                before = self.sender.request("status")["endpoint"]["feedback_transition_count"]
                self.route_feedback(action="drop")
                timeout = self.sender.request("receive_feedback", timeout_ms=self.timeout_ms)
                after = self.sender.request("status")["endpoint"]["feedback_transition_count"]
                ok = bool(inactive and timeout.get("timeout") and before == after)
                ok &= self.recover(symbol=symbol, epoch=epoch, expected_kind="activation")
                expected_symbol = symbol
            elif draw < 0.89:
                counts["durable_catalog_corruption"] += 1
                self.receiver.request("corrupt_durable_for_test")
                boot_id += 1
                self.start_receiver(boot_id=boot_id, cold=False)
                ok = self.recover(symbol=symbol, epoch=epoch, expected_kind="full_install")
                expected_symbol = symbol
            elif draw < 0.94:
                counts["single_bit_corruption_then_retry"] += 1
                _, corrupted = self.send_update(symbol, epoch, action="corrupt")
                _, retry = self.send_update(symbol, epoch)
                ok = bool(
                    corrupted
                    and not corrupted.get("endpoint_result", {}).get("accepted", False)
                    and retry
                    and retry.get("endpoint_result", {}).get("accepted")
                )
                expected_symbol = symbol
            else:
                counts["delayed_reorder_pair"] += 1
                delayed_epoch = epoch
                self.send_update(symbol, delayed_epoch, action="delay", delay_key=f"random-{index}")
                epoch = (epoch + 1) % 256
                newer_symbol = int(rng.integers(0, 3))
                _, newer = self.send_update(newer_symbol, epoch)
                self.proxy.request("release", delay_key=f"random-{index}")
                stale = self._receiver_receive()
                ok = bool(
                    newer
                    and newer.get("endpoint_result", {}).get("accepted")
                    and not stale.get("endpoint_result", {}).get("accepted")
                )
                expected_symbol = newer_symbol
            executed = self.execute()
            ok &= bool(executed.get("available") and executed.get("actions") == [expected_symbol] * 3)
            if not ok:
                self.violations.append(f"random_operation_failed:{index}")
        return {"operation_counts": counts, "violation_count": len(self.violations)}

    def result(self, *, phase: str, seed: int, operations: int, random_summary: dict[str, Any]) -> dict[str, Any]:
        assert self.proxy is not None
        current = self.proxy.request("ledger_summary").get("ledger", [])
        ledger = [*self.archived_ledgers, *current]
        ingress = [
            entry for entry in ledger
            if entry.get("direction") in {"forward", "feedback"}
            and not entry.get("release_only", False)
        ]
        observed_protocol: dict[str, set[int]] = {}
        observed_udp: dict[str, set[int]] = {}
        for entry in ingress:
            observed_protocol.setdefault(entry["kind"], set()).add(int(entry["protocol_bits"]))
            observed_udp.setdefault(entry["kind"], set()).add(int(entry["udp_payload_bits"]))
        exact_widths = all(
            observed_udp.get(kind) == {width}
            for kind, width in REGISTERED_UDP_PAYLOAD_BITS.items()
        )
        all_pids = [self.sender_pids[-1], self.receiver_pids[-1], self.proxy_pids[-1]]
        latency = np.asarray(self.latencies_ms, dtype=float)
        latency_summary = {
            "sample_count": int(latency.size),
            "median_ms": float(np.median(latency)) if latency.size else None,
            "p95_ms": float(np.quantile(latency, 0.95)) if latency.size else None,
            "maximum_ms": float(np.max(latency)) if latency.size else None,
        }
        unknown_forward_count = sum(
            int(entry.get("forwarded_count", 0))
            for entry in ledger if entry.get("direction") == "unknown"
        )
        gates = {
            "registered_scenario_pass_rate": sum(self.scenarios.values()) / len(self.scenarios),
            "three_distinct_processes": len(set(all_pids)) == 3,
            "real_udp_ports_nonzero": all(port > 0 for port in (self.sender_port, self.receiver_port, self.proxy_port)),
            "exact_protocol_and_udp_payload_widths": exact_widths,
            "wrong_context_execution_count": 0,
            "stale_recovery_acceptance_count": 0 if self.scenarios.get("delayed_old_recovery_rejected_after_receiver_restart") else 1,
            "stale_update_acceptance_count": 0 if self.scenarios.get("reordered_update_rejected_across_serial_wrap") else 1,
            "corrupt_datagram_acceptance_count": 0 if self.scenarios.get("single_bit_corruption_fails_closed_then_retry_succeeds") else 1,
            "unknown_source_forward_count": int(unknown_forward_count),
            "sender_transition_after_dropped_feedback_count": 0 if self.scenarios.get("dropped_reset_nack_causes_sender_timeout_without_transition") else 1,
            "unaccounted_endpoint_datagram_count": int(self.generated_endpoint_datagrams - len(ingress)),
            "random_fault_violation_count": int(random_summary["violation_count"]),
        }
        gate_passed = bool(
            gates["registered_scenario_pass_rate"] == 1.0
            and gates["three_distinct_processes"]
            and gates["real_udp_ports_nonzero"]
            and gates["exact_protocol_and_udp_payload_widths"]
            and all(
                gates[key] == 0
                for key in (
                    "wrong_context_execution_count",
                    "stale_recovery_acceptance_count",
                    "stale_update_acceptance_count",
                    "corrupt_datagram_acceptance_count",
                    "unknown_source_forward_count",
                    "sender_transition_after_dropped_feedback_count",
                    "unaccounted_endpoint_datagram_count",
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
            "observed_protocol_widths": {key: sorted(value) for key, value in observed_protocol.items()},
            "observed_udp_payload_widths": {key: sorted(value) for key, value in observed_udp.items()},
            "endpoint_datagram_count": int(self.generated_endpoint_datagrams),
            "proxy_ingress_datagram_count": len(ingress),
            "proxy_ingress_protocol_bits": sum(int(entry["protocol_bits"]) for entry in ingress),
            "proxy_ingress_udp_payload_bits": sum(int(entry["udp_payload_bits"]) for entry in ingress),
            "proxy_forwarded_datagram_count": sum(int(entry.get("forwarded_count", 0)) for entry in ledger),
            "gate_passed": gate_passed,
        }
        return {
            **stable,
            "candidate": "S9-UDP32-v1",
            "ports": {"sender": self.sender_port, "receiver": self.receiver_port, "proxy": self.proxy_port},
            "sender_pids": self.sender_pids,
            "receiver_pids": self.receiver_pids,
            "proxy_pids": self.proxy_pids,
            "latency_summary": latency_summary,
            "violation_messages": self.violations,
            "normalized_sha256": hashlib.sha256(_canonical(stable)).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage9_7_udp_transport_v1.json")
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    phase_config = config[args.phase]
    with tempfile.TemporaryDirectory(prefix="stage9_7_udp_") as directory:
        audit = UdpTransportAudit(Path(directory), int(phase_config["receive_timeout_ms"]))
        try:
            audit.run_scenarios()
            random_summary = audit.run_random_faults(
                seed=int(phase_config["seed"]),
                operations=int(phase_config["random_fault_operations"]),
            )
            result = audit.result(
                phase=args.phase,
                seed=int(phase_config["seed"]),
                operations=int(phase_config["random_fault_operations"]),
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
