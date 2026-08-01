#!/usr/bin/env python
"""Three-role JSONL control worker with an actual UDP data plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_semcom.stage9_6_process_protocol import (  # noqa: E402
    ReceiverEndpoint,
    SenderEndpoint,
)
from spectrum_semcom.stage9_7_udp_transport import (  # noqa: E402
    corrupt_protocol_payload,
    decode_udp_datagram,
    encode_udp_datagram,
)


class UdpSocketBase:
    def __init__(self, bind_port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", int(bind_port)))
        self.proxy_address: tuple[str, int] | None = None

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()
        return str(host), int(port)

    def configure_proxy(self, *, host: str, port: int) -> dict[str, Any]:
        self.proxy_address = (str(host), int(port))
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "local_address": list(self.local_address),
            "proxy_address": None if self.proxy_address is None else list(self.proxy_address),
        }

    def _send_datagram(self, kind: str, wire: dict[str, Any]) -> dict[str, Any]:
        if self.proxy_address is None:
            raise ValueError("UDP proxy is not configured")
        datagram = encode_udp_datagram(kind, wire)
        sent = self.socket.sendto(datagram, self.proxy_address)
        if sent != len(datagram):
            raise RuntimeError("partial UDP datagram send")
        decoded = decode_udp_datagram(datagram)
        return {
            "sent": True,
            "kind": kind,
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "datagram_sha256": decoded.sha256,
        }


class UdpSender(UdpSocketBase):
    def __init__(self, bind_port: int) -> None:
        super().__init__(bind_port)
        self.endpoint = SenderEndpoint()

    def status(self) -> dict[str, Any]:
        return {**super().status(), "endpoint": self.endpoint.status()}

    def send_update(self, *, symbol: int, update_epoch: int) -> dict[str, Any]:
        frame = self.endpoint.make_update(symbol=int(symbol), update_epoch=int(update_epoch))
        return {**self._send_datagram("update", frame["wire"]), **self.status()}

    def send_recovery(self) -> dict[str, Any]:
        frame = self.endpoint.make_recovery()
        return {**self._send_datagram(frame["kind"], frame["wire"]), **self.status()}

    def receive_feedback(self, *, timeout_ms: int) -> dict[str, Any]:
        self.socket.settimeout(float(timeout_ms) / 1000.0)
        before = self.endpoint.feedback_transition_count
        try:
            datagram, source = self.socket.recvfrom(65535)
        except socket.timeout:
            return {
                "received": False,
                "timeout": True,
                "feedback_transition_delta": 0,
                **self.status(),
            }
        decoded = decode_udp_datagram(datagram)
        if decoded.kind != "reset_nack":
            raise ValueError("sender received non-feedback UDP datagram")
        result = self.endpoint.receive_reset_nack(wire=decoded.wire)
        return {
            "received": True,
            "timeout": False,
            "source": list(source),
            "kind": decoded.kind,
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "feedback_transition_delta": self.endpoint.feedback_transition_count - before,
            "endpoint_result": result,
            **self.status(),
        }


class UdpReceiver(UdpSocketBase):
    def __init__(self, bind_port: int, state_path: str) -> None:
        super().__init__(bind_port)
        self.endpoint = ReceiverEndpoint(state_path)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "endpoint": self.endpoint.status()}

    def boot(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        return {"boot_result": self.endpoint.boot(boot_id=int(boot_id), cold=bool(cold)), **self.status()}

    def execute(self) -> dict[str, Any]:
        return {"execute_result": self.endpoint.execute(), **self.status()}

    def corrupt_durable_for_test(self) -> dict[str, Any]:
        self.endpoint.corrupt_durable_for_test()
        return self.status()

    def receive_once(self, *, timeout_ms: int) -> dict[str, Any]:
        self.socket.settimeout(float(timeout_ms) / 1000.0)
        try:
            datagram, source = self.socket.recvfrom(65535)
        except socket.timeout:
            return {"received": False, "timeout": True, **self.status()}
        try:
            decoded = decode_udp_datagram(datagram)
        except ValueError as exc:
            return {
                "received": True,
                "accepted": False,
                "decode_error": str(exc),
                "source": list(source),
                **self.status(),
            }
        if decoded.kind == "update":
            endpoint_result = self.endpoint.receive_update(wire=decoded.wire)
        elif decoded.kind in {"activation", "full_install"}:
            endpoint_result = self.endpoint.receive_recovery(
                kind=decoded.kind, wire=decoded.wire
            )
        else:
            raise ValueError("receiver received feedback frame")
        feedback = endpoint_result.get("feedback_wire")
        clean_result = dict(endpoint_result)
        clean_result.pop("feedback_wire", None)
        emitted = None
        if feedback is not None:
            emitted = self._send_datagram("reset_nack", feedback)
        return {
            "received": True,
            "timeout": False,
            "source": list(source),
            "kind": decoded.kind,
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "endpoint_result": clean_result,
            "feedback_emitted": emitted,
            **self.status(),
        }


class UdpFaultProxy:
    def __init__(self, bind_port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", int(bind_port)))
        self.sender_address: tuple[str, int] | None = None
        self.receiver_address: tuple[str, int] | None = None
        self.delayed: dict[str, tuple[bytes, str]] = {}
        self.ledger: list[dict[str, Any]] = []

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()
        return str(host), int(port)

    def configure_endpoints(
        self, *, sender_host: str, sender_port: int, receiver_host: str, receiver_port: int
    ) -> dict[str, Any]:
        self.sender_address = (str(sender_host), int(sender_port))
        self.receiver_address = (str(receiver_host), int(receiver_port))
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "local_address": list(self.local_address),
            "sender_address": None if self.sender_address is None else list(self.sender_address),
            "receiver_address": None if self.receiver_address is None else list(self.receiver_address),
            "delayed_count": len(self.delayed),
            "ledger_count": len(self.ledger),
        }

    def _direction(self, source: tuple[str, int]) -> str | None:
        if source == self.sender_address:
            return "forward"
        if source == self.receiver_address:
            return "feedback"
        return None

    def _destination(self, direction: str) -> tuple[str, int]:
        destination = self.receiver_address if direction == "forward" else self.sender_address
        if destination is None:
            raise ValueError("proxy endpoints are not configured")
        return destination

    def pump(
        self, *, action: str, timeout_ms: int, delay_key: str | None = None
    ) -> dict[str, Any]:
        self.socket.settimeout(float(timeout_ms) / 1000.0)
        try:
            datagram, source = self.socket.recvfrom(65535)
        except socket.timeout:
            return {"received": False, "timeout": True, **self.status()}
        direction = self._direction(source)
        if direction is None:
            self.ledger.append({
                "direction": "unknown",
                "action": "reject_unknown_source",
                "source": list(source),
                "udp_payload_bits": 8 * len(datagram),
                "sha256": hashlib.sha256(datagram).hexdigest(),
                "forwarded_count": 0,
            })
            return {"received": True, "unknown_source": True, "forwarded_count": 0, **self.status()}
        decoded = decode_udp_datagram(datagram)
        entry = {
            "direction": direction,
            "kind": decoded.kind,
            "action": str(action),
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "sha256": decoded.sha256,
            "forwarded_count": 0,
        }
        self.ledger.append(entry)
        if action == "drop":
            pass
        elif action == "delay":
            if not delay_key or delay_key in self.delayed:
                raise ValueError("delay requires a new delay key")
            self.delayed[str(delay_key)] = (bytes(datagram), direction)
        elif action == "deliver":
            self.socket.sendto(datagram, self._destination(direction))
            entry["forwarded_count"] = 1
        elif action == "duplicate":
            destination = self._destination(direction)
            self.socket.sendto(datagram, destination)
            self.socket.sendto(datagram, destination)
            entry["forwarded_count"] = 2
        elif action == "corrupt":
            self.socket.sendto(corrupt_protocol_payload(datagram), self._destination(direction))
            entry["forwarded_count"] = 1
        else:
            raise ValueError("unknown proxy action")
        return {
            "received": True,
            "unknown_source": False,
            "direction": direction,
            "kind": decoded.kind,
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "forwarded_count": entry["forwarded_count"],
            **self.status(),
        }

    def release(self, *, delay_key: str) -> dict[str, Any]:
        datagram, direction = self.delayed.pop(str(delay_key))
        self.socket.sendto(datagram, self._destination(direction))
        decoded = decode_udp_datagram(datagram)
        self.ledger.append({
            "direction": direction,
            "kind": decoded.kind,
            "action": "release_delayed",
            "protocol_bits": decoded.protocol_bits,
            "udp_payload_bits": decoded.udp_payload_bits,
            "sha256": decoded.sha256,
            "forwarded_count": 1,
            "release_only": True,
        })
        return {"released": True, "kind": decoded.kind, **self.status()}

    def ledger_summary(self) -> dict[str, Any]:
        return {"ledger": self.ledger, **self.status()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("sender", "receiver", "proxy"), required=True)
    parser.add_argument("--bind-port", type=int, default=0)
    parser.add_argument("--state-path")
    args = parser.parse_args()
    if args.role == "sender":
        worker = UdpSender(args.bind_port)
        operations = {
            "status": worker.status,
            "configure_proxy": worker.configure_proxy,
            "send_update": worker.send_update,
            "send_recovery": worker.send_recovery,
            "receive_feedback": worker.receive_feedback,
        }
    elif args.role == "receiver":
        if not args.state_path:
            parser.error("--state-path is required for receiver")
        worker = UdpReceiver(args.bind_port, args.state_path)
        operations = {
            "status": worker.status,
            "configure_proxy": worker.configure_proxy,
            "boot": worker.boot,
            "execute": worker.execute,
            "receive_once": worker.receive_once,
            "corrupt_durable_for_test": worker.corrupt_durable_for_test,
        }
    else:
        worker = UdpFaultProxy(args.bind_port)
        operations = {
            "status": worker.status,
            "configure_endpoints": worker.configure_endpoints,
            "pump": worker.pump,
            "release": worker.release,
            "ledger_summary": worker.ledger_summary,
        }
    print(json.dumps({"ready": True, **worker.status()}, sort_keys=True), flush=True)
    for line in sys.stdin:
        started = time.perf_counter_ns()
        try:
            request = json.loads(line)
            operation = request.pop("op")
            if operation not in operations:
                raise ValueError(f"unknown operation: {operation}")
            response = {"ok": True, **operations[operation](**request)}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        response["control_elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
        print(json.dumps(response, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
