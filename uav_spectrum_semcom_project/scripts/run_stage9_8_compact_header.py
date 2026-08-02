#!/usr/bin/env python
"""Run Stage-9.8 compact-header confirmation on the Stage-9.7 UDP harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import run_stage9_7_udp_transport as base

from spectrum_semcom.stage9_7_udp_transport import REGISTERED_UDP_PAYLOAD_BITS as H32_BITS
from spectrum_semcom.stage9_8_compact_transport import (
    REGISTERED_COMPACT_UDP_PAYLOAD_BITS as H8_BITS,
    decode_compact_udp_datagram,
    escape_roundtrip_self_check,
)


ROOT = Path(__file__).resolve().parents[1]
base.WORKER = ROOT / "scripts" / "stage9_8_udp_worker.py"
base.REGISTERED_UDP_PAYLOAD_BITS = H8_BITS


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _batch_upper_bound(update_count: int, batch_size: int) -> dict[str, Any]:
    full, remainder = divmod(int(update_count), int(batch_size))
    full_width = 8 + 8 * math.ceil(42 * int(batch_size) / 8)
    remainder_width = 0 if remainder == 0 else 8 + 8 * math.ceil(42 * remainder / 8)
    total = full * full_width + remainder_width
    individual = int(update_count) * H8_BITS["update"]
    saving = 0.0 if individual == 0 else 100.0 * (individual - total) / individual
    return {
        "batch_size": int(batch_size),
        "update_count": int(update_count),
        "ideal_udp_payload_bits": int(total),
        "individual_h8_udp_payload_bits": int(individual),
        "ideal_saving_percent": float(saving),
        "average_wait_scenes_full_batch": (int(batch_size) - 1) / 2,
        "maximum_wait_scenes": int(batch_size) - 1,
        "lost_updates_per_lost_datagram": int(batch_size),
        "adopted": False,
    }


def _codec_checks() -> dict[str, bool]:
    escape_ok = escape_roundtrip_self_check()
    invalid_rejected = False
    try:
        decode_compact_udp_datagram(bytes([0xA7]) + bytes(5))
    except ValueError:
        invalid_rejected = True
    return {
        "escape_56_bit_roundtrip": bool(escape_ok),
        "invalid_length_rejected": bool(invalid_rejected),
    }


def _enrich_result(
    audit: base.UdpTransportAudit,
    result: dict[str, Any],
) -> dict[str, Any]:
    assert audit.proxy is not None
    current = audit.proxy.request("ledger_summary").get("ledger", [])
    ledger = [*audit.archived_ledgers, *current]
    ingress = [
        entry for entry in ledger
        if entry.get("direction") in {"forward", "feedback"}
        and not entry.get("release_only", False)
    ]
    frame_counts: dict[str, int] = {}
    for entry in ingress:
        frame_counts[entry["kind"]] = frame_counts.get(entry["kind"], 0) + 1
    h32_total = sum(frame_counts[kind] * H32_BITS[kind] for kind in frame_counts)
    h8_total = int(result["proxy_ingress_udp_payload_bits"])
    protocol_total = int(result["proxy_ingress_protocol_bits"])
    saving_percent = 100.0 * (h32_total - h8_total) / h32_total
    overhead_percent = 100.0 * (h8_total - protocol_total) / protocol_total
    exact_24 = h32_total - h8_total == 24 * len(ingress)
    codec = _codec_checks()
    comparison = {
        "frame_counts": frame_counts,
        "counterfactual_h32_udp_payload_bits": int(h32_total),
        "observed_h8_udp_payload_bits": int(h8_total),
        "h8_saving_vs_h32_percent": float(saving_percent),
        "h8_overhead_relative_to_protocol_percent": float(overhead_percent),
        "exact_24_bit_saving_per_normal_datagram": bool(exact_24),
    }
    batching = {
        "status": "analytical_upper_bound_not_adopted",
        "reason": "No temporal task-regret evaluation is permitted in Stage-9.8.",
        "B2": _batch_upper_bound(frame_counts.get("update", 0), 2),
        "B4": _batch_upper_bound(frame_counts.get("update", 0), 4),
    }
    result["candidate"] = "S9-CH8-v1"
    result["codec_checks"] = codec
    result["matched_ledger_comparison"] = comparison
    result["batching_upper_bound"] = batching
    result["gates"].update(
        {
            "minimum_h8_udp_payload_saving_vs_h32_percent": saving_percent >= 20.0,
            "maximum_h8_overhead_relative_to_protocol_percent": overhead_percent <= 25.0,
            "exact_24_bit_saving_per_normal_datagram": bool(exact_24),
            **codec,
        }
    )
    result["gate_passed"] = bool(
        result["gate_passed"]
        and all(
            result["gates"][key]
            for key in (
                "minimum_h8_udp_payload_saving_vs_h32_percent",
                "maximum_h8_overhead_relative_to_protocol_percent",
                "exact_24_bit_saving_per_normal_datagram",
                "escape_56_bit_roundtrip",
                "invalid_length_rejected",
            )
        )
    )
    stable = {
        key: value
        for key, value in result.items()
        if key not in {
            "candidate",
            "ports",
            "sender_pids",
            "receiver_pids",
            "proxy_pids",
            "latency_summary",
            "violation_messages",
            "normalized_sha256",
        }
    }
    result["normalized_sha256"] = hashlib.sha256(_canonical(stable)).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage9_8_compact_header_v1.json")
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    phase_config = config[args.phase]
    with tempfile.TemporaryDirectory(prefix="stage9_8_h8_") as directory:
        audit = base.UdpTransportAudit(Path(directory), int(phase_config["receive_timeout_ms"]))
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
            result = _enrich_result(audit, result)
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
