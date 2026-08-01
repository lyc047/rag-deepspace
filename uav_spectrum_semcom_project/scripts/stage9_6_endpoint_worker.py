#!/usr/bin/env python
"""JSONL process endpoint for Stage-9.6 binary protocol verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_semcom.stage9_6_process_protocol import (  # noqa: E402
    ReceiverEndpoint,
    SenderEndpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--state-path")
    args = parser.parse_args()
    if args.role == "sender":
        endpoint = SenderEndpoint()
        operations = {
            "status": endpoint.status,
            "receive_reset_nack": endpoint.receive_reset_nack,
            "make_recovery": endpoint.make_recovery,
            "make_update": endpoint.make_update,
        }
    else:
        if not args.state_path:
            parser.error("--state-path is required for receiver")
        endpoint = ReceiverEndpoint(args.state_path)
        operations = {
            "boot": endpoint.boot,
            "status": endpoint.status,
            "execute": endpoint.execute,
            "receive_recovery": endpoint.receive_recovery,
            "receive_update": endpoint.receive_update,
            "corrupt_durable_for_test": endpoint.corrupt_durable_for_test,
        }
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.pop("op")
            if operation not in operations:
                raise ValueError(f"unknown operation: {operation}")
            response = {"ok": True, **operations[operation](**request)}
        except Exception as exc:  # worker must fail closed, not terminate
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
