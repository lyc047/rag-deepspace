#!/usr/bin/env python
"""Hardened JSONL receiver worker for Stage-9.1 process tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_semcom.stage9_1_hybrid_recovery import (  # noqa: E402
    HardenedPersistentReceiver,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True)
    args = parser.parse_args()
    receiver = HardenedPersistentReceiver(args.state_path)
    for line in sys.stdin:
        request = json.loads(line)
        operation = request.pop("op")
        if operation == "boot":
            response = receiver.boot(**request)
        elif operation == "status":
            response = receiver.status()
        elif operation == "restore":
            response = receiver.restore(**request)
        elif operation == "update":
            response = receiver.update(**request)
        elif operation == "execute":
            response = receiver.execute()
        else:
            raise ValueError(f"unknown operation: {operation}")
        print(json.dumps(response, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

