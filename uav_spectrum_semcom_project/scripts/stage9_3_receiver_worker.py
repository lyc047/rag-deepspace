#!/usr/bin/env python
"""JSONL worker for the Stage-9.3 persistent tagged receiver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectrum_semcom.stage9_3_tagged_recovery import TaggedRecoveryReceiver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True)
    args = parser.parse_args()
    receiver = TaggedRecoveryReceiver(args.state_path)
    operations = {
        "boot": receiver.boot,
        "status": receiver.status,
        "begin_recovery": receiver.begin_recovery,
        "restore": receiver.restore,
        "update": receiver.update,
        "enqueue_update": receiver.enqueue_update,
        "advance_update_clock": receiver.advance_update_clock,
        "deliver_queued": receiver.deliver_queued,
        "execute": receiver.execute,
    }
    for line in sys.stdin:
        request = json.loads(line)
        operation = request.pop("op")
        if operation not in operations:
            raise ValueError(f"unknown operation: {operation}")
        response = operations[operation](**request)
        print(json.dumps(response, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

