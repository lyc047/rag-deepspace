#!/usr/bin/env python
"""Freeze the exact Stage-5 Final executable file set; do not consume data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.stage5_final_governance import build_code_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_unified_external_final_protocol_v1.json",
    )
    parser.add_argument(
        "--access-state",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_external_final_access_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/external_final_freeze_v1/code_snapshot.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-5 Final snapshot")
    state = json.loads(args.access_state.read_text(encoding="utf-8"))
    if (
        state.get("access_count") != 0
        or state.get("final_signal_values_accessed") is not False
    ):
        raise RuntimeError("cannot freeze after external Final access")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    snapshot = build_code_snapshot(PROJECT_DIR, protocol)
    snapshot["created_utc"] = datetime.now(timezone.utc).isoformat()
    snapshot["access_count_at_freeze"] = 0
    atomic_write_json(args.output, snapshot)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": len(snapshot["files"]),
                "snapshot_sha256": snapshot["executable_snapshot_sha256"],
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
