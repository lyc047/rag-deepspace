#!/usr/bin/env python
"""Atomically consume the single C1-v4 Final access."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json")
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json")
    parser.add_argument("--inventory", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_freeze_v1/executable_snapshot.json")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state.get("status") != "not_accessed" or state.get("access_count") != 0 or state.get("receipt") is not None:
        raise RuntimeError("C1-v4 Final access is not available")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if canonical_json_sha256(snapshot.get("files", [])) != snapshot.get("executable_snapshot_sha256"):
        raise ValueError("snapshot digest mismatch")
    for row in snapshot["files"]:
        path = PROJECT_DIR / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"frozen file missing or changed: {row['path']}")
    receipt = {
        "consumed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol), "inventory_sha256": sha256_file(args.inventory),
        "executable_snapshot_sha256": snapshot["executable_snapshot_sha256"],
    }
    updated = {**state, "status": "access_consumed", "access_count": 1, "receipt": receipt, "receipt_sha256": canonical_json_sha256(receipt)}
    atomic_write_json(args.state, updated)
    print(json.dumps({"status": updated["status"], "access_count": 1, "receipt_sha256": updated["receipt_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
