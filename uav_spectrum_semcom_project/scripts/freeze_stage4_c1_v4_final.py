#!/usr/bin/env python
"""Freeze executable inputs before the single C1-v4 Final access."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_freeze_v1/executable_snapshot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing Final snapshot")
    state_path = PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "not_accessed" or state.get("access_count") != 0:
        raise RuntimeError("freeze must precede Final access")
    inventory_path = PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "materialized_metadata_only_not_accessed" or inventory.get("measurement_values_loaded") is not False:
        raise ValueError("Final inventory is not eligible for freezing")
    paths = [
        PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json",
        PROJECT_DIR / "configs/stage2_digital_link.json",
        inventory_path,
        PROJECT_DIR / "scripts/run_stage2_digital_link.py",
        PROJECT_DIR / "scripts/build_stage4_c1_v4_final_cache.py",
        PROJECT_DIR / "scripts/run_stage4_c1_v4_final.py",
        PROJECT_DIR / "scripts/consume_stage4_c1_v4_final_access.py",
        Path(__file__).resolve(),
        PROJECT_DIR / "pyproject.toml",
        PROJECT_DIR / "requirements-lock.txt",
    ]
    paths.extend(sorted((PROJECT_DIR / "src/spectrum_semcom").glob("*.py")))
    resolved = sorted({path.resolve() for path in paths}, key=lambda value: str(value).lower())
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("snapshot inputs are missing: " + "; ".join(missing))
    files = [{
        "path": str(path.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
    } for path in resolved]
    payload = {
        "version": "1.0", "status": "frozen_before_single_c1_v4_final_access",
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "executable_snapshot_sha256": canonical_json_sha256(files),
        "environment": environment_snapshot(["numpy"]),
        "access_state_deliberately_excluded": str(state_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "files": len(files), "snapshot_sha256": payload["executable_snapshot_sha256"], "access_count": 0}, indent=2))


if __name__ == "__main__":
    main()
