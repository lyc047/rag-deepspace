#!/usr/bin/env python
"""Freeze a non-confirmatory Stage-6 external repair validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6_external_repair_validation_v1.json"
)


def _verify_bound_file(entry: dict) -> Path:
    path = PROJECT_DIR / entry["path"]
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"bound artifact changed: {entry['path']}")
    return path


def build_snapshot(config: dict) -> dict:
    parent = config["parent_final"]
    for entry in parent.values():
        if isinstance(entry, dict) and "path" in entry:
            _verify_bound_file(entry)
    parent_snapshot = json.loads(
        (PROJECT_DIR / parent["code_snapshot"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    permitted = config["permitted_change"]
    paths: set[Path] = set()
    for row in parent_snapshot["files"]:
        path = PROJECT_DIR / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing parent-frozen file: {row['path']}")
        current = sha256_file(path)
        if current != row["sha256"]:
            raise ValueError(f"unpermitted parent-frozen change: {row['path']}")
        paths.add(path)
    adapter_path = PROJECT_DIR / permitted["path"]
    if not adapter_path.is_file():
        raise FileNotFoundError("repair-only compatibility adapter is missing")
    if any(row["path"] == permitted["path"] for row in parent_snapshot["files"]):
        raise ValueError("repair adapter unexpectedly belongs to parent snapshot")
    paths.update(
        PROJECT_DIR / value for value in config["freeze"]["additional_files"]
    )
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"repair snapshot inputs missing: {missing}")
    files = [
        {
            "path": path.relative_to(PROJECT_DIR).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return {
        "version": "1.0",
        "status": "stage6_external_repair_validation_code_frozen",
        "validation_id": config["validation_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": config["classification"],
        "parent_snapshot_sha256": parent_snapshot[
            "executable_snapshot_sha256"
        ],
        "deviation_audit": {
            "changed_parent_frozen_file_count": 0,
            "added_repair_adapter_file_count": 1,
            "changes": [
                {
                    "path": permitted["path"],
                    "parent_sha256": None,
                    "repair_sha256": sha256_file(adapter_path),
                    "classification": "isolated_data_adapter_addition",
                }
            ],
            "algorithm_or_threshold_change_detected": False,
        },
        "files": files,
        "executable_snapshot_sha256": canonical_json_sha256(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or PROJECT_DIR / config["execution"]["freeze_snapshot"]
    if output.exists():
        raise FileExistsError("refusing to overwrite repair-validation snapshot")
    snapshot = build_snapshot(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, snapshot)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": len(snapshot["files"]),
                "snapshot_sha256": snapshot["executable_snapshot_sha256"],
                "changed_parent_frozen_files": snapshot["deviation_audit"][
                    "changed_parent_frozen_file_count"
                ],
                "added_repair_adapter_files": snapshot["deviation_audit"][
                    "added_repair_adapter_file_count"
                ],
                "classification": snapshot["classification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
