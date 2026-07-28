"""Governance helpers for the single-access Stage-6 external Final."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .final_holdout import canonical_json_sha256, sha256_path
from .stage5_final_governance import validate_external_final_catalog


SNAPSHOT_STATUS = "stage6_external_final_code_frozen"


def build_stage6_code_snapshot(
    project_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = protocol["freeze"]
    paths: set[Path] = set()
    for pattern in freeze["include_globs"]:
        paths.update(
            path for path in project_dir.glob(pattern) if path.is_file()
        )
    paths.update(project_dir / value for value in freeze["include_files"])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot inputs missing: {missing}")
    files = [
        {
            "path": path.relative_to(project_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return {
        "version": "1.0",
        "status": SNAPSHOT_STATUS,
        "protocol_id": protocol["protocol_id"],
        "files": files,
        "executable_snapshot_sha256": canonical_json_sha256(files),
    }


def verify_stage6_code_snapshot(
    project_dir: Path, snapshot: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    files = snapshot.get("files", [])
    if (
        snapshot.get("status") != SNAPSHOT_STATUS
        or canonical_json_sha256(files)
        != snapshot.get("executable_snapshot_sha256")
    ):
        errors.append("snapshot identity or canonical hash is invalid")
    for row in files:
        path = project_dir / row["path"]
        if not path.is_file():
            errors.append(f"missing frozen file: {row['path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"frozen file size changed: {row['path']}")
        elif sha256_path(path) != row["sha256"]:
            errors.append(f"frozen file SHA-256 changed: {row['path']}")
    return errors


__all__ = [
    "build_stage6_code_snapshot",
    "validate_external_final_catalog",
    "verify_stage6_code_snapshot",
]
