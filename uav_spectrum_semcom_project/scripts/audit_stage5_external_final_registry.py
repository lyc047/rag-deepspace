#!/usr/bin/env python
"""Audit Stage-5 download bundles and normalized archives without signal access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PROJECT_DIR = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(registry_path: Path) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = Path(registry["local_root"])
    entries = [registry["development_adapter_pilot"], *registry["final_candidates"]]
    rows = []
    for entry in entries:
        path = root / Path(entry["archive"])
        bundle = entry["download_bundle"]
        bundle_path = Path(bundle["path"])
        row = {
            "dataset_id": entry["dataset_id"],
            "path": str(path),
            "exists": path.is_file(),
            "expected_size_bytes": int(entry["archive_size_bytes"]),
            "size_matches": None,
            "sha256_matches": None,
            "download_bundle_path": str(bundle_path),
            "download_bundle_exists": bundle_path.is_file(),
            "download_bundle_size_matches": None,
            "download_bundle_sha256_matches": None,
            "registered_member_present": None,
            "registered_member_size_matches": None,
            "archive_contents_opened": False,
            "signal_values_loaded": False,
        }
        if bundle_path.is_file():
            row["download_bundle_size_matches"] = (
                bundle_path.stat().st_size == int(bundle["size_bytes"])
            )
            row["download_bundle_sha256_matches"] = (
                sha256_file(bundle_path) == bundle["sha256"]
            )
            with zipfile.ZipFile(bundle_path) as handle:
                try:
                    member = handle.getinfo(bundle["member"])
                except KeyError:
                    row["registered_member_present"] = False
                    row["registered_member_size_matches"] = False
                else:
                    row["registered_member_present"] = True
                    row["registered_member_size_matches"] = (
                        member.file_size == int(entry["archive_size_bytes"])
                    )
        if path.is_file():
            row["actual_size_bytes"] = path.stat().st_size
            row["size_matches"] = (
                row["actual_size_bytes"] == row["expected_size_bytes"]
            )
            row["sha256_matches"] = (
                sha256_file(path) == entry["archive_sha256"]
            )
        rows.append(row)
    present = sum(row["exists"] for row in rows)
    valid = sum(
        bool(row["size_matches"] and row["sha256_matches"]) for row in rows
    )
    return {
        "registry_id": registry["registry_id"],
        "registered_archives": len(rows),
        "present_archives": present,
        "valid_archives": valid,
        "ready_for_metadata_preflight": valid == len(rows),
        "download_bundles_valid": all(
            row["download_bundle_size_matches"]
            and row["download_bundle_sha256_matches"]
            and row["registered_member_present"]
            and row["registered_member_size_matches"]
            for row in rows
        ),
        "signal_values_loaded": False,
        "method_outputs_loaded": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_external_final_registry_v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.registry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
