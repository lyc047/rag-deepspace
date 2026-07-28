#!/usr/bin/env python
"""Normalize verified Dryad bundles without opening inner signal archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def copy_and_hash(source, target: Path) -> str:
    digest = hashlib.sha256()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as sink:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            sink.write(block)
    temporary.replace(target)
    return digest.hexdigest()


def prepare(registry_path: Path) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = Path(registry["local_root"])
    entries = [
        registry["development_adapter_pilot"],
        *registry["final_candidates"],
    ]
    rows = []
    for entry in entries:
        bundle = entry["download_bundle"]
        bundle_path = Path(bundle["path"])
        if not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        if bundle_path.stat().st_size != int(bundle["size_bytes"]):
            raise ValueError(f"download bundle size mismatch: {bundle_path}")
        if sha256_file(bundle_path) != bundle["sha256"]:
            raise ValueError(f"download bundle SHA-256 mismatch: {bundle_path}")
        target = root / entry["archive"]
        reused = False
        if (
            target.is_file()
            and target.stat().st_size == int(entry["archive_size_bytes"])
            and sha256_file(target) == entry["archive_sha256"]
        ):
            reused = True
        else:
            with zipfile.ZipFile(bundle_path) as handle:
                info = handle.getinfo(bundle["member"])
                if info.file_size != int(entry["archive_size_bytes"]):
                    raise ValueError(
                        f"inner archive size mismatch: {bundle['member']}"
                    )
                with handle.open(info) as source:
                    digest = copy_and_hash(source, target)
            if digest != entry["archive_sha256"]:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"inner archive SHA-256 mismatch: {bundle['member']}"
                )
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "role": entry["role"]
                if "role" in entry
                else "single_access_external_final",
                "download_bundle": str(bundle_path),
                "normalized_archive": str(target),
                "normalized_archive_size_bytes": target.stat().st_size,
                "normalized_archive_sha256": sha256_file(target),
                "reused_existing_verified_archive": reused,
                "inner_archive_contents_opened": False,
                "signal_values_loaded": False,
            }
        )
    return {
        "version": "1.0",
        "status": "download_bundles_and_inner_archives_verified",
        "registry_id": registry["registry_id"],
        "row_count": len(rows),
        "rows": rows,
        "governance": {
            "outer_bundle_bytes_read_for_integrity": True,
            "inner_archive_bytes_read_for_integrity": True,
            "inner_archive_member_listing_opened": False,
            "sigmf_metadata_opened": False,
            "signal_values_loaded": False,
            "method_outputs_loaded": False,
            "final_access_consumed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_external_final_registry_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/external_archive_preflight_v1/archive_preflight.json",
    )
    args = parser.parse_args()
    result = prepare(args.registry)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "archives": result["row_count"],
                "signal_values_loaded": False,
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
