#!/usr/bin/env python
"""Verify, safely extract, audit, and pilot-register the LW1 archive.

This script never creates a stage-4 final catalog and never consumes final
access.  It is intended to run once the background archive download is fully
assembled and its Dryad checksum is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    build_permanently_excluded_pilot_manifest,
    discover_aerpaw_pairs,
    load_aerpaw_power_sweep,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


EXPECTED_ARCHIVE_SHA256 = "595e7ea4c3164b43ba1e49f154ceec583c9f4f14c565b1525e49a27311401951"


def validate_member_name(name: str) -> None:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            validate_member_name(member.filename)
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--pilot-count", type=int, default=20)
    parser.add_argument("--validate-count", type=int, default=20)
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.archive)
    if actual_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_sha256}")
    safe_extract(args.archive, args.extract_root)
    pairs, discovery_errors = discover_aerpaw_pairs(args.extract_root)
    if discovery_errors:
        raise ValueError("pair discovery failed: " + "; ".join(discovery_errors[:20]))
    if len(pairs) < args.pilot_count:
        raise ValueError("not enough complete sweep pairs for the requested pilot")

    validated = []
    for pair in pairs[: max(args.validate_count, 0)]:
        sweep = load_aerpaw_power_sweep(pair.meta_path, pair.data_path)
        validated.append(
            {
                "stem": pair.stem,
                "site": sweep.site,
                "capture_datetime": sweep.capture_datetime,
                "n_bins": sweep.n_bins,
                "frequency_span_mhz": [float(sweep.frequencies_mhz[0]), float(sweep.frequencies_mhz[-1])],
                "power_range_dbm": [float(sweep.powers_dbm.min()), float(sweep.powers_dbm.max())],
            }
        )

    args.work_root.mkdir(parents=True, exist_ok=True)
    pilot = build_permanently_excluded_pilot_manifest(
        pairs,
        dataset_root=args.extract_root,
        count=args.pilot_count,
        dataset_id="aerpaw-lw1-feb2022-dryad-hmgqnk9zn",
    )
    atomic_write_json(args.work_root / "aerpaw_lw1_pilot_manifest_v1.json", pilot)
    audit = {
        "audit_version": "1.0",
        "archive": str(args.archive.resolve()),
        "archive_sha256": actual_sha256,
        "complete_pair_count": len(pairs),
        "validated_sample_count": len(validated),
        "validated_examples": validated,
        "pilot_manifest": "aerpaw_lw1_pilot_manifest_v1.json",
        "format": "rf32_le received-power sweeps, not complex I/Q",
        "automatic_final_registration": False,
        "final_access_consumed": False,
    }
    atomic_write_json(args.work_root / "aerpaw_lw1_ingestion_audit_v1.json", audit)
    print(json.dumps({"pair_count": len(pairs), "pilot_count": args.pilot_count, "work_root": str(args.work_root), "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
