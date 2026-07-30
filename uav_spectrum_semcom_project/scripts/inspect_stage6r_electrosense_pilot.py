#!/usr/bin/env python
"""Consume only the four-site Pilot role and characterize its PSD arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import (  # noqa: E402
    claim_role_access,
    load_npy_member,
    read_json,
    role_members,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


DEFAULT_ARCHIVE = Path(r"F:\spectrum_bands.tar.gz")
DEFAULT_REGISTRY = (
    PROJECT_DIR
    / "results/stage6r/electrosense_final_registry_v1/registry.json"
)
DEFAULT_ACCESS_STATE = (
    PROJECT_DIR / "configs/stage6r_electrosense_final_access_state_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/electrosense_pilot_inspection_v1/result.json"
)


def _diagnostics(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("Pilot array contains no finite values")
    quantiles = np.quantile(
        finite_values, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    )
    return {
        "shape": list(values.shape),
        "finite_fraction": float(np.mean(finite)),
        "minimum": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "row_median_standard_deviation": float(
            np.std(np.median(values, axis=1))
        ),
        "column_median_standard_deviation": float(
            np.std(np.median(values, axis=0))
        ),
        "median_adjacent_row_absolute_difference": float(
            np.median(np.abs(np.diff(values, axis=0)))
        ),
        "median_adjacent_column_absolute_difference": float(
            np.median(np.abs(np.diff(values, axis=1)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--access-state", type=Path, default=DEFAULT_ACCESS_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Pilot inspection")
    registry = read_json(args.registry)
    archive_record = registry["archive"]
    if args.archive.stat().st_size != archive_record["size_bytes"]:
        raise ValueError("archive size changed after registration")
    if sha256_file(args.archive) != archive_record["sha256"]:
        raise ValueError("archive SHA-256 changed after registration")
    receipt = claim_role_access(
        registry_path=args.registry,
        access_state_path=args.access_state,
        role="pilot",
        actor="Codex Stage-6R external validation",
        purpose="freeze ElectroSense value semantics and adapter only",
    )
    site_rows: list[dict] = []
    for metadata in role_members(registry, "pilot"):
        values = load_npy_member(args.archive, metadata["member"])
        site_rows.append(
            {
                "site": metadata["site"],
                "member": metadata["member"],
                "frequency_low_mhz": metadata["frequency_low_mhz"],
                "frequency_high_mhz": metadata["frequency_high_mhz"],
                "diagnostics": _diagnostics(values),
            }
        )
    result = {
        "schema_version": "stage6r-electrosense-pilot-inspection-v1",
        "access_receipt": receipt,
        "site_count": len(site_rows),
        "sites": site_rows,
        "scope_boundary": (
            "Only Pilot signal values were read. Stage-6 Final, confirmation "
            "lockbox, reserve, and structural-reserve values remain unread."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
