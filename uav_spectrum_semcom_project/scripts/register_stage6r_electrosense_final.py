#!/usr/bin/env python
"""Register ElectroSense metadata without reading any PSD signal values."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import (  # noqa: E402
    SPLIT_SALT,
    deterministic_site_split,
    inventory_archive,
    select_site_members,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


DEFAULT_ARCHIVE = Path(r"F:\spectrum_bands.tar.gz")
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/electrosense_final_registry_v1/registry.json"
)
DEFAULT_ACCESS_STATE = (
    PROJECT_DIR / "configs/stage6r_electrosense_final_access_state_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--access-state", type=Path, default=DEFAULT_ACCESS_STATE)
    args = parser.parse_args()
    if args.output.exists() or args.access_state.exists():
        raise FileExistsError("refusing to overwrite registry or access state")
    inventory = inventory_archive(args.archive)
    selected = select_site_members(inventory)
    split = deterministic_site_split(selected)
    all_sites = sorted({row.site for row in inventory})
    noneligible = sorted(set(all_sites) - set(selected))
    registry = {
        "schema_version": "stage6r-electrosense-registry-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only_registration": True,
        "signal_values_accessed": False,
        "archive": {
            "path": str(args.archive),
            "size_bytes": args.archive.stat().st_size,
            "sha256": sha256_file(args.archive),
        },
        "dataset": {
            "name": "CrowdSpectrumizer spectrum_bands",
            "source": "Zenodo record 7521246",
            "technology": "fm",
            "selection_rule": (
                "lexicographically first valid FM member per site; "
                "shape rows>=200 and columns>=64"
            ),
            "npy_member_count": len(inventory),
            "observed_site_count": len(all_sites),
            "eligible_site_count": len(selected),
            "noneligible_sites": noneligible,
        },
        "split": {
            "method": "ascending SHA-256 digest of '<salt>|<site>'",
            "salt": SPLIT_SALT,
            "roles": split,
            "structural_reserve_sites": noneligible,
        },
        "selected_site_members": {
            site: row.to_json() for site, row in selected.items()
        },
        "inventory": [row.to_json() for row in inventory],
    }
    access_state = {
        "schema_version": "stage6r-electrosense-access-state-v1",
        "registry_sha256": None,
        "roles": {
            role: {
                "access_count": 0,
                "signal_values_accessed": False,
                "first_access_at_utc": None,
                "purpose": {
                    "pilot": "adapter validation only",
                    "stage6_final": "single Stage-6R external Final",
                    "confirmation_lockbox": "future independent confirmation",
                    "reserve": "structural reserve only",
                }[role],
            }
            for role in split
        },
        "prohibitions": [
            "Do not reset access_count.",
            "Do not use Final, lockbox, or reserve signal values for tuning.",
            "Do not replace an accessed invalid Final site based on outcome.",
        ],
    }
    atomic_write_json(args.output, registry)
    access_state["registry_sha256"] = sha256_file(args.output)
    atomic_write_json(args.access_state, access_state)
    print(
        json.dumps(
            {
                "registry": str(args.output),
                "access_state": str(args.access_state),
                "npy_member_count": len(inventory),
                "observed_site_count": len(all_sites),
                "eligible_site_count": len(selected),
                "roles": split,
                "signal_values_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

