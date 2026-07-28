#!/usr/bin/env python
"""Build the frozen Stage-6 200-scene catalog from filenames only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage5_final_governance import (  # noqa: E402
    discover_timestamped_sigmf_pairs,
    expected_cluster_id,
    select_spaced_evenly,
)
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    validate_external_final_catalog,
    verify_stage6_code_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_external_final_registry_v1.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_freeze_v1/code_snapshot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_catalog_v1/catalog.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 Final catalog")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    errors = verify_stage6_code_snapshot(PROJECT_DIR, snapshot)
    if errors:
        raise ValueError("code snapshot verification failed: " + "; ".join(errors))
    root = Path(registry["local_root"])
    sampling = protocol["sampling"]
    timezone = ZoneInfo(sampling["local_timezone"])
    scenes: list[dict] = []
    campaign_audit: dict[str, dict] = {}
    for entry in registry["final_candidates"]:
        campaign = entry["dataset_id"]
        archive = root / entry["archive"]
        if (
            not archive.is_file()
            or archive.stat().st_size != int(entry["archive_size_bytes"])
            or sha256_file(archive) != entry["archive_sha256"]
        ):
            raise ValueError(f"registered archive integrity failed: {archive}")
        rows, discovery_errors = discover_timestamped_sigmf_pairs(archive)
        if discovery_errors:
            raise ValueError(
                f"{campaign} member discovery failed: {discovery_errors}"
            )
        selected = select_spaced_evenly(
            rows,
            count=int(sampling["campaign_allocations"][campaign]),
            minimum_spacing_s=float(sampling["minimum_spacing_minutes"]) * 60.0,
        )
        campaign_audit[campaign] = {
            "discovered_pairs": len(rows),
            "selected_scenes": len(selected),
            "archive_sha256": entry["archive_sha256"],
            "signal_values_loaded": False,
            "sigmf_metadata_loaded": False,
        }
        for row in selected:
            local = datetime.fromisoformat(row["timestamp_local"]).replace(
                tzinfo=timezone
            )
            scenes.append(
                {
                    "scene_id": f"{campaign}:{row['stem']}",
                    "campaign_id": campaign,
                    "timestamp_local": row["timestamp_local"],
                    "timestamp_utc": local.astimezone(
                        ZoneInfo("UTC")
                    ).isoformat(),
                    "cluster_id": expected_cluster_id(
                        campaign,
                        row["timestamp_local"],
                        int(sampling["cluster_minutes"]),
                    ),
                    "archive_path": str(archive),
                    "archive_sha256": entry["archive_sha256"],
                    "meta_member": row["meta_member"],
                    "data_member": row["data_member"],
                    "meta_size_bytes": row["meta_size_bytes"],
                    "data_size_bytes": row["data_size_bytes"],
                    "meta_crc32": row["meta_crc32"],
                    "data_crc32": row["data_crc32"],
                    "selection_used_signal_values": False,
                }
            )
    scenes.sort(key=lambda row: (row["campaign_id"], row["timestamp_local"]))
    catalog = {
        "version": "1.0",
        "status": "candidate_external_final_catalog",
        "catalog_id": "stage6_external_final_catalog_v1",
        "protocol_sha256": sha256_file(args.protocol),
        "registry_sha256": sha256_file(args.registry),
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "scene_count": len(scenes),
        "campaign_audit": campaign_audit,
        "scenes": scenes,
        "signal_values_loaded_for_selection": False,
        "method_outputs_loaded_for_selection": False,
        "final_access_consumed": False,
    }
    errors = validate_external_final_catalog(catalog, protocol, registry)
    if errors:
        raise ValueError("Stage-6 Final catalog invalid: " + "; ".join(errors))
    catalog["catalog_sha256"] = canonical_json_sha256(catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, catalog)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_count": len(scenes),
                "catalog_sha256": catalog["catalog_sha256"],
                "signal_values_loaded": False,
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
