#!/usr/bin/env python
"""Build deterministic proxy-label sidecars and a candidate final catalog.

The command is gated on a passed three-site pre-final inventory.  It evaluates
no learned method and does not register or consume the final holdout.  Its
output must still pass ``register_stage4_final_catalog.py`` before use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    channel_occupancy,
    channel_power_dbm,
    load_aerpaw_power_sweep,
    robust_power_threshold_dbm,
    select_frequency_band,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json, validate_final_catalog  # noqa: E402


def site_source_paths(scene: dict, site: str) -> tuple[Path, Path]:
    rows = [row for row in scene["sources"] if row["site"] == site]
    by_role = {row["role"]: Path(row["path"]) for row in rows}
    if set(by_role) != {"meta", "power"}:
        raise ValueError(f"{scene['scene_id']}/{site} must have exactly meta and power sources")
    return by_role["meta"], by_role["power"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_DIR / "results/stage4/aerpaw_three_site_prefinal_v1/prefinal_inventory.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json",
    )
    parser.add_argument(
        "--development-registry",
        type=Path,
        default=PROJECT_DIR / "configs/stage4_development_registry_v1.json",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\final_compact\labels"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results/stage4/aerpaw_three_site_prefinal_v1/aerpaw_candidate_final_catalog.json",
    )
    args = parser.parse_args()

    access = json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    if access.get("access_count") != 0 or access.get("status") != "not_accessed":
        raise RuntimeError("candidate catalog construction is forbidden after final access")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    development = json.loads(args.development_registry.read_text(encoding="utf-8"))
    if inventory.get("status") != "prefinal_inventory_passed_not_a_final_catalog":
        raise ValueError("three-site pre-final inventory has not passed")
    if inventory.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("pre-final inventory was built under a different protocol hash")
    if inventory.get("final_access_consumed") is not False or inventory.get("model_imported_or_executed") is not False:
        raise ValueError("pre-final inventory has an invalid governance state")

    sites = tuple(protocol["sites"])
    proxy = protocol["resource_proxy"]
    threshold_rule = proxy["threshold_rule"]
    args.label_root.mkdir(parents=True, exist_ok=True)
    catalog_scenes = []
    for scene in inventory["scenes"]:
        per_site = {}
        for site in sites:
            meta_path, data_path = site_source_paths(scene, site)
            declared = {(row["site"], row["role"]): row for row in scene["sources"]}
            for role, path in (("meta", meta_path), ("power", data_path)):
                if sha256_file(path) != declared[(site, role)]["sha256"]:
                    raise ValueError(f"source changed after pre-final audit: {path}")
            sweep = load_aerpaw_power_sweep(meta_path, data_path)
            band = select_frequency_band(
                sweep,
                low_mhz=float(proxy["frequency_low_mhz_inclusive"]),
                high_mhz=float(proxy["frequency_high_mhz_exclusive"]),
            )
            threshold = robust_power_threshold_dbm(
                band,
                sigma_multiplier=float(threshold_rule["sigma_multiplier"]),
                gaussian_mad_scale=float(threshold_rule["gaussian_mad_scale"]),
                sigma_floor_db=float(threshold_rule["sigma_floor_db"]),
            )
            per_site[site] = {
                "threshold_dbm": threshold,
                "occupancy_proxy": channel_occupancy(
                    band, int(proxy["n_equal_width_channels"]), threshold
                ).tolist(),
                "channel_mean_power_dbm": channel_power_dbm(
                    band, int(proxy["n_equal_width_channels"]), reducer="mean"
                ).tolist(),
            }
        reference = np.mean(
            np.asarray([per_site[site]["occupancy_proxy"] for site in sites], dtype=np.float64), axis=0
        )
        label = {
            "schema": "aerpaw_stage4_proxy_label_v1",
            "scene_id": scene["scene_id"],
            "protocol_sha256": sha256_file(args.protocol),
            "frequency_band_mhz": [
                proxy["frequency_low_mhz_inclusive"],
                proxy["frequency_high_mhz_exclusive"],
            ],
            "per_site": per_site,
            "three_site_reference_occupancy_proxy": reference.tolist(),
            "C1_reference_site": "LW1",
            "label_is_independently_annotated": False,
            "derivation": "Frozen deterministic received-power mapping; no learned-model output.",
            "claim_boundary": proxy["label_boundary"],
        }
        label_path = args.label_root / (scene["scene_id"].replace(":", "_") + ".json")
        atomic_write_json(label_path, label)
        catalog_scenes.append(
            {
                "scene_id": scene["scene_id"],
                "independence_group_id": scene["independence_group_id"],
                "collection_event_id": scene["collection_event_id"],
                "start_time_utc": scene["anchor_datetime_utc"],
                "receiver_ids": list(sites),
                "provenance_ids": [
                    f"dryad:hmgqnk9zn:{site}:{scene['stems_by_site'][site]}" for site in sites
                ],
                "source_files": [
                    {"path": row["path"], "sha256": row["sha256"], "site": row["site"], "role": row["role"]}
                    for row in scene["sources"]
                ],
                "label_ref": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "integrity_status": "passed_without_model_output_access",
                "external_validation_scope": "real_power_sweep_proxy_three_fixed_sites",
            }
        )

    catalog = {
        "catalog_id": "aerpaw-fixed-sites-feb2022-proxy-final-v1",
        "status": "candidate_final_catalog",
        "created_date": "2026-07-21",
        "dataset_doi": protocol["dataset_doi"],
        "protocol_sha256": sha256_file(args.protocol),
        "prefinal_inventory_sha256": sha256_file(args.inventory),
        "scenes": catalog_scenes,
        "governance": {
            "learned_model_executed": False,
            "final_access_consumed": False,
            "must_be_registered_before_use": True,
            "proxy_label_boundary": proxy["label_boundary"],
        },
    }
    errors = validate_final_catalog(catalog, development, minimum_scenes=int(protocol["scene_independence"]["minimum_scene_count"]))
    if errors:
        raise ValueError("candidate final catalog validation failed: " + "; ".join(errors))
    atomic_write_json(args.output, catalog)
    print(json.dumps({"output": str(args.output), "scene_count": len(catalog_scenes), "registered": False, "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
