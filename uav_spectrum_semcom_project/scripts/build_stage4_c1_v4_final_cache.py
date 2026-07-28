#!/usr/bin/env python
"""Load the registered C1-v4 Final measurements after access is consumed."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import AerpawZipPair  # noqa: E402
from spectrum_semcom.c1_v4_block_semantics import build_block_semantic_scene, load_aerpaw_zip_power_sweep  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def verify_binding(protocol_path: Path, inventory_path: Path, snapshot_path: Path, state_path: Path) -> tuple[dict, dict]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if state.get("status") != "access_consumed" or state.get("access_count") != 1:
        raise RuntimeError("Final measurement loading requires the single-use receipt")
    receipt = state.get("receipt", {})
    if receipt.get("protocol_sha256") != sha256_file(protocol_path) or receipt.get("inventory_sha256") != sha256_file(inventory_path):
        raise ValueError("Final receipt is not bound to protocol/inventory")
    if canonical_json_sha256(snapshot.get("files", [])) != snapshot.get("executable_snapshot_sha256"):
        raise ValueError("snapshot digest mismatch")
    if receipt.get("executable_snapshot_sha256") != snapshot.get("executable_snapshot_sha256"):
        raise ValueError("Final receipt is not bound to executable snapshot")
    for row in snapshot["files"]:
        path = PROJECT_DIR / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"frozen file missing or changed: {row['path']}")
    return json.loads(protocol_path.read_text(encoding="utf-8")), json.loads(inventory_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json")
    parser.add_argument("--inventory", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_freeze_v1/executable_snapshot.json")
    parser.add_argument("--state", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_cache_v1")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    output = args.out_dir / "c1_v4_final_cache_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite the single-use Final cache")
    protocol, inventory = verify_binding(args.protocol, args.inventory, args.snapshot, args.state)
    if inventory.get("prior_provenance_overlap") != 0 or inventory.get("measurement_values_loaded") is not False:
        raise ValueError("registered Final inventory is not eligible")
    task = protocol["resource_task"]
    handles = {site: zipfile.ZipFile(Path(value["path"]).resolve()) for site, value in protocol["archives"].items()}
    arrays = {name: [] for name in (
        "scene_ids", "site_ids", "timestamps_local", "cluster_ids", "features", "occupancy",
        "channel_power_dbm", "normalized_channel_power", "block_cost_dbm", "normalized_block_cost", "best_block_indicator",
    )}
    provenance = []
    try:
        for index, row in enumerate(inventory["scenes"]):
            site = row["site"]
            pair = AerpawZipPair(row["source_stem"], Path(protocol["archives"][site]["path"]).resolve(), row["meta_member"], row["data_member"], row["timestamp_local"])
            meta = handles[site].getinfo(pair.meta_member); data = handles[site].getinfo(pair.data_member)
            if meta.file_size != row["meta_size"] or f"{meta.CRC:08x}" != row["meta_crc32"] or data.file_size != row["data_size"] or f"{data.CRC:08x}" != row["data_crc32"]:
                raise ValueError(f"registered ZIP member changed: {site}/{pair.stem}")
            sweep = load_aerpaw_zip_power_sweep(pair, handles[site])
            if sweep.site != site or np.any((sweep.powers_dbm < -200.0) | (sweep.powers_dbm > 20.0)):
                raise ValueError(f"invalid Final real-power sweep: {site}/{pair.stem}")
            scene = build_block_semantic_scene(
                sweep, frequency_low_mhz=float(task["frequency_low_mhz"]), frequency_high_mhz=float(task["frequency_high_mhz"]),
                n_channels=int(task["n_channels"]), demand_channels=int(task["demand_channels"]),
                sigma_multiplier=float(task["threshold_sigma_multiplier"]), gaussian_mad_scale=float(task["threshold_gaussian_mad_scale"]),
                sigma_floor_db=float(task["threshold_sigma_floor_db"]),
            )
            scene_id = f"c1-v4-final:{site}:{pair.stem}"
            arrays["scene_ids"].append(scene_id); arrays["site_ids"].append(site)
            arrays["timestamps_local"].append(pair.timestamp_local); arrays["cluster_ids"].append(row["cluster_id"])
            for name in ("features", "occupancy", "channel_power_dbm", "normalized_channel_power", "block_cost_dbm", "normalized_block_cost", "best_block_indicator"):
                arrays[name].append(getattr(scene, name))
            provenance.append({"scene_id": scene_id, **row, "archive_sha256": protocol["archives"][site]["sha256"]})
            if (index + 1) % 50 == 0:
                print(f"Final cache: processed {index + 1}/{len(inventory['scenes'])}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "c1_v4_final.npz"
    np.savez_compressed(cache_path, **{
        key: np.asarray(value) if key in {"scene_ids", "site_ids", "timestamps_local", "cluster_ids"} else np.asarray(value, dtype=np.float32)
        for key, value in arrays.items()
    })
    provenance_path = args.out_dir / "c1_v4_final_provenance.json"
    atomic_write_json(provenance_path, {"status": "single_use_final_measurements_loaded", "scenes": provenance})
    result = {
        "version": "1.0", "status": "c1_v4_final_cache_complete",
        "protocol_sha256": sha256_file(args.protocol), "inventory_sha256": sha256_file(args.inventory),
        "snapshot_sha256": sha256_file(args.snapshot), "access_receipt_sha256": json.loads(args.state.read_text(encoding="utf-8"))["receipt_sha256"],
        "scene_count": len(provenance), "site_counts": {site: arrays["site_ids"].count(site) for site in protocol["archives"]},
        "cluster_count": len(set(arrays["cluster_ids"])),
        "cache": str(cache_path.relative_to(PROJECT_DIR)).replace("\\", "/"), "cache_sha256": sha256_file(cache_path),
        "provenance": str(provenance_path.relative_to(PROJECT_DIR)).replace("\\", "/"), "provenance_sha256": sha256_file(provenance_path),
        "governance": {"access_count": 1, "prior_provenance_overlap": 0, "measurement_values_loaded_after_freeze": True},
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "scenes": len(provenance), "clusters": result["cluster_count"], "access_count": 1}, indent=2))


if __name__ == "__main__":
    main()
