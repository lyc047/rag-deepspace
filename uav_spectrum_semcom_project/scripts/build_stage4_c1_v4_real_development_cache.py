#!/usr/bin/env python
"""Build time-blocked real-power development caches for C1-v4 without prior holdout reuse."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    discover_aerpaw_zip_pairs,
    filter_aerpaw_pairs_by_local_window,
    sha256_file,
    thin_aerpaw_pairs_by_time,
)
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    build_block_semantic_scene,
    load_aerpaw_zip_power_sweep,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


def evenly_spaced(items: list, count: int) -> list:
    if count < 1 or len(items) < count:
        raise ValueError(f"cannot select {count} values from {len(items)} eligible values")
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    if len(set(int(value) for value in indices)) != count:
        raise RuntimeError("even spacing repeated an index")
    return [items[int(value)] for value in indices]


def collect_excluded_stems(protocol: dict) -> set[str]:
    exclusions = protocol["permanent_exclusions"]
    stems = set(exclusions["pilot_stems"])
    pilot_path = Path(exclusions["pilot_manifest"])
    if pilot_path.is_file():
        if sha256_file(pilot_path) != exclusions["pilot_manifest_sha256"]:
            raise ValueError("pilot manifest hash mismatch")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        observed = {Path(row["meta_path"]).name.removesuffix(".sigmf-meta") for row in pilot["scenes"]}
        if observed != stems:
            raise ValueError("embedded pilot stem set differs from the original manifest")
    final_path = PROJECT_DIR / exclusions["original_final_registry"]
    if sha256_file(final_path) != exclusions["original_final_registry_sha256"]:
        raise ValueError("original Final registry hash mismatch")
    original = json.loads(final_path.read_text(encoding="utf-8"))
    for scene in original["scenes"]:
        stems.update(value.rsplit(":", 1)[-1] for value in scene["provenance_ids"])
    temporal_path = PROJECT_DIR / exclusions["c1_temporal_inventory"]
    if sha256_file(temporal_path) != exclusions["c1_temporal_inventory_sha256"]:
        raise ValueError("C1 temporal inventory hash mismatch")
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    for rows in temporal["scenes"].values():
        stems.update(row["source_stem"] for row in rows)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_development_protocol.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_real_cache_v1")
    parser.add_argument("--archive", action="append", default=[], metavar="SITE=PATH")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    old_access = json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    temporal_access = json.loads((PROJECT_DIR / "configs/stage4_c1_temporal_access_state.json").read_text(encoding="utf-8"))
    if old_access.get("access_count") != 1 or temporal_access.get("access_count") != 1:
        raise RuntimeError("prior access receipts are not in their expected immutable states")
    verified_path = PROJECT_DIR / protocol["verified_archive_audit"]["path"]
    if sha256_file(verified_path) != protocol["verified_archive_audit"]["sha256"]:
        raise ValueError("verified archive audit artifact changed")
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    excluded = collect_excluded_stems(protocol)
    archive_paths = {site: Path(path) for site, path in protocol["archives"].items()}
    for value in args.archive:
        if "=" not in value:
            raise ValueError("--archive must use SITE=PATH")
        site, path = value.split("=", 1)
        if site not in archive_paths:
            raise ValueError(f"unknown archive site: {site}")
        archive_paths[site] = Path(path)
    pairs_by_site = {}
    archive_audit = {}
    for site, configured_path in archive_paths.items():
        path = configured_path.resolve()
        declared = verified["archive_audit"][site]
        if not path.is_file():
            raise FileNotFoundError(f"{site} archive is unavailable; reconnect or override its path: {path}")
        if path.name != Path(declared["path"]).name or path.stat().st_size != int(declared["size_bytes"]):
            raise ValueError(f"{site} archive filename or size differs from the verified audit")
        if path != Path(declared["path"]).resolve() and sha256_file(path) != declared["sha256"]:
            raise ValueError(f"relocated {site} archive SHA-256 mismatch")
        pairs, errors = discover_aerpaw_zip_pairs(path)
        if errors:
            raise ValueError(f"{site} central-directory pair errors: {'; '.join(errors[:10])}")
        pairs_by_site[site] = pairs
        archive_audit[site] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "verified_sha256": declared["sha256"],
            "pair_count": len(pairs),
        }

    selections = {}
    selection_audit = {}
    used: set[str] = set()
    for split, rule in protocol["splits"].items():
        selections[split] = []
        selection_audit[split] = []
        for site in protocol["archives"]:
            window = filter_aerpaw_pairs_by_local_window(
                pairs_by_site[site],
                start_local_inclusive=rule["start_local_inclusive"],
                end_local_exclusive=rule["end_local_exclusive"],
            )
            eligible = [pair for pair in window if pair.stem not in excluded and pair.stem not in used]
            thinned = thin_aerpaw_pairs_by_time(eligible, min_separation_s=float(rule["minimum_separation_s"]))
            chosen = evenly_spaced(thinned, int(rule["selected_per_site"]))
            selections[split].extend((site, pair) for pair in chosen)
            used.update(pair.stem for pair in chosen)
            selection_audit[split].append(
                {
                    "site": site,
                    "raw_window_pairs": len(window),
                    "permanent_exclusions": len(window) - len(eligible),
                    "after_thinning": len(thinned),
                    "selected": len(chosen),
                    "first_local": chosen[0].timestamp_local,
                    "last_local": chosen[-1].timestamp_local,
                }
            )
    if used & excluded or len(used) != sum(len(rows) for rows in selections.values()):
        raise ValueError("development split overlap or permanent-exclusion leakage")

    task = protocol["resource_task"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {site: zipfile.ZipFile(path) for site, path in archive_paths.items()}
    split_results = {}
    try:
        for split, entries in selections.items():
            arrays = {
                "scene_ids": [], "site_ids": [], "timestamps_local": [], "cluster_ids": [],
                "features": [], "occupancy": [], "channel_power_dbm": [], "normalized_channel_power": [],
                "block_cost_dbm": [], "normalized_block_cost": [], "best_block_indicator": [],
            }
            provenance = []
            for index, (site, pair) in enumerate(sorted(entries, key=lambda item: (item[1].timestamp_local, item[0], item[1].stem))):
                sweep = load_aerpaw_zip_power_sweep(pair, handles[site])
                if sweep.site != site or np.any((sweep.powers_dbm < -200.0) | (sweep.powers_dbm > 20.0)):
                    raise ValueError(f"invalid real-power sweep: {site}/{pair.stem}")
                scene = build_block_semantic_scene(
                    sweep,
                    frequency_low_mhz=float(task["frequency_low_mhz"]),
                    frequency_high_mhz=float(task["frequency_high_mhz"]),
                    n_channels=int(task["n_channels"]),
                    demand_channels=int(task["demand_channels"]),
                    sigma_multiplier=float(task["threshold_sigma_multiplier"]),
                    gaussian_mad_scale=float(task["threshold_gaussian_mad_scale"]),
                    sigma_floor_db=float(task["threshold_sigma_floor_db"]),
                )
                scene_id = f"c1-v4-dev:{site}:{pair.stem}"
                timestamp = datetime.fromisoformat(pair.timestamp_local)
                arrays["scene_ids"].append(scene_id); arrays["site_ids"].append(site)
                arrays["timestamps_local"].append(pair.timestamp_local)
                arrays["cluster_ids"].append(f"{site}:{timestamp.strftime('%Y%m%dT%H')}")
                for name in ("features", "occupancy", "channel_power_dbm", "normalized_channel_power", "block_cost_dbm", "normalized_block_cost", "best_block_indicator"):
                    arrays[name].append(getattr(scene, name))
                meta_info = handles[site].getinfo(pair.meta_member); data_info = handles[site].getinfo(pair.data_member)
                provenance.append(
                    {
                        "scene_id": scene_id, "site": site, "source_stem": pair.stem,
                        "archive_sha256": archive_audit[site]["verified_sha256"],
                        "meta_member": pair.meta_member, "meta_size": meta_info.file_size, "meta_crc32": f"{meta_info.CRC:08x}",
                        "data_member": pair.data_member, "data_size": data_info.file_size, "data_crc32": f"{data_info.CRC:08x}",
                    }
                )
                if (index + 1) % 30 == 0:
                    print(f"{split}: processed {index + 1}/{len(entries)}", flush=True)
            cache_path = args.out_dir / f"c1_v4_{split}.npz"
            np.savez_compressed(
                cache_path,
                **{
                    key: np.asarray(value) if key in {"scene_ids", "site_ids", "timestamps_local", "cluster_ids"} else np.asarray(value, dtype=np.float32)
                    for key, value in arrays.items()
                },
            )
            provenance_path = args.out_dir / f"c1_v4_{split}_provenance.json"
            atomic_write_json(provenance_path, {"split": split, "scenes": provenance})
            split_results[split] = {
                "scene_count": len(entries),
                "site_counts": {site: sum(1 for value, _ in entries if value == site) for site in protocol["archives"]},
                "cluster_count": len(set(arrays["cluster_ids"])),
                "cache": str(cache_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "cache_sha256": sha256_file(cache_path),
                "provenance": str(provenance_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "provenance_sha256": sha256_file(provenance_path),
            }
    finally:
        for handle in handles.values():
            handle.close()

    result = {
        "version": "1.0",
        "status": "c1_v4_real_development_cache_complete",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "verified_archive_audit_sha256": sha256_file(verified_path),
        "archive_audit": archive_audit,
        "selection_audit": selection_audit,
        "splits": split_results,
        "governance": {
            "prior_pilot_or_holdout_stem_overlap": 0,
            "prior_holdout_measurement_values_loaded": False,
            "prior_holdout_metrics_loaded": False,
            "model_outputs_loaded": False,
            "development_only": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = args.out_dir / "c1_v4_cache_result.json"
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "splits": {key: value["scene_count"] for key, value in split_results.items()}, "prior_holdout_values_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
