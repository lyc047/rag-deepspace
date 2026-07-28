#!/usr/bin/env python
"""Materialize a leakage-audited C1 calibration set and opaque temporal holdout."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    AerpawZipPair,
    channel_occupancy,
    channel_power_dbm,
    discover_aerpaw_zip_pairs,
    extract_aerpaw_zip_pairs,
    filter_aerpaw_pairs_by_local_window,
    load_aerpaw_power_sweep,
    pair_datetime_utc,
    robust_power_threshold_dbm,
    select_frequency_band,
    sha256_file,
    thin_aerpaw_pairs_by_time,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


def evenly_spaced(items: list, count: int) -> list:
    if count < 1 or len(items) < count:
        raise ValueError(f"requested {count} scenes from only {len(items)} eligible scenes")
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    if len(set(int(index) for index in indices)) != count:
        raise RuntimeError("deterministic even spacing repeated an index")
    return [items[int(index)] for index in indices]


def pilot_stems(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {Path(row["meta_path"]).name.removesuffix(".sigmf-meta") for row in value["scenes"]}


def original_final_stems(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    stems = set()
    for scene in value["scenes"]:
        for provenance in scene["provenance_ids"]:
            stems.add(provenance.rsplit(":", 1)[-1])
    return stems


def select_partition(pairs_by_site: dict[str, list[AerpawZipPair]], rows: list[dict], excluded: set[str]):
    selected = []
    audit = []
    for row in rows:
        window = filter_aerpaw_pairs_by_local_window(
            pairs_by_site[row["site"]],
            start_local_inclusive=row["start_local_inclusive"],
            end_local_exclusive=row["end_local_exclusive"],
        )
        eligible = [pair for pair in window if pair.stem not in excluded]
        thinned = thin_aerpaw_pairs_by_time(eligible, min_separation_s=float(row["minimum_separation_s"]))
        chosen = evenly_spaced(thinned, int(row["selected_count"]))
        selected.extend((row, pair) for pair in chosen)
        audit.append(
            {
                **row,
                "raw_window_pair_count": len(window),
                "permanent_exclusion_count": len(window) - len(eligible),
                "eligible_after_thinning": len(thinned),
                "selected_count_observed": len(chosen),
                "first_selected_local": chosen[0].timestamp_local,
                "last_selected_local": chosen[-1].timestamp_local,
            }
        )
    return selected, audit


def scene_record(split: str, row: dict, pair, source_files: list[dict]) -> dict:
    local = datetime.fromisoformat(pair.timestamp_local)
    return {
        "scene_id": f"aerpaw-c1-temporal:{row['site']}:{pair.stem}",
        "split": split,
        "stratum": row.get("stratum", "calibration"),
        "site": row["site"],
        "timestamp_local_america_new_york": pair.timestamp_local,
        "timestamp_utc": pair_datetime_utc(pair).isoformat().replace("+00:00", "Z"),
        "cluster_id": f"{row['site']}:{local.strftime('%Y%m%dT%H')}",
        "source_stem": pair.stem,
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/aerpaw_c1_temporal_holdout_protocol_v1.json")
    parser.add_argument("--compact-root", type=Path, default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\c1_temporal_compact_v1"))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/aerpaw_c1_temporal_prefinal_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    access = json.loads((PROJECT_DIR / "configs/stage4_c1_temporal_access_state.json").read_text(encoding="utf-8"))
    if access.get("status") != "not_accessed" or access.get("access_count") != 0:
        raise RuntimeError("temporal holdout preparation is forbidden after its access is consumed")
    old_access = json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    if old_access.get("status") != "access_consumed" or old_access.get("access_count") != 1:
        raise RuntimeError("the original three-site Final access receipt is not in its expected immutable state")

    exclusions = protocol["permanent_exclusions"]
    pilot_path = Path(exclusions["pilot_manifest"])
    final_path = PROJECT_DIR / exclusions["original_final_registry"]
    if sha256_file(pilot_path) != exclusions["pilot_manifest_sha256"]:
        raise ValueError("pilot manifest hash mismatch")
    if sha256_file(final_path) != exclusions["original_final_registry_file_sha256"]:
        raise ValueError("original Final registry hash mismatch")
    excluded = pilot_stems(pilot_path) | original_final_stems(final_path)

    pairs_by_site = {}
    archive_audit = {}
    for site, declared in protocol["official_archives"].items():
        path = Path(declared["path"])
        if path.stat().st_size != int(declared["size_bytes"]):
            raise ValueError(f"{site} archive size mismatch")
        digest = sha256_file(path)
        if digest != declared["sha256"]:
            raise ValueError(f"{site} archive SHA-256 mismatch")
        pairs, errors = discover_aerpaw_zip_pairs(path)
        if errors:
            raise ValueError(f"{site} pair discovery errors: {'; '.join(errors[:10])}")
        pairs_by_site[site] = pairs
        archive_audit[site] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "pair_count": len(pairs),
            "first_local": pairs[0].timestamp_local,
            "last_local": pairs[-1].timestamp_local,
        }

    calibration, calibration_audit = select_partition(
        pairs_by_site, protocol["partitions"]["calibration"], excluded
    )
    holdout, holdout_audit = select_partition(
        pairs_by_site, protocol["partitions"]["confirmation_holdout"], excluded
    )
    chosen_stems = [pair.stem for _, pair in calibration + holdout]
    if len(set(chosen_stems)) != len(chosen_stems) or set(chosen_stems) & excluded:
        raise ValueError("partition overlap or permanent-exclusion leakage detected")

    proxy = protocol["resource_proxy"]
    records = {"calibration": [], "confirmation_holdout": []}
    calibration_arrays = {"scene_ids": [], "base_occupancy": [], "channel_power_dbm": [], "cluster_ids": []}
    for split, entries in (("calibration", calibration), ("confirmation_holdout", holdout)):
        by_site: dict[str, list] = {}
        row_by_stem = {}
        for row, pair in entries:
            by_site.setdefault(row["site"], []).append(pair)
            row_by_stem[pair.stem] = row
        for site, pairs in by_site.items():
            extracted = extract_aerpaw_zip_pairs(
                pairs, destination=args.compact_root / split, site=site
            )
            for pair in extracted:
                files = [
                    {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for role, path in (("meta", pair.meta_path), ("power", pair.data_path))
                ]
                record = scene_record(split, row_by_stem[pair.stem], pair, files)
                records[split].append(record)
                if split == "calibration":
                    sweep = load_aerpaw_power_sweep(pair.meta_path, pair.data_path)
                    if sweep.site != site or np.any((sweep.powers_dbm < -200.0) | (sweep.powers_dbm > 20.0)):
                        raise ValueError(f"calibration sweep integrity failure: {pair.stem}")
                    band = select_frequency_band(
                        sweep,
                        low_mhz=float(proxy["frequency_low_mhz_inclusive"]),
                        high_mhz=float(proxy["frequency_high_mhz_exclusive"]),
                    )
                    threshold = robust_power_threshold_dbm(
                        band,
                        sigma_multiplier=float(proxy["threshold_sigma_multiplier"]),
                        gaussian_mad_scale=float(proxy["threshold_gaussian_mad_scale"]),
                        sigma_floor_db=float(proxy["threshold_sigma_floor_db"]),
                    )
                    calibration_arrays["scene_ids"].append(record["scene_id"])
                    calibration_arrays["base_occupancy"].append(channel_occupancy(band, int(proxy["n_equal_width_channels"]), threshold))
                    calibration_arrays["channel_power_dbm"].append(channel_power_dbm(band, int(proxy["n_equal_width_channels"])))
                    calibration_arrays["cluster_ids"].append(record["cluster_id"])

    for split in records:
        records[split].sort(key=lambda item: (item["timestamp_utc"], item["site"], item["scene_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "temporal_calibration_cache.npz"
    np.savez_compressed(
        cache_path,
        scene_ids=np.asarray(calibration_arrays["scene_ids"]),
        base_occupancy=np.asarray(calibration_arrays["base_occupancy"], dtype=np.float32),
        channel_power_dbm=np.asarray(calibration_arrays["channel_power_dbm"], dtype=np.float32),
        cluster_ids=np.asarray(calibration_arrays["cluster_ids"]),
    )
    inventory = {
        "version": "1.0",
        "status": "calibration_materialized_holdout_opaque_not_accessed",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "archive_audit": archive_audit,
        "partition_audit": {"calibration": calibration_audit, "confirmation_holdout": holdout_audit},
        "calibration_cache": str(cache_path.resolve()),
        "calibration_cache_sha256": sha256_file(cache_path),
        "calibration_scene_count": len(records["calibration"]),
        "confirmation_scene_count": len(records["confirmation_holdout"]),
        "confirmation_cluster_count": len({row["cluster_id"] for row in records["confirmation_holdout"]}),
        "scenes": records,
        "model_imported_or_executed": False,
        "confirmation_measurement_values_interpreted": False,
        "original_final_reused": False,
        "claim_boundary": "Data integrity and time-split audit only; no confirmation model output or performance metric was computed."
    }
    output = args.out_dir / "prefinal_inventory.json"
    atomic_write_json(output, inventory)
    print(json.dumps({"output": str(output), "calibration": len(records["calibration"]), "confirmation": len(records["confirmation_holdout"]), "clusters": inventory["confirmation_cluster_count"], "confirmation_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
