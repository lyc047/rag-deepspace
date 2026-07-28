#!/usr/bin/env python
"""Verify three AERPAW archives and build a compact pre-final inventory.

Only ZIP central directories are used for timestamp matching.  After the
predeclared 30-minute thinning and deterministic even-spacing rule selects 200
scenes, only those members are extracted and schema-validated.  No learned
model is imported or executed, no final catalog is created, and final access is
not consumed.
"""

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
    align_aerpaw_pairs_by_timestamp,
    channel_occupancy,
    discover_aerpaw_zip_pairs,
    extract_aligned_archive_scenes,
    robust_power_threshold_dbm,
    select_frequency_band,
    sha256_file,
    thin_aligned_scenes_by_time,
    validate_aligned_power_scene,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


DEFAULT_ARCHIVES = {
    "LW1": Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsLW1Feb2022_SigMF.zip"),
    "CC1": Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsCC1Feb2022_SigMF.zip"),
    "CC2": Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsCC2Feb2022_SigMF.zip"),
}


def parse_archive_overrides(values: list[str]) -> dict[str, Path]:
    paths = dict(DEFAULT_ARCHIVES)
    for value in values:
        if "=" not in value:
            raise ValueError("--archive entries must use SITE=PATH")
        site, path = value.split("=", 1)
        if site not in paths:
            raise ValueError(f"unknown AERPAW site: {site}")
        paths[site] = Path(path)
    return paths


def evenly_spaced(items: list, count: int) -> list:
    if len(items) < count or count < 1:
        raise ValueError("not enough items for deterministic even spacing")
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    if len(set(int(value) for value in indices)) != count:
        raise RuntimeError("even-spacing unexpectedly repeated an index")
    return [items[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", default=[], metavar="SITE=PATH")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\work\aerpaw_lw1_pilot_manifest_v1.json"),
    )
    parser.add_argument(
        "--compact-root",
        type=Path,
        default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\final_compact"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results/stage4/aerpaw_three_site_prefinal_v1/prefinal_inventory.json",
    )
    args = parser.parse_args()

    access_path = PROJECT_DIR / "configs/stage4_final_access_state.json"
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access.get("access_count") != 0 or access.get("status") != "not_accessed":
        raise RuntimeError("pre-final preparation is forbidden after final access")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    sites = tuple(protocol["sites"])
    archive_paths = parse_archive_overrides(args.archive)

    archive_audit = {}
    pairs_by_site = {}
    for site in sites:
        path = archive_paths[site].resolve()
        declared = protocol["official_archives"][site]
        if not path.is_file():
            raise FileNotFoundError(f"{site} archive is not complete: {path}")
        if path.name.endswith(".crdownload") or path.name.endswith(".aria2"):
            raise ValueError(f"{site} archive is still a partial download: {path}")
        actual_size = path.stat().st_size
        if actual_size != int(declared["size_bytes"]):
            raise ValueError(f"{site} archive size mismatch: {actual_size} != {declared['size_bytes']}")
        digest = sha256_file(path)
        if digest != declared["sha256"]:
            raise ValueError(f"{site} archive SHA-256 mismatch: {digest}")
        pairs, errors = discover_aerpaw_zip_pairs(path)
        if errors:
            raise ValueError(f"{site} ZIP pair discovery failed: {'; '.join(errors[:20])}")
        pairs_by_site[site] = pairs
        archive_audit[site] = {
            "path": str(path),
            "size_bytes": actual_size,
            "sha256": digest,
            "complete_pair_count": len(pairs),
            "first_local_timestamp": pairs[0].timestamp_local,
            "last_local_timestamp": pairs[-1].timestamp_local,
        }

    alignment = protocol["three_site_alignment"]
    aligned, alignment_report = align_aerpaw_pairs_by_timestamp(
        pairs_by_site,
        expected_sites=sites,
        anchor_site=protocol["anchor_site"],
        max_offset_s=float(alignment["max_absolute_offset_from_anchor_s"]),
        local_timezone=protocol["campaign_timezone"],
    )
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    if sha256_file(args.pilot_manifest) != protocol["pilot_governance"]["manifest_sha256"]:
        raise ValueError("pilot manifest differs from the frozen protocol")
    pilot_stems = {Path(scene["meta_path"]).name.removesuffix(".sigmf-meta") for scene in pilot["scenes"]}
    nonpilot = [
        scene
        for scene in aligned
        if scene.pairs_by_site[protocol["anchor_site"]].stem not in pilot_stems
    ]
    thinned = thin_aligned_scenes_by_time(
        nonpilot,
        min_separation_s=float(protocol["scene_independence"]["minimum_utc_separation_s"]),
    )
    minimum = int(protocol["scene_independence"]["minimum_scene_count"])
    if len(thinned) < minimum:
        raise ValueError(f"only {len(thinned)} aligned, thinned scenes; protocol requires {minimum}")
    selected = evenly_spaced(thinned, minimum)
    compact = extract_aligned_archive_scenes(
        selected,
        destination=args.compact_root,
        expected_sites=sites,
    )

    proxy = protocol["resource_proxy"]
    threshold_rule = proxy["threshold_rule"]
    inventory = []
    for scene in compact:
        offsets = [float(value) for value in scene.offsets_from_anchor_s.values()]
        if max(offsets) - min(offsets) > float(alignment["max_three_site_span_s"]):
            raise ValueError(f"scene exceeds three-site span gate: {scene.scene_id}")
        sweeps, errors = validate_aligned_power_scene(
            scene,
            expected_sites=sites,
            max_offset_s=float(alignment["max_absolute_offset_from_anchor_s"]),
            frequency_atol_mhz=float(alignment["frequency_grid_atol_mhz"]),
            plausible_power_range_dbm=tuple(protocol["integrity_gates_before_any_final_registration"]["power_range_dbm"]),
        )
        if errors:
            raise ValueError(f"{scene.scene_id} failed compact validation: {'; '.join(errors)}")
        sources = []
        for site in sites:
            sweep = sweeps[site]
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
            occupancy = channel_occupancy(band, int(proxy["n_equal_width_channels"]), threshold)
            if not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
                raise ValueError(f"{scene.scene_id}/{site} proxy mapping is invalid")
            for role, path in (("meta", sweep.meta_path), ("power", sweep.data_path)):
                sources.append(
                    {
                        "site": site,
                        "role": role,
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        inventory.append(
            {
                "scene_id": scene.scene_id,
                "anchor_datetime_utc": scene.anchor_datetime_utc,
                "independence_group_id": "aerpaw-30min:" + datetime.fromisoformat(
                    scene.anchor_datetime_utc.replace("Z", "+00:00")
                ).strftime("%Y%m%dT%H%MZ"),
                "collection_event_id": "aerpaw-sweep-triplet:" + scene.scene_id.rsplit(":", 1)[-1],
                "offsets_from_anchor_s": dict(scene.offsets_from_anchor_s),
                "stems_by_site": {site: scene.pairs_by_site[site].stem for site in sites},
                "sources": sources,
                "schema_and_proxy_mapping_validated_without_model": True,
            }
        )

    result = {
        "audit_version": "1.0",
        "status": "prefinal_inventory_passed_not_a_final_catalog",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "archive_audit": archive_audit,
        "alignment_report": alignment_report,
        "aligned_before_pilot_exclusion": len(aligned),
        "pilot_overlap_removed": len(aligned) - len(nonpilot),
        "aligned_after_30min_thinning": len(thinned),
        "selected_compact_scene_count": len(inventory),
        "selection_rule": protocol["scene_independence"]["selection_rule"],
        "compact_root": str(args.compact_root.resolve()),
        "scenes": inventory,
        "model_imported_or_executed": False,
        "final_catalog_created": False,
        "final_access_consumed": False,
        "claim_boundary": "Integrity, alignment, and proxy-interface audit only; no method performance was computed.",
    }
    atomic_write_json(args.output, result)
    report = args.output.with_name("prefinal_inventory_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# AERPAW Three-Site Pre-final Inventory",
                "",
                f"- Selected compact scenes: {len(inventory)}",
                f"- Aligned before thinning: {len(nonpilot)}",
                f"- Aligned after 30-minute thinning: {len(thinned)}",
                "- Model execution: no",
                "- Final catalog creation: no",
                "- Final access consumed: no",
                "",
                "This report is an integrity/alignment gate, not a performance result.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "scenes": len(inventory), "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
