#!/usr/bin/env python
"""Audit the permanently excluded LW1 pilot and freeze proxy diagnostics.

This script reads only the preselected pilot power sweeps.  It never imports a
learned model, never creates a final catalog, and aborts unless final access is
still zero.
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
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\extracted"),
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=Path(r"F:\uav_spectrum_final_data\aerpaw_sub6_feb2022\work\aerpaw_lw1_pilot_manifest_v1.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results/stage4/aerpaw_pilot_resource_proxy_v1/pilot_resource_proxy_result.json",
    )
    args = parser.parse_args()

    access = json.loads((PROJECT_DIR / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    if access.get("access_count") != 0 or access.get("status") != "not_accessed":
        raise RuntimeError("pilot freeze is forbidden after final access")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    expected_hash = protocol["pilot_governance"]["manifest_sha256"]
    actual_hash = sha256_file(args.pilot_manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"pilot manifest SHA-256 mismatch: {actual_hash} != {expected_hash}")
    if manifest.get("status") != "pilot_only_permanently_excluded_from_final":
        raise ValueError("manifest is not a permanently excluded pilot")
    if manifest.get("scene_count") != protocol["pilot_governance"]["scene_count"]:
        raise ValueError("pilot scene count differs from frozen protocol")
    if manifest.get("final_access_consumed") is not False:
        raise ValueError("pilot manifest reports final access")

    proxy = protocol["resource_proxy"]
    low = float(proxy["frequency_low_mhz_inclusive"])
    high = float(proxy["frequency_high_mhz_exclusive"])
    n_channels = int(proxy["n_equal_width_channels"])
    threshold_rule = proxy["threshold_rule"]
    thresholds = []
    occupancies = []
    costs = []
    bin_counts = []
    sites = []
    for scene in manifest["scenes"]:
        sweep = load_aerpaw_power_sweep(
            args.dataset_root / scene["meta_path"],
            args.dataset_root / scene["data_path"],
        )
        if sweep.site != protocol["pilot_governance"]["site"]:
            raise ValueError(f"pilot metadata site mismatch: {sweep.site}")
        band = select_frequency_band(sweep, low_mhz=low, high_mhz=high)
        threshold = robust_power_threshold_dbm(
            band,
            sigma_multiplier=float(threshold_rule["sigma_multiplier"]),
            gaussian_mad_scale=float(threshold_rule["gaussian_mad_scale"]),
            sigma_floor_db=float(threshold_rule["sigma_floor_db"]),
        )
        thresholds.append(threshold)
        occupancies.append(channel_occupancy(band, n_channels, threshold))
        costs.append(channel_power_dbm(band, n_channels, reducer="mean"))
        bin_counts.append(band.n_bins)
        sites.append(sweep.site)

    occupancy = np.stack(occupancies).astype(np.float64)
    power = np.stack(costs).astype(np.float64)
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    if not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
        raise RuntimeError("pilot occupancy mapping produced invalid values")
    if not np.all(np.isfinite(power)):
        raise RuntimeError("pilot resource-cost mapping produced invalid values")

    result = {
        "experiment_id": "aerpaw_lw1_pilot_resource_proxy_freeze_v1",
        "role": "pilot_only_mapping_and_interface_freeze_not_model_evaluation",
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "pilot_manifest_path": str(args.pilot_manifest),
        "pilot_manifest_sha256": actual_hash,
        "pilot_scene_count": int(len(occupancy)),
        "pilot_sites": sorted(set(sites)),
        "frequency_band_mhz": [low, high],
        "bins_per_selected_band": sorted(set(bin_counts)),
        "n_equal_width_channels": n_channels,
        "threshold_rule": threshold_rule,
        "threshold_dbm_summary": quantiles(thresholds_array),
        "channel_occupancy_summary": {
            "pooled": quantiles(occupancy),
            "per_channel_median": np.median(occupancy, axis=0).tolist(),
            "per_channel_q10": np.quantile(occupancy, 0.10, axis=0).tolist(),
            "per_channel_q90": np.quantile(occupancy, 0.90, axis=0).tolist(),
        },
        "channel_mean_power_dbm_summary": {
            "pooled": quantiles(power),
            "per_channel_median": np.median(power, axis=0).tolist(),
        },
        "sanity_checks": {
            "all_pilot_sweeps_loaded": len(occupancy) == manifest["scene_count"],
            "single_expected_site": sorted(set(sites)) == [protocol["pilot_governance"]["site"]],
            "selected_band_has_constant_bin_count": len(set(bin_counts)) == 1,
            "occupancy_is_finite_and_bounded": True,
            "resource_cost_is_finite": True,
            "model_imported_or_executed": False,
            "final_catalog_created": False,
            "final_access_consumed": False,
        },
        "final_access_consumed": False,
        "claim_boundary": proxy["label_boundary"],
    }
    atomic_write_json(args.output, result)
    report_path = args.output.with_name("pilot_resource_proxy_report.md")
    lines = [
        "# AERPAW LW1 Pilot Resource-Proxy Freeze",
        "",
        f"- Pilot scenes: {result['pilot_scene_count']} (permanently excluded from final)",
        f"- Band: {low:g}–{high:g} MHz, {n_channels} equal-width channels",
        f"- Threshold: per-sweep median + {threshold_rule['sigma_multiplier']:g} × 1.4826 × MAD",
        f"- Median threshold: {result['threshold_dbm_summary']['median']:.3f} dBm",
        f"- Median pooled occupancy: {result['channel_occupancy_summary']['pooled']['median']:.6f}",
        "- Learned-model outputs used: no",
        "- Final access consumed: no",
        "",
        "This artifact freezes an input/resource proxy only. It is not a detection-accuracy or final-result report.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(args.output), "report": str(report_path), "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
