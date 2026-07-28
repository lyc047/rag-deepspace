#!/usr/bin/env python
"""Validate the excluded 2023 helikite pilot and build N-channel caches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_helikite import (  # noqa: E402
    crop_helikite_sweep,
    discover_helikite_power_pairs,
    infer_integer_hour_clock_correction,
    load_helikite_position_log,
    load_helikite_power_sweep,
    nearest_helikite_position,
)
from spectrum_semcom.aerpaw_spectrum import channel_power_dbm  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


def cluster_id(dataset_id: str, timestamp_local: str) -> str:
    value = datetime.fromisoformat(timestamp_local)
    minute = (value.minute // 15) * 15
    return f"{dataset_id}:{value:%Y%m%dT%H}{minute:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_helikite_pilot_protocol_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/helikite_pilot_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "helikite_pilot_result.json"
    cache_path = args.out_dir / "helikite_pilot_cache.npz"
    if output.exists() or cache_path.exists():
        raise FileExistsError("refusing to overwrite helikite pilot v1")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = protocol["source"]
    governance = protocol["governance"]
    quality = protocol["quality_gates"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_sigmf_metadata_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or protocol["resource_task"]["parameters_may_be_retuned_from_pilot"]
    ):
        raise ValueError("helikite pilot governance is invalid")
    archive = Path(source["archive"])
    if sha256_file(archive) != source["archive_sha256"]:
        raise ValueError("pilot archive SHA-256 mismatch")
    root = Path(source["extracted_root"])
    pairs, discovery_errors = discover_helikite_power_pairs(root / "pow_spec")
    gps_meta = next(
        path
        for path in sorted((root / "GPS_logs").glob("*.sigmf-meta"))
        if "vehicleOut" in path.name
    )
    gps = load_helikite_position_log(
        gps_meta,
        verify_sha512=bool(
            protocol["quality_gates"]["verify_sigmf_sha512"]
        ),
    )
    reference_unix = np.asarray(
        [
            datetime.fromisoformat(pair.timestamp_local)
            .replace(tzinfo=ZoneInfo(source["local_timezone"]))
            .timestamp()
            for pair in pairs
        ],
        dtype=np.float64,
    )
    correction_range = quality["gps_clock_correction_search_hours"]
    clock_correction = infer_integer_hour_clock_correction(
        gps,
        reference_unix,
        minimum_hours=int(correction_range[0]),
        maximum_hours=int(correction_range[1]),
    )
    expected_count = int(source["expected_power_sweeps"])
    band_low, band_high = map(
        float, protocol["resource_task"]["analysis_band_mhz"]
    )
    n_values = tuple(int(value) for value in protocol["resource_task"]["n_channels"])
    arrays: dict[str, list] = {
        "scene_ids": [],
        "timestamps_local": [],
        "timestamps_utc": [],
        "cluster_ids": [],
        "longitude_deg": [],
        "latitude_deg": [],
        "altitude_m": [],
        "gps_offset_s": [],
    }
    for n_channels in n_values:
        arrays[f"channel_power_n{n_channels}"] = []
    reference_frequency = None
    power_min = float("inf")
    power_max = float("-inf")
    band_bin_counts = []
    gps_offsets = []
    timestamps = []
    validation_errors = list(discovery_errors)
    for index, pair in enumerate(pairs):
        try:
            sweep = load_helikite_power_sweep(
                pair,
                site=source["site_id"],
                local_timezone=source["local_timezone"],
                verify_sha512=bool(quality["verify_sigmf_sha512"]),
            )
            cropped = crop_helikite_sweep(
                sweep, low_mhz=band_low, high_mhz=band_high
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            validation_errors.append(f"{pair.stem}: {exc}")
            continue
        if reference_frequency is None:
            reference_frequency = cropped.frequencies_mhz
        elif (
            reference_frequency.shape != cropped.frequencies_mhz.shape
            or not np.allclose(
                reference_frequency,
                cropped.frequencies_mhz,
                rtol=0.0,
                atol=float(quality["maximum_frequency_grid_error_mhz"]),
            )
        ):
            validation_errors.append(f"{pair.stem}: frequency grid mismatch")
        power_min = min(power_min, float(np.min(cropped.powers_dbm)))
        power_max = max(power_max, float(np.max(cropped.powers_dbm)))
        band_bin_counts.append(cropped.n_bins)
        local = datetime.fromisoformat(pair.timestamp_local).replace(
            tzinfo=ZoneInfo(source["local_timezone"])
        )
        unix = local.timestamp()
        correction_s = float(clock_correction["correction_s"])
        position = nearest_helikite_position(gps, unix - correction_s)
        corrected_offset = (
            position["position_timestamp_unix_s"] + correction_s - unix
        )
        gps_offsets.append(abs(corrected_offset))
        timestamps.append(local)
        arrays["scene_ids"].append(
            f"{source['dataset_id']}:{pair.stem}"
        )
        arrays["timestamps_local"].append(pair.timestamp_local)
        arrays["timestamps_utc"].append(
            local.astimezone(ZoneInfo("UTC")).isoformat()
        )
        arrays["cluster_ids"].append(
            cluster_id(source["dataset_id"], pair.timestamp_local)
        )
        for key in (
            "longitude_deg",
            "latitude_deg",
            "altitude_m",
        ):
            arrays[key].append(position[key])
        arrays["gps_offset_s"].append(corrected_offset)
        for n_channels in n_values:
            arrays[f"channel_power_n{n_channels}"].append(
                channel_power_dbm(cropped, n_channels)
            )
        if (index + 1) % 50 == 0:
            print(f"pilot sweeps: {index + 1}/{len(pairs)}", flush=True)
    gaps = np.asarray(
        [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
        ],
        dtype=np.float64,
    )
    gates = {
        "expected_sweep_count": len(pairs) == expected_count,
        "all_pairs_loaded": len(arrays["scene_ids"]) == len(pairs),
        "no_discovery_or_validation_errors": not validation_errors,
        "analysis_band_bin_count": bool(band_bin_counts)
        and min(band_bin_counts)
        >= int(quality["minimum_bins_in_analysis_band"]),
        "frequency_grid_consistent": reference_frequency is not None
        and not any("frequency grid mismatch" in row for row in validation_errors),
        "power_range_plausible": power_min
        >= float(quality["power_range_dbm"][0])
        and power_max <= float(quality["power_range_dbm"][1]),
        "gps_alignment": bool(gps_offsets)
        and max(gps_offsets)
        <= float(quality["maximum_nearest_gps_offset_s"]),
        "sweep_spacing": gaps.size > 0
        and float(np.min(gaps)) >= float(quality["minimum_sweep_gap_s"])
        and float(np.max(gaps)) <= float(quality["maximum_sweep_gap_s"]),
    }
    if not all(gates.values()):
        raise ValueError(f"helikite pilot quality gate failed: {gates}")
    args.out_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        **{
            key: np.asarray(value)
            for key, value in arrays.items()
        },
    )
    result = {
        "version": "1.0",
        "status": "stage5_helikite_pilot_adapter_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "pilot_archive_sha256": source["archive_sha256"],
        "cache": str(cache_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "cache_sha256": sha256_file(cache_path),
        "quality_gates": gates,
        "quality_summary": {
            "power_sweep_count": len(pairs),
            "analysis_band_mhz": [band_low, band_high],
            "analysis_band_bins": int(min(band_bin_counts)),
            "frequency_resolution_mhz": float(
                np.median(np.diff(reference_frequency))
            ),
            "observed_power_range_dbm": [power_min, power_max],
            "sweep_gap_range_s": [
                float(np.min(gaps)),
                float(np.max(gaps)),
            ],
            "maximum_absolute_gps_offset_s": float(max(gps_offsets)),
            "gps_clock_correction": clock_correction,
            "altitude_range_m": [
                float(np.min(arrays["altitude_m"])),
                float(np.max(arrays["altitude_m"])),
            ],
            "cluster_count_15_minute": len(set(arrays["cluster_ids"])),
            "channel_cache_shapes": {
                str(n): [len(pairs), n] for n in n_values
            },
        },
        "validation_errors": validation_errors,
        "governance": {
            "pilot_permanently_excluded_from_final": True,
            "external_final_archives_opened": False,
            "external_final_sigmf_metadata_opened": False,
            "external_final_signal_values_loaded": False,
            "algorithm_or_success_threshold_retuned": False,
            "final_access_consumed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "cache": str(cache_path),
                "sweeps": len(pairs),
                "all_quality_gates_pass": all(gates.values()),
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
