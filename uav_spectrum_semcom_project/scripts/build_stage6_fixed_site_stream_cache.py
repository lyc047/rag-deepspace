#!/usr/bin/env python
"""Build a compact Stage-6 cache by streaming the 2022 fixed-site ZIPs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    discover_aerpaw_zip_pairs,
)
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    load_aerpaw_zip_power_sweep,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "configs"
    / "stage6_matched_reliability_development_v1.json"
)


def _channel_slices(
    frequency_mhz: np.ndarray,
    n_channels: int,
) -> tuple[slice, ...]:
    edges = np.linspace(
        float(frequency_mhz[0]),
        float(frequency_mhz[-1]),
        int(n_channels) + 1,
    )
    indices = np.searchsorted(frequency_mhz, edges, side="left")
    indices[0] = 0
    indices[-1] = frequency_mhz.size
    output = tuple(
        slice(int(indices[index]), int(indices[index + 1]))
        for index in range(int(n_channels))
    )
    if any(value.start == value.stop for value in output):
        raise ValueError("frequency partition produced an empty channel")
    return output


def _aggregate(
    values: np.ndarray,
    slices: tuple[slice, ...],
) -> np.ndarray:
    return np.asarray(
        [
            np.mean(values[value_slice], dtype=np.float64)
            for value_slice in slices
        ],
        dtype=np.float32,
    )


def _site_timing(pairs) -> tuple[float, float]:
    times = [
        datetime.fromisoformat(value.timestamp_local) for value in pairs
    ]
    gaps = np.asarray(
        [
            (right - left).total_seconds()
            for left, right in zip(times, times[1:])
            if right.date() == left.date()
        ],
        dtype=np.float64,
    )
    if gaps.size < 1:
        raise ValueError("site has no within-date timing gaps")
    median = float(np.median(gaps))
    return median, max(3.0 * median, 120.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_DIR
            / "results/stage6/fixed_site_stream_cache_v1/cache.npz"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=(
            PROJECT_DIR
            / "results/stage6/fixed_site_stream_cache_v1/result.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    result_path = args.result.resolve()
    if output.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite fixed-site cache")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["development_sources"][
            "external_final_archives_may_be_opened"
        ]
        or protocol["development_sources"][
            "external_final_signal_values_may_be_loaded"
        ]
    ):
        raise ValueError("external Final access must remain forbidden")
    archives = protocol["development_sources"][
        "grouped_validation_archives"
    ]
    n_values = tuple(
        int(value) for value in protocol["task_grid"]["n_channels"]
    )
    band_low, band_high = 3550.0, 3700.0
    arrays: dict[str, list] = {
        "scene_ids": [],
        "site_ids": [],
        "timestamps_local": [],
        "outer_group_ids": [],
        "session_group_ids": [],
    }
    for n_channels in n_values:
        arrays[f"channel_power_n{n_channels}"] = []
    archive_audit = {}
    power_min = float("inf")
    power_max = float("-inf")
    total_started = time.perf_counter()

    for entry in archives:
        site = str(entry["site"])
        archive_path = Path(entry["path"]).resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        pairs, errors = discover_aerpaw_zip_pairs(archive_path)
        if errors:
            raise ValueError(
                f"{site} ZIP discovery errors: {'; '.join(errors[:10])}"
            )
        if len(pairs) != int(entry["paired_scene_count"]):
            raise ValueError(f"{site} pair count changed")
        median_gap, gap_threshold = _site_timing(pairs)
        site_started = time.perf_counter()
        with zipfile.ZipFile(archive_path) as handle:
            reference = load_aerpaw_zip_power_sweep(pairs[0], handle)
            if reference.site != site:
                raise ValueError(f"{site} reference metadata mismatch")
            frequency = np.asarray(reference.frequencies_mhz, dtype=np.float64)
            band_mask = (frequency >= band_low) & (frequency < band_high)
            cropped_frequency = frequency[band_mask]
            if cropped_frequency.size != 2500:
                raise ValueError(
                    f"{site} analysis band bin count changed: "
                    f"{cropped_frequency.size}"
                )
            slices_by_n = {
                n_channels: _channel_slices(
                    cropped_frequency, n_channels
                )
                for n_channels in n_values
            }
            previous_time: datetime | None = None
            previous_outer: str | None = None
            segment = -1
            for index, pair in enumerate(pairs):
                info = handle.getinfo(pair.data_member)
                if info.file_size != frequency.size * 4:
                    raise ValueError(
                        f"{site}/{pair.stem} data size changed"
                    )
                raw = handle.read(pair.data_member)
                values = np.frombuffer(raw, dtype="<f4")
                if (
                    values.size != frequency.size
                    or not np.all(np.isfinite(values))
                    or np.any(values < -200.0)
                    or np.any(values > 20.0)
                ):
                    raise ValueError(
                        f"{site}/{pair.stem} invalid power values"
                    )
                cropped = values[band_mask]
                current = datetime.fromisoformat(pair.timestamp_local)
                outer = f"{site}:{current.strftime('%Y%m%d')}"
                if (
                    previous_time is None
                    or outer != previous_outer
                    or (current - previous_time).total_seconds()
                    > gap_threshold
                ):
                    segment += 1
                arrays["scene_ids"].append(
                    f"aerpaw-fixed-2022:{site}:{pair.stem}"
                )
                arrays["site_ids"].append(site)
                arrays["timestamps_local"].append(pair.timestamp_local)
                arrays["outer_group_ids"].append(outer)
                arrays["session_group_ids"].append(
                    f"{outer}:segment{segment:04d}"
                )
                for n_channels in n_values:
                    arrays[f"channel_power_n{n_channels}"].append(
                        _aggregate(cropped, slices_by_n[n_channels])
                    )
                power_min = min(power_min, float(np.min(cropped)))
                power_max = max(power_max, float(np.max(cropped)))
                previous_time = current
                previous_outer = outer
                if (index + 1) % 1000 == 0:
                    print(
                        f"{site}: {index + 1}/{len(pairs)}",
                        flush=True,
                    )
        archive_audit[site] = {
            "path": str(archive_path),
            "size_bytes": int(archive_path.stat().st_size),
            "pair_count": len(pairs),
            "median_gap_seconds": median_gap,
            "session_gap_threshold_seconds": gap_threshold,
            "elapsed_seconds": float(time.perf_counter() - site_started),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            key: (
                np.asarray(value)
                if key
                in {
                    "scene_ids",
                    "site_ids",
                    "timestamps_local",
                    "outer_group_ids",
                    "session_group_ids",
                }
                else np.asarray(value, dtype=np.float32)
            )
            for key, value in arrays.items()
        },
    )
    scene_count = len(arrays["scene_ids"])
    result = {
        "version": "1.0",
        "status": "fixed_site_stream_cache_complete",
        "protocol_sha256": sha256_file(protocol_path),
        "cache": str(output),
        "cache_sha256": sha256_file(output),
        "scene_count": scene_count,
        "site_counts": {
            site: arrays["site_ids"].count(site)
            for site in sorted(set(arrays["site_ids"]))
        },
        "outer_group_count": len(set(arrays["outer_group_ids"])),
        "session_segment_count": len(
            set(arrays["session_group_ids"])
        ),
        "analysis_band_mhz": [band_low, band_high],
        "analysis_band_bins": 2500,
        "power_range_dbm": [power_min, power_max],
        "channel_shapes": {
            str(n_channels): [
                scene_count,
                n_channels,
            ]
            for n_channels in n_values
        },
        "archive_audit": archive_audit,
        "elapsed_seconds": float(time.perf_counter() - total_started),
        "governance": {
            "role": "development_only_historically_accessed_2022",
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
            "full_archives_extracted": False,
            "streamed_data_members_only": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "Compact development cache from the historically accessed 2022 "
            "fixed-site activity. It supports grouped validation but is not "
            "an independent external Final."
        ),
    }
    atomic_write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
