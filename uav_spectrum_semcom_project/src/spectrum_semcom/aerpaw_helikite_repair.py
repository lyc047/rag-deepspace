"""Compatibility adapter used only by the Stage-6 repair validation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from .aerpaw_helikite import load_helikite_zip_power_sweep
from .aerpaw_spectrum import AerpawPowerSweep


def _compact_frequency_axis_mhz(
    global_meta: Mapping[str, Any], sample_count: int
) -> np.ndarray:
    required = ("aerpaw:freq_start", "aerpaw:freq_stop", "aerpaw:bin_width")
    if any(name not in global_meta for name in required):
        raise ValueError(
            "unannotated power layout requires AERPAW frequency-axis metadata"
        )
    start_hz, stop_hz, bin_width_hz = (
        float(global_meta[name]) for name in required
    )
    if (
        not all(np.isfinite(value) for value in (start_hz, stop_hz, bin_width_hz))
        or start_hz >= stop_hz
        or bin_width_hz <= 0.0
        or sample_count < 2
    ):
        raise ValueError("invalid AERPAW frequency-axis metadata")
    expected_count = int(round((stop_hz - start_hz) / bin_width_hz)) + 1
    endpoint_error_hz = abs(
        stop_hz - (start_hz + (sample_count - 1) * bin_width_hz)
    )
    if (
        sample_count != expected_count
        or endpoint_error_hz > 0.5 * bin_width_hz
    ):
        raise ValueError(
            "helikite compact power vector does not match its frequency axis"
        )
    return (
        np.linspace(start_hz, stop_hz, sample_count, dtype=np.float64) / 1.0e6
    )


def load_helikite_zip_power_sweep_compatible(
    archive: zipfile.ZipFile,
    *,
    meta_member: str,
    data_member: str,
    timestamp_local: str,
    site: str,
    local_timezone: str = "America/New_York",
    verify_sha512: bool = True,
) -> AerpawPowerSweep:
    """Decode both annotated 2023/2024 and compact 2025 power sweeps."""

    metadata = json.loads(archive.read(meta_member).decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("SigMF metadata root must be an object")
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("SigMF annotations must be a list")
    if annotations:
        return load_helikite_zip_power_sweep(
            archive,
            meta_member=meta_member,
            data_member=data_member,
            timestamp_local=timestamp_local,
            site=site,
            local_timezone=local_timezone,
            verify_sha512=verify_sha512,
        )

    global_meta = metadata.get("global")
    if not isinstance(global_meta, Mapping):
        raise ValueError("SigMF global metadata is required")
    if str(global_meta.get("core:datatype", "")).lower() != "rf32_le":
        raise ValueError("compact helikite power layout must use rf32_le")
    payload = archive.read(data_member)
    if len(payload) % np.dtype("<f4").itemsize:
        raise ValueError("helikite ZIP power size is not a float32 vector")
    expected_sha512 = str(global_meta.get("core:sha512", "")).lower()
    if verify_sha512 and (
        len(expected_sha512) != 128
        or hashlib.sha512(payload).hexdigest() != expected_sha512
    ):
        raise ValueError("helikite ZIP power SHA-512 mismatch")
    powers = np.frombuffer(payload, dtype="<f4").copy()
    frequencies_mhz = _compact_frequency_axis_mhz(global_meta, powers.size)
    local = datetime.fromisoformat(timestamp_local).replace(
        tzinfo=ZoneInfo(local_timezone)
    )
    return AerpawPowerSweep(
        frequencies_mhz,
        powers,
        site,
        local.astimezone(ZoneInfo("UTC")).isoformat(),
        Path(f"{archive.filename}!{meta_member}"),
        Path(f"{archive.filename}!{data_member}"),
        metadata,
    )


__all__ = ["load_helikite_zip_power_sweep_compatible"]
