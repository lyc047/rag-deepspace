"""Strict adapter for AERPAW helikite frequency/power SigMF sweeps."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from .aerpaw_spectrum import AerpawPowerSweep


_POWER_NAME = re.compile(r"^spec_results_(\d{8})_(\d{6})\.sigmf-meta$")


@dataclass(frozen=True)
class HelikitePowerPair:
    stem: str
    meta_path: Path
    data_path: Path
    timestamp_local: str


@dataclass(frozen=True)
class HelikitePositionLog:
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    altitude_m: np.ndarray
    timestamp_unix_s: np.ndarray
    meta_path: Path
    data_path: Path

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.longitude_deg,
                self.latitude_deg,
                self.altitude_m,
                self.timestamp_unix_s,
            )
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("position arrays must be one-dimensional")
        if len({value.size for value in arrays}) != 1 or arrays[0].size < 1:
            raise ValueError("position arrays must have equal non-zero length")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("position arrays must be finite")
        if np.any(np.diff(arrays[3]) < 0):
            raise ValueError("position timestamps must be non-decreasing")
        for name, value in zip(
            (
                "longitude_deg",
                "latitude_deg",
                "altitude_m",
                "timestamp_unix_s",
            ),
            arrays,
        ):
            object.__setattr__(self, name, value)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SigMF metadata root must be an object")
    return value


def _data_path(meta_path: Path) -> Path:
    if not meta_path.name.endswith(".sigmf-meta"):
        raise ValueError("metadata filename must end with .sigmf-meta")
    return meta_path.with_name(
        meta_path.name.removesuffix(".sigmf-meta") + ".sigmf-data"
    )


def _sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _global_metadata(metadata: Mapping[str, Any], datatype: str) -> Mapping[str, Any]:
    value = metadata.get("global")
    if not isinstance(value, Mapping):
        raise ValueError("SigMF global metadata is required")
    observed = str(value.get("core:datatype", "")).lower()
    if observed != datatype:
        raise ValueError(f"expected {datatype}, found {observed!r}")
    return value


def _annotation_map(
    metadata: Mapping[str, Any], required: tuple[str, ...]
) -> dict[str, tuple[int, int]]:
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("SigMF annotations must be a list")
    output: dict[str, tuple[int, int]] = {}
    for row in annotations:
        if not isinstance(row, Mapping):
            raise ValueError("SigMF annotation must be an object")
        name = str(row.get("core:comment", "")).strip().lower()
        if name in required:
            if name in output:
                raise ValueError(f"duplicate annotation: {name}")
            start = int(row.get("core:sample_start", -1))
            count = int(row.get("core:sample_count", -1))
            if start < 0 or count < 1:
                raise ValueError(f"invalid annotation extent: {name}")
            output[name] = (start, count)
    if set(output) != set(required):
        raise ValueError(
            f"required annotations differ: {sorted(output)} != {sorted(required)}"
        )
    return output


def discover_helikite_power_pairs(
    root: str | Path,
) -> tuple[list[HelikitePowerPair], list[str]]:
    """Discover only frequency/power sweep pairs; GPS files are excluded."""

    root = Path(root)
    pairs: list[HelikitePowerPair] = []
    errors: list[str] = []
    for meta in sorted(root.rglob("spec_results_*.sigmf-meta")):
        match = _POWER_NAME.match(meta.name)
        if match is None:
            errors.append(f"unexpected power metadata filename: {meta}")
            continue
        data = _data_path(meta)
        if not data.is_file():
            errors.append(f"missing power data pair: {meta}")
            continue
        timestamp = datetime.strptime(
            "".join(match.groups()), "%Y%m%d%H%M%S"
        ).isoformat()
        pairs.append(
            HelikitePowerPair(
                meta.name.removesuffix(".sigmf-meta"),
                meta.resolve(),
                data.resolve(),
                timestamp,
            )
        )
    return pairs, errors


def load_helikite_power_sweep(
    pair_or_meta: HelikitePowerPair | str | Path,
    *,
    site: str,
    local_timezone: str = "America/New_York",
    verify_sha512: bool = True,
) -> AerpawPowerSweep:
    """Load one rf32 vector containing concatenated MHz and dBm arrays."""

    if isinstance(pair_or_meta, HelikitePowerPair):
        pair = pair_or_meta
    else:
        meta_path = Path(pair_or_meta)
        match = _POWER_NAME.match(meta_path.name)
        if match is None:
            raise ValueError("invalid helikite power filename")
        pair = HelikitePowerPair(
            meta_path.name.removesuffix(".sigmf-meta"),
            meta_path,
            _data_path(meta_path),
            datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").isoformat(),
        )
    metadata = _load_object(pair.meta_path)
    global_meta = _global_metadata(metadata, "rf32_le")
    annotations = _annotation_map(metadata, ("freqs", "powers"))
    f_start, f_count = annotations["freqs"]
    p_start, p_count = annotations["powers"]
    if f_count != p_count or f_start != 0 or p_start != f_count:
        raise ValueError("frequency and power annotations must be equal contiguous arrays")
    expected_bytes = (p_start + p_count) * np.dtype("<f4").itemsize
    if pair.data_path.stat().st_size != expected_bytes:
        raise ValueError("helikite power file size does not match annotations")
    expected_sha512 = str(global_meta.get("core:sha512", "")).lower()
    if verify_sha512 and (
        len(expected_sha512) != 128
        or _sha512_file(pair.data_path) != expected_sha512
    ):
        raise ValueError("helikite power file SHA-512 mismatch")
    values = np.fromfile(pair.data_path, dtype="<f4")
    frequencies = values[f_start : f_start + f_count].astype(np.float64)
    powers = values[p_start : p_start + p_count]
    local = datetime.fromisoformat(pair.timestamp_local).replace(
        tzinfo=ZoneInfo(local_timezone)
    )
    return AerpawPowerSweep(
        frequencies_mhz=frequencies,
        powers_dbm=powers,
        site=site,
        capture_datetime=local.astimezone(ZoneInfo("UTC")).isoformat(),
        meta_path=pair.meta_path,
        data_path=pair.data_path,
        metadata=metadata,
    )


def load_helikite_zip_power_sweep(
    archive: zipfile.ZipFile,
    *,
    meta_member: str,
    data_member: str,
    timestamp_local: str,
    site: str,
    local_timezone: str = "America/New_York",
    verify_sha512: bool = True,
) -> AerpawPowerSweep:
    """Load one paired sweep from a ZIP after Final access is consumed."""

    metadata_value = json.loads(archive.read(meta_member).decode("utf-8"))
    if not isinstance(metadata_value, dict):
        raise ValueError("SigMF metadata root must be an object")
    global_meta = _global_metadata(metadata_value, "rf32_le")
    annotations = _annotation_map(metadata_value, ("freqs", "powers"))
    f_start, f_count = annotations["freqs"]
    p_start, p_count = annotations["powers"]
    if f_count != p_count or f_start != 0 or p_start != f_count:
        raise ValueError("frequency and power annotations must be equal contiguous arrays")
    payload = archive.read(data_member)
    expected_bytes = (p_start + p_count) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError("helikite ZIP power size does not match annotations")
    expected_sha512 = str(global_meta.get("core:sha512", "")).lower()
    if verify_sha512 and (
        len(expected_sha512) != 128
        or hashlib.sha512(payload).hexdigest() != expected_sha512
    ):
        raise ValueError("helikite ZIP power SHA-512 mismatch")
    values = np.frombuffer(payload, dtype="<f4")
    frequencies = values[f_start : f_start + f_count].astype(np.float64)
    powers = values[p_start : p_start + p_count].copy()
    local = datetime.fromisoformat(timestamp_local).replace(
        tzinfo=ZoneInfo(local_timezone)
    )
    pseudo_meta = Path(f"{archive.filename}!{meta_member}")
    pseudo_data = Path(f"{archive.filename}!{data_member}")
    return AerpawPowerSweep(
        frequencies,
        powers,
        site,
        local.astimezone(ZoneInfo("UTC")).isoformat(),
        pseudo_meta,
        pseudo_data,
        metadata_value,
    )


def load_helikite_position_log(
    meta_path: str | Path, *, verify_sha512: bool = True
) -> HelikitePositionLog:
    meta_path = Path(meta_path)
    data_path = _data_path(meta_path)
    metadata = _load_object(meta_path)
    global_meta = _global_metadata(metadata, "rf64_le")
    annotations = _annotation_map(
        metadata, ("longitude", "latitude", "altitude", "timestamp")
    )
    count = {value[1] for value in annotations.values()}
    if len(count) != 1:
        raise ValueError("position annotations must have equal counts")
    extents = sorted(
        (start, start + length, name)
        for name, (start, length) in annotations.items()
    )
    if extents[0][0] != 0 or any(
        left[1] != right[0] for left, right in zip(extents, extents[1:])
    ):
        raise ValueError("position annotations must form one contiguous vector")
    expected_bytes = extents[-1][1] * np.dtype("<f8").itemsize
    if data_path.stat().st_size != expected_bytes:
        raise ValueError("position data size does not match annotations")
    expected_sha512 = str(global_meta.get("core:sha512", "")).lower()
    if verify_sha512 and (
        len(expected_sha512) != 128 or _sha512_file(data_path) != expected_sha512
    ):
        raise ValueError("position file SHA-512 mismatch")
    values = np.fromfile(data_path, dtype="<f8")

    def take(name: str) -> np.ndarray:
        start, length = annotations[name]
        return values[start : start + length]

    return HelikitePositionLog(
        take("longitude"),
        take("latitude"),
        take("altitude"),
        take("timestamp"),
        meta_path.resolve(),
        data_path.resolve(),
    )


def nearest_helikite_position(
    log: HelikitePositionLog, timestamp_unix_s: float
) -> dict[str, float]:
    if not np.isfinite(timestamp_unix_s):
        raise ValueError("query timestamp must be finite")
    index = int(np.searchsorted(log.timestamp_unix_s, timestamp_unix_s))
    candidates = [
        value
        for value in (index - 1, index)
        if 0 <= value < log.timestamp_unix_s.size
    ]
    if not candidates:
        raise ValueError("position log is empty")
    chosen = min(
        candidates,
        key=lambda value: (
            abs(float(log.timestamp_unix_s[value]) - timestamp_unix_s),
            value,
        ),
    )
    return {
        "longitude_deg": float(log.longitude_deg[chosen]),
        "latitude_deg": float(log.latitude_deg[chosen]),
        "altitude_m": float(log.altitude_m[chosen]),
        "position_timestamp_unix_s": float(log.timestamp_unix_s[chosen]),
        "offset_s": float(log.timestamp_unix_s[chosen] - timestamp_unix_s),
    }


def infer_integer_hour_clock_correction(
    log: HelikitePositionLog,
    reference_timestamps_unix_s: np.ndarray,
    *,
    minimum_hours: int = -14,
    maximum_hours: int = 14,
) -> dict[str, float]:
    """Infer a clock-zone correction from timestamps only, never signal values.

    The correction is added to raw GPS timestamps. Candidate corrections are
    restricted to exact integer hours so this cannot become an unconstrained
    alignment fit.
    """

    reference = np.asarray(
        reference_timestamps_unix_s, dtype=np.float64
    ).reshape(-1)
    if (
        reference.size < 1
        or not np.all(np.isfinite(reference))
        or minimum_hours > maximum_hours
    ):
        raise ValueError("invalid clock-correction references or range")
    candidates = []
    for hours in range(int(minimum_hours), int(maximum_hours) + 1):
        correction = float(hours * 3600)
        residuals = []
        for timestamp in reference:
            position = nearest_helikite_position(
                log, float(timestamp - correction)
            )
            residuals.append(
                abs(
                    position["position_timestamp_unix_s"]
                    + correction
                    - float(timestamp)
                )
            )
        candidates.append(
            (
                float(np.median(residuals)),
                float(np.max(residuals)),
                abs(hours),
                hours,
                correction,
            )
        )
    median, maximum, _, hours, correction = min(candidates)
    return {
        "correction_hours": float(hours),
        "correction_s": correction,
        "median_absolute_residual_s": median,
        "maximum_absolute_residual_s": maximum,
    }


def crop_helikite_sweep(
    sweep: AerpawPowerSweep, *, low_mhz: float, high_mhz: float
) -> AerpawPowerSweep:
    if not np.isfinite(low_mhz) or not np.isfinite(high_mhz) or low_mhz >= high_mhz:
        raise ValueError("invalid analysis band")
    mask = (sweep.frequencies_mhz >= low_mhz) & (
        sweep.frequencies_mhz < high_mhz
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("analysis band contains fewer than two bins")
    return AerpawPowerSweep(
        sweep.frequencies_mhz[mask],
        sweep.powers_dbm[mask],
        sweep.site,
        sweep.capture_datetime,
        sweep.meta_path,
        sweep.data_path,
        sweep.metadata,
    )
