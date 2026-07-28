"""AERPAW fixed-node power-sweep ingestion and governance utilities.

The February 2022 fixed-node deposit uses SigMF metadata, but its data files
contain one real-valued float32 power vector per sweep rather than complex I/Q.
Keeping this adapter separate from :mod:`data_io` prevents an ``rf32_le``
power trace from being silently interpreted as interleaved I/Q samples.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np


_NAME_PATTERN = re.compile(r"^results_(\d{8})_(\d{6})\.sigmf-meta$")


@dataclass(frozen=True)
class AerpawPowerSweep:
    """One fixed-node spectrum sweep in MHz and dBm."""

    frequencies_mhz: np.ndarray
    powers_dbm: np.ndarray
    site: str
    capture_datetime: str
    meta_path: Path
    data_path: Path
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_mhz, dtype=np.float64)
        powers = np.asarray(self.powers_dbm, dtype=np.float32)
        if frequencies.ndim != 1 or powers.ndim != 1 or frequencies.size < 2:
            raise ValueError("frequency and power arrays must be non-trivial one-dimensional vectors")
        if frequencies.shape != powers.shape:
            raise ValueError("frequency and power arrays must have equal length")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(powers)):
            raise ValueError("frequency and power arrays must be finite")
        if np.any(np.diff(frequencies) <= 0):
            raise ValueError("frequency axis must be strictly increasing")
        if not str(self.site).strip() or not str(self.capture_datetime).strip():
            raise ValueError("site and capture_datetime are required")
        object.__setattr__(self, "frequencies_mhz", frequencies)
        object.__setattr__(self, "powers_dbm", powers)

    @property
    def n_bins(self) -> int:
        return int(self.powers_dbm.size)


@dataclass(frozen=True)
class AerpawPair:
    stem: str
    meta_path: Path
    data_path: Path
    timestamp_local: str


@dataclass(frozen=True)
class AerpawZipPair:
    """A metadata-only pair reference inside an unextracted site archive."""

    stem: str
    archive_path: Path
    meta_member: str
    data_member: str
    timestamp_local: str


@dataclass(frozen=True)
class AerpawAlignedScene:
    """One deterministic, one-to-one timestamp match across fixed sites."""

    scene_id: str
    anchor_datetime_utc: str
    pairs_by_site: Mapping[str, AerpawPair | AerpawZipPair]
    offsets_from_anchor_s: Mapping[str, float]

    def __post_init__(self) -> None:
        sites = tuple(self.pairs_by_site)
        if len(sites) < 2 or len(sites) != len(set(sites)):
            raise ValueError("an aligned scene requires at least two unique sites")
        if set(sites) != set(self.offsets_from_anchor_s):
            raise ValueError("pair and offset site sets must match")
        if not self.scene_id or not self.anchor_datetime_utc:
            raise ValueError("scene_id and anchor_datetime_utc are required")
        if any(not np.isfinite(value) for value in self.offsets_from_anchor_s.values()):
            raise ValueError("alignment offsets must be finite")


def _data_path(meta_path: Path) -> Path:
    if not meta_path.name.endswith(".sigmf-meta"):
        raise ValueError("AERPAW metadata path must end with .sigmf-meta")
    return meta_path.with_name(meta_path.name.removesuffix(".sigmf-meta") + ".sigmf-data")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SigMF metadata root must be an object")
    return value


def load_aerpaw_power_sweep(meta_path: str | Path, data_path: str | Path | None = None) -> AerpawPowerSweep:
    """Load and strictly validate one ``rf32_le`` AERPAW power sweep."""

    meta_path = Path(meta_path)
    data_path = Path(data_path) if data_path is not None else _data_path(meta_path)
    metadata = _read_json(meta_path)
    global_meta = metadata.get("global")
    if not isinstance(global_meta, Mapping):
        raise ValueError("SigMF global metadata is required")
    datatype = str(global_meta.get("core:datatype", "")).lower()
    if datatype != "rf32_le":
        raise ValueError(f"AERPAW power adapter requires rf32_le, found {datatype!r}")

    frequencies = np.asarray(global_meta.get("dataset:frequency_axis_MHz", []), dtype=np.float64)
    declared_bins = int(global_meta.get("dataset:num_bins", frequencies.size))
    if frequencies.ndim != 1 or frequencies.size != declared_bins or declared_bins < 2:
        raise ValueError("dataset frequency axis and declared bin count are inconsistent")
    expected_bytes = declared_bins * np.dtype("<f4").itemsize
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    actual_bytes = data_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"power file size mismatch: {actual_bytes} != {expected_bytes}")
    powers = np.fromfile(data_path, dtype="<f4")

    captures = metadata.get("captures", [])
    capture = captures[0] if isinstance(captures, list) and captures else {}
    if not isinstance(capture, Mapping):
        capture = {}
    capture_datetime = str(capture.get("core:datetime", ""))
    site = str(global_meta.get("dataset:site", ""))
    span = global_meta.get("dataset:frequency_span_MHz")
    if isinstance(span, list) and len(span) == 2 and not np.allclose(
        np.asarray(span, dtype=np.float64), frequencies[[0, -1]], rtol=0.0, atol=1e-5
    ):
        raise ValueError("declared frequency span does not match frequency axis endpoints")
    return AerpawPowerSweep(
        frequencies_mhz=frequencies,
        powers_dbm=powers,
        site=site,
        capture_datetime=capture_datetime,
        meta_path=meta_path,
        data_path=data_path,
        metadata=metadata,
    )


def discover_aerpaw_pairs(root: str | Path) -> tuple[list[AerpawPair], list[str]]:
    """Discover valid filename pairs without reading measurement values."""

    root = Path(root)
    errors: list[str] = []
    pairs: list[AerpawPair] = []
    metas = sorted(path for path in root.rglob("*.sigmf-meta") if "__MACOSX" not in path.parts and not path.name.startswith("._"))
    data_paths = {path.resolve() for path in root.rglob("*.sigmf-data") if "__MACOSX" not in path.parts and not path.name.startswith("._")}
    used_data: set[Path] = set()
    for meta in metas:
        match = _NAME_PATTERN.match(meta.name)
        if match is None:
            errors.append(f"unexpected metadata filename: {meta}")
            continue
        data = _data_path(meta).resolve()
        if data not in data_paths:
            errors.append(f"missing data pair: {meta}")
            continue
        used_data.add(data)
        timestamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=None).isoformat()
        pairs.append(AerpawPair(meta.stem.removesuffix(".sigmf"), meta.resolve(), data, timestamp))
    for orphan in sorted(data_paths - used_data):
        errors.append(f"orphan data file: {orphan}")
    return pairs, errors


def _safe_zip_member(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return normalized


def discover_aerpaw_zip_pairs(archive_path: str | Path) -> tuple[list[AerpawZipPair], list[str]]:
    """Discover sweep pairs from ZIP metadata without reading measurement bytes."""

    archive = Path(archive_path).resolve()
    errors: list[str] = []
    pairs: list[AerpawZipPair] = []
    with zipfile.ZipFile(archive) as handle:
        members: set[str] = set()
        for info in handle.infolist():
            try:
                normalized = _safe_zip_member(info.filename)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if info.is_dir() or "__MACOSX" in normalized.parts or normalized.name.startswith("._"):
                continue
            members.add(normalized.as_posix())
    metas = sorted(name for name in members if name.endswith(".sigmf-meta"))
    used_data: set[str] = set()
    for meta in metas:
        normalized = PurePosixPath(meta)
        match = _NAME_PATTERN.match(normalized.name)
        if match is None:
            errors.append(f"unexpected metadata filename: {meta}")
            continue
        data = normalized.with_name(normalized.name.removesuffix(".sigmf-meta") + ".sigmf-data").as_posix()
        if data not in members:
            errors.append(f"missing data pair: {meta}")
            continue
        used_data.add(data)
        timestamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").isoformat()
        pairs.append(
            AerpawZipPair(
                stem=normalized.name.removesuffix(".sigmf-meta"),
                archive_path=archive,
                meta_member=meta,
                data_member=data,
                timestamp_local=timestamp,
            )
        )
    for orphan in sorted(name for name in members if name.endswith(".sigmf-data") and name not in used_data):
        errors.append(f"orphan data file: {orphan}")
    return pairs, errors


def filter_aerpaw_pairs_by_local_window(
    pairs: Iterable[AerpawPair | AerpawZipPair],
    *,
    start_local_inclusive: str,
    end_local_exclusive: str,
) -> list[AerpawPair | AerpawZipPair]:
    """Select filename-timestamped pairs inside one predeclared local-time window."""

    start = datetime.fromisoformat(start_local_inclusive)
    end = datetime.fromisoformat(end_local_exclusive)
    if start.tzinfo is not None or end.tzinfo is not None or start >= end:
        raise ValueError("local window bounds must be naive ISO datetimes with start before end")
    selected = []
    for pair in pairs:
        current = datetime.fromisoformat(pair.timestamp_local)
        if current.tzinfo is not None:
            raise ValueError("pair timestamp_local must be a naive ISO datetime")
        if start <= current < end:
            selected.append(pair)
    return sorted(selected, key=lambda item: (item.timestamp_local, item.stem))


def thin_aerpaw_pairs_by_time(
    pairs: Iterable[AerpawPair | AerpawZipPair], *, min_separation_s: float
) -> list[AerpawPair | AerpawZipPair]:
    """Greedily retain the earliest single-site pair after each time interval."""

    if not np.isfinite(min_separation_s) or min_separation_s < 0:
        raise ValueError("min_separation_s must be finite and non-negative")
    selected: list[AerpawPair | AerpawZipPair] = []
    last: datetime | None = None
    for pair in sorted(pairs, key=lambda item: (item.timestamp_local, item.stem)):
        current = datetime.fromisoformat(pair.timestamp_local)
        if current.tzinfo is not None:
            raise ValueError("pair timestamp_local must be a naive ISO datetime")
        if last is None or (current - last).total_seconds() >= min_separation_s:
            selected.append(pair)
            last = current
    return selected


def extract_aerpaw_zip_pairs(
    pairs: Iterable[AerpawZipPair], *, destination: str | Path, site: str
) -> list[AerpawPair]:
    """Extract selected pairs without interpreting their measurement values."""

    destination = Path(destination).resolve() / str(site)
    destination.mkdir(parents=True, exist_ok=True)
    archives: dict[Path, zipfile.ZipFile] = {}
    output: list[AerpawPair] = []
    try:
        for pair in pairs:
            if not isinstance(pair, AerpawZipPair):
                raise TypeError("extract_aerpaw_zip_pairs requires ZIP pair references")
            handle = archives.setdefault(pair.archive_path, zipfile.ZipFile(pair.archive_path))
            paths: dict[str, Path] = {}
            for member_name in (pair.meta_member, pair.data_member):
                target = destination / _safe_zip_member(member_name).name
                info = handle.getinfo(member_name)
                if target.exists():
                    if target.stat().st_size != info.file_size:
                        raise ValueError(f"existing compact file has wrong size: {target}")
                else:
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    with handle.open(info) as source, temporary.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)
                    if temporary.stat().st_size != info.file_size:
                        temporary.unlink(missing_ok=True)
                        raise ValueError(f"extracted compact file has wrong size: {target}")
                    temporary.replace(target)
                paths[member_name] = target
            output.append(
                AerpawPair(
                    pair.stem,
                    paths[pair.meta_member],
                    paths[pair.data_member],
                    pair.timestamp_local,
                )
            )
    finally:
        for handle in archives.values():
            handle.close()
    return output


def pair_datetime_utc(pair: AerpawPair | AerpawZipPair, local_timezone: str = "America/New_York") -> datetime:
    """Interpret filename time in the campaign timezone and return aware UTC."""

    local = datetime.fromisoformat(pair.timestamp_local)
    if local.tzinfo is not None:
        raise ValueError("AERPAW filename timestamp must be timezone-naive local time")
    return local.replace(tzinfo=ZoneInfo(local_timezone)).astimezone(timezone.utc)


def align_aerpaw_pairs_by_timestamp(
    pairs_by_site: Mapping[str, Iterable[AerpawPair | AerpawZipPair]],
    *,
    expected_sites: Iterable[str],
    anchor_site: str,
    max_offset_s: float,
    local_timezone: str = "America/New_York",
) -> tuple[list[AerpawAlignedScene], dict[str, Any]]:
    """Match sites without reusing a sweep or interpolating measurement values.

    The closest unused sweep is chosen deterministically.  Ties are resolved by
    earlier UTC time and then filename stem.  An anchor is dropped when any site
    lacks a sweep inside the predeclared tolerance.
    """

    sites = tuple(expected_sites)
    if len(sites) < 2 or len(sites) != len(set(sites)):
        raise ValueError("expected_sites must contain at least two unique site IDs")
    if anchor_site not in sites:
        raise ValueError("anchor_site must be one of expected_sites")
    if not np.isfinite(max_offset_s) or max_offset_s < 0:
        raise ValueError("max_offset_s must be finite and non-negative")
    if set(pairs_by_site) != set(sites):
        raise ValueError("pairs_by_site keys must exactly match expected_sites")

    ordered: dict[str, list[tuple[datetime, AerpawPair | AerpawZipPair]]] = {}
    for site in sites:
        rows = sorted(
            ((pair_datetime_utc(pair, local_timezone), pair) for pair in pairs_by_site[site]),
            key=lambda row: (row[0], row[1].stem),
        )
        if not rows:
            raise ValueError(f"site {site} has no discoverable pairs")
        ordered[site] = rows

    used: dict[str, set[int]] = {site: set() for site in sites}
    timestamps = {site: [row[0] for row in rows] for site, rows in ordered.items()}
    scenes: list[AerpawAlignedScene] = []
    rejected = 0
    for anchor_index, (anchor_time, anchor_pair) in enumerate(ordered[anchor_site]):
        chosen: dict[str, tuple[int, datetime, AerpawPair]] = {
            anchor_site: (anchor_index, anchor_time, anchor_pair)
        }
        valid = True
        for site in sites:
            if site == anchor_site:
                continue
            insertion = bisect_left(timestamps[site], anchor_time)
            candidates = []
            left = insertion - 1
            while left >= 0:
                site_time, pair = ordered[site][left]
                delta = abs((site_time - anchor_time).total_seconds())
                if delta > max_offset_s:
                    break
                if left not in used[site]:
                    candidates.append((delta, site_time, pair.stem, left, pair))
                left -= 1
            right = insertion
            while right < len(ordered[site]):
                site_time, pair = ordered[site][right]
                delta = abs((site_time - anchor_time).total_seconds())
                if delta > max_offset_s:
                    break
                if right not in used[site]:
                    candidates.append((delta, site_time, pair.stem, right, pair))
                right += 1
            if not candidates:
                valid = False
                break
            delta, site_time, _, index, pair = min(candidates)
            if delta > max_offset_s:
                valid = False
                break
            chosen[site] = (index, site_time, pair)
        if not valid:
            rejected += 1
            continue
        for site, (index, _, _) in chosen.items():
            used[site].add(index)
        anchor_text = anchor_time.isoformat().replace("+00:00", "Z")
        compact = anchor_time.strftime("%Y%m%dT%H%M%SZ")
        scenes.append(
            AerpawAlignedScene(
                scene_id=f"aerpaw-three-site:{compact}",
                anchor_datetime_utc=anchor_text,
                pairs_by_site={site: chosen[site][2] for site in sites},
                offsets_from_anchor_s={
                    site: float((chosen[site][1] - anchor_time).total_seconds()) for site in sites
                },
            )
        )
    return scenes, {
        "anchor_site": anchor_site,
        "expected_sites": list(sites),
        "max_offset_s": float(max_offset_s),
        "anchor_count": len(ordered[anchor_site]),
        "aligned_scene_count": len(scenes),
        "rejected_anchor_count": rejected,
        "per_site_input_count": {site: len(ordered[site]) for site in sites},
        "per_site_unused_count": {site: len(ordered[site]) - len(used[site]) for site in sites},
        "measurement_interpolation_used": False,
        "pair_reuse_used": False,
    }


def extract_aligned_archive_scenes(
    scenes: Iterable[AerpawAlignedScene],
    *,
    destination: str | Path,
    expected_sites: Iterable[str],
) -> list[AerpawAlignedScene]:
    """Extract only selected ZIP members into a compact per-site directory.

    Existing byte-identical files are reused.  Any conflicting file aborts the
    operation, preventing an incomplete download or stale extraction from being
    silently accepted.
    """

    destination = Path(destination).resolve()
    sites = tuple(expected_sites)
    output: list[AerpawAlignedScene] = []
    archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for scene in scenes:
            extracted: dict[str, AerpawPair] = {}
            for site in sites:
                pair = scene.pairs_by_site[site]
                if isinstance(pair, AerpawPair):
                    extracted[site] = pair
                    continue
                if not isinstance(pair, AerpawZipPair):
                    raise TypeError("aligned scene contains an unsupported pair type")
                handle = archives.setdefault(pair.archive_path, zipfile.ZipFile(pair.archive_path))
                site_root = destination / site
                site_root.mkdir(parents=True, exist_ok=True)
                output_paths: dict[str, Path] = {}
                for member_name in (pair.meta_member, pair.data_member):
                    safe_name = _safe_zip_member(member_name).name
                    target = site_root / safe_name
                    info = handle.getinfo(member_name)
                    if target.exists():
                        if target.stat().st_size != info.file_size:
                            raise ValueError(f"existing compact file has wrong size: {target}")
                    else:
                        temporary = target.with_suffix(target.suffix + ".tmp")
                        with handle.open(info) as source, temporary.open("wb") as sink:
                            shutil.copyfileobj(source, sink, length=1024 * 1024)
                        if temporary.stat().st_size != info.file_size:
                            temporary.unlink(missing_ok=True)
                            raise ValueError(f"extracted compact file has wrong size: {target}")
                        temporary.replace(target)
                    output_paths[member_name] = target
                extracted[site] = AerpawPair(
                    pair.stem,
                    output_paths[pair.meta_member],
                    output_paths[pair.data_member],
                    pair.timestamp_local,
                )
            output.append(
                AerpawAlignedScene(
                    scene.scene_id,
                    scene.anchor_datetime_utc,
                    extracted,
                    scene.offsets_from_anchor_s,
                )
            )
    finally:
        for handle in archives.values():
            handle.close()
    return output


def thin_aligned_scenes_by_time(
    scenes: Iterable[AerpawAlignedScene], *, min_separation_s: float
) -> list[AerpawAlignedScene]:
    """Keep deterministic earliest scenes separated by a fixed UTC interval."""

    if not np.isfinite(min_separation_s) or min_separation_s < 0:
        raise ValueError("min_separation_s must be finite and non-negative")
    ordered = sorted(scenes, key=lambda scene: (scene.anchor_datetime_utc, scene.scene_id))
    selected: list[AerpawAlignedScene] = []
    last: datetime | None = None
    for scene in ordered:
        current = datetime.fromisoformat(scene.anchor_datetime_utc.replace("Z", "+00:00"))
        if current.tzinfo is None:
            raise ValueError("aligned scene UTC timestamp must include a timezone")
        if last is None or (current - last).total_seconds() >= min_separation_s:
            selected.append(scene)
            last = current
    return selected


def select_frequency_band(
    sweep: AerpawPowerSweep, *, low_mhz: float, high_mhz: float
) -> AerpawPowerSweep:
    """Return an in-memory sub-band; source provenance paths remain unchanged."""

    if not np.isfinite(low_mhz) or not np.isfinite(high_mhz) or low_mhz >= high_mhz:
        raise ValueError("frequency bounds must be finite and strictly increasing")
    mask = (sweep.frequencies_mhz >= low_mhz) & (sweep.frequencies_mhz < high_mhz)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("requested frequency band contains fewer than two bins")
    return AerpawPowerSweep(
        frequencies_mhz=sweep.frequencies_mhz[mask],
        powers_dbm=sweep.powers_dbm[mask],
        site=sweep.site,
        capture_datetime=sweep.capture_datetime,
        meta_path=sweep.meta_path,
        data_path=sweep.data_path,
        metadata=sweep.metadata,
    )


def robust_power_threshold_dbm(
    sweep: AerpawPowerSweep,
    *,
    sigma_multiplier: float = 3.0,
    gaussian_mad_scale: float = 1.4826,
    sigma_floor_db: float = 0.25,
) -> float:
    """Return a deterministic per-sweep median/MAD CFAR-like threshold.

    This normalizes fixed-site gain and noise-floor offsets without labels,
    model outputs, or information from other scenes.
    """

    for name, value in (
        ("sigma_multiplier", sigma_multiplier),
        ("gaussian_mad_scale", gaussian_mad_scale),
        ("sigma_floor_db", sigma_floor_db),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    median = float(np.median(sweep.powers_dbm))
    mad = float(np.median(np.abs(sweep.powers_dbm.astype(np.float64) - median)))
    robust_sigma = max(float(gaussian_mad_scale * mad), float(sigma_floor_db))
    return median + float(sigma_multiplier) * robust_sigma


def validate_aligned_power_scene(
    scene: AerpawAlignedScene,
    *,
    expected_sites: Iterable[str],
    max_offset_s: float,
    frequency_atol_mhz: float = 1e-5,
    plausible_power_range_dbm: tuple[float, float] = (-200.0, 20.0),
) -> tuple[dict[str, AerpawPowerSweep], list[str]]:
    """Load a matched scene and enforce pre-model integrity/compatibility gates."""

    sites = tuple(expected_sites)
    errors: list[str] = []
    if set(scene.pairs_by_site) != set(sites):
        errors.append("scene site set does not match expected_sites")
        return {}, errors
    if any(abs(float(value)) > max_offset_s for value in scene.offsets_from_anchor_s.values()):
        errors.append("scene exceeds timestamp alignment tolerance")
    sweeps: dict[str, AerpawPowerSweep] = {}
    reference_frequency: np.ndarray | None = None
    low, high = plausible_power_range_dbm
    for site in sites:
        pair = scene.pairs_by_site[site]
        try:
            sweep = load_aerpaw_power_sweep(pair.meta_path, pair.data_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{site} load failed: {exc}")
            continue
        sweeps[site] = sweep
        if sweep.site != site:
            errors.append(f"{site} metadata reports site {sweep.site!r}")
        if np.any(sweep.powers_dbm < low) or np.any(sweep.powers_dbm > high):
            errors.append(f"{site} power values fall outside plausible range [{low}, {high}] dBm")
        if float(np.ptp(sweep.powers_dbm)) < 1e-3:
            errors.append(f"{site} power sweep is effectively constant")
        if reference_frequency is None:
            reference_frequency = sweep.frequencies_mhz
        elif reference_frequency.shape != sweep.frequencies_mhz.shape or not np.allclose(
            reference_frequency, sweep.frequencies_mhz, rtol=0.0, atol=frequency_atol_mhz
        ):
            errors.append(f"{site} frequency grid does not match the anchor grid")
    return sweeps, errors


def channel_power_dbm(sweep: AerpawPowerSweep, n_channels: int, reducer: str = "mean") -> np.ndarray:
    """Aggregate an irregular but monotone frequency grid into equal-width bands."""

    if n_channels < 1 or n_channels > sweep.n_bins:
        raise ValueError("n_channels must be between one and the number of frequency bins")
    edges = np.linspace(sweep.frequencies_mhz[0], sweep.frequencies_mhz[-1], n_channels + 1)
    indices = np.searchsorted(sweep.frequencies_mhz, edges, side="left")
    indices[0], indices[-1] = 0, sweep.n_bins
    values: list[float] = []
    for index in range(n_channels):
        band = sweep.powers_dbm[indices[index] : indices[index + 1]]
        if band.size == 0:
            raise ValueError("frequency grid produced an empty channel")
        if reducer == "mean":
            value = float(np.mean(band, dtype=np.float64))
        elif reducer == "median":
            value = float(np.median(band))
        elif reducer == "max":
            value = float(np.max(band))
        else:
            raise ValueError("reducer must be mean, median, or max")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def channel_occupancy(sweep: AerpawPowerSweep, n_channels: int, threshold_dbm: float) -> np.ndarray:
    """Return the fraction of bins above an explicitly supplied pilot-frozen threshold."""

    if not np.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    edges = np.linspace(sweep.frequencies_mhz[0], sweep.frequencies_mhz[-1], n_channels + 1)
    indices = np.searchsorted(sweep.frequencies_mhz, edges, side="left")
    indices[0], indices[-1] = 0, sweep.n_bins
    return np.asarray(
        [np.mean(sweep.powers_dbm[indices[i] : indices[i + 1]] > threshold_dbm) for i in range(n_channels)],
        dtype=np.float32,
    )


def cleanest_contiguous_block(channel_cost_dbm: np.ndarray, demand_channels: int) -> tuple[int, float]:
    """Select the minimum-power contiguous block and return start index and mean dBm."""

    costs = np.asarray(channel_cost_dbm, dtype=np.float64)
    if costs.ndim != 1 or not np.all(np.isfinite(costs)):
        raise ValueError("channel_cost_dbm must be a finite one-dimensional vector")
    if demand_channels < 1 or demand_channels > costs.size:
        raise ValueError("demand_channels is outside the available channel count")
    means = np.convolve(costs, np.ones(demand_channels) / demand_channels, mode="valid")
    start = int(np.argmin(means))
    return start, float(means[start])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_permanently_excluded_pilot_manifest(
    pairs: Iterable[AerpawPair], *, dataset_root: str | Path, count: int, dataset_id: str
) -> dict[str, Any]:
    """Create a deterministic pilot manifest that cannot masquerade as final data."""

    root = Path(dataset_root).resolve()
    ordered = sorted(pairs, key=lambda item: (item.timestamp_local, item.stem))
    if count < 1 or len(ordered) < count:
        raise ValueError("pilot count must be positive and no larger than the available pair count")
    # Even spacing avoids choosing a convenient contiguous time window after viewing outcomes.
    selected_indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    selected = [ordered[int(index)] for index in selected_indices]
    scenes = []
    for pair in selected:
        scenes.append(
            {
                "scene_id": f"{dataset_id}:pilot:{pair.stem}",
                "collection_event_id": f"{dataset_id}:{pair.stem}",
                "meta_path": str(pair.meta_path.relative_to(root)).replace("\\", "/"),
                "data_path": str(pair.data_path.relative_to(root)).replace("\\", "/"),
                "timestamp_local_america_new_york": pair.timestamp_local,
            }
        )
    return {
        "manifest_version": "1.0",
        "dataset_id": dataset_id,
        "status": "pilot_only_permanently_excluded_from_final",
        "selection_rule": "deterministic_even_spacing_over_filename_timestamp_before_model_output_access",
        "scene_count": len(scenes),
        "scenes": scenes,
        "labels_or_model_outputs_accessed": False,
        "final_access_consumed": False,
        "claim_boundary": "Pilot scenes are permanently ineligible for stage-4 final registration.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
