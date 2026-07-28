from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


FIESTA_FILE_RE = re.compile(r"CF(?P<center_mhz>\d+)MHz_BD(?P<bandwidth_mhz>\d+)MHz_(?P<n_bins>\d+)\.txt$")


@dataclass(frozen=True)
class FiestaSpectrumFile:
    path: Path
    device: str
    center_hz: float
    bandwidth_hz: float
    n_bins: int


@dataclass(frozen=True)
class FiestaEvent:
    start_bin: int
    end_bin: int
    f_low_hz: float
    f_high_hz: float
    peak_dbm: float
    mean_dbm: float


@dataclass(frozen=True)
class FiestaFrameSummary:
    device: str
    center_hz: float
    bandwidth_hz: float
    timestamp_s: int
    latitude: float
    longitude: float
    n_bins: int
    n_events: int
    raw_float32_bits: int
    raw_int16_bits: int
    semantic_event_bits: int


def discover_fiesta_files(root: str | Path) -> list[FiestaSpectrumFile]:
    root_path = Path(root)
    files: list[FiestaSpectrumFile] = []
    for path in sorted(root_path.rglob("CF*MHz_BD*MHz_*.txt")):
        match = FIESTA_FILE_RE.match(path.name)
        if not match:
            continue
        device = path.parent.parent.name
        files.append(
            FiestaSpectrumFile(
                path=path,
                device=device,
                center_hz=float(match.group("center_mhz")) * 1e6,
                bandwidth_hz=float(match.group("bandwidth_mhz")) * 1e6,
                n_bins=int(match.group("n_bins")),
            )
        )
    return files


def load_fiesta_file(file: FiestaSpectrumFile, max_rows: int | None = None) -> np.ndarray:
    data = np.loadtxt(file.path, dtype=np.float32, max_rows=max_rows)
    if data.ndim == 1:
        data = data[None, :]
    expected_cols = 3 + file.n_bins
    if data.shape[1] != expected_cols:
        raise ValueError(f"{file.path} has {data.shape[1]} columns, expected {expected_cols}")
    return data


def contiguous_true_regions(mask: np.ndarray, min_bins: int = 2) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_bins:
                regions.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_bins:
        regions.append((start, len(mask)))
    return regions


def extract_fiesta_events(
    psd_dbm: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
    threshold_margin_db: float = 8.0,
    min_bins: int = 2,
) -> list[FiestaEvent]:
    """Extract coarse occupied-band events from one PSD snapshot.

    The threshold is intentionally simple and local: bins above the snapshot
    median plus a margin are treated as candidate occupied bins. It is a
    transparent baseline for payload accounting, not a final detector.
    """

    values = np.asarray(psd_dbm, dtype=np.float32)
    threshold = float(np.median(values) + threshold_margin_db)
    mask = values >= threshold
    bin_hz = bandwidth_hz / len(values)
    f_min = center_hz - bandwidth_hz / 2
    events: list[FiestaEvent] = []
    for start, end in contiguous_true_regions(mask, min_bins=min_bins):
        segment = values[start:end]
        events.append(
            FiestaEvent(
                start_bin=start,
                end_bin=end,
                f_low_hz=f_min + start * bin_hz,
                f_high_hz=f_min + end * bin_hz,
                peak_dbm=float(np.max(segment)),
                mean_dbm=float(np.mean(segment)),
            )
        )
    return events


def fiesta_raw_float32_bits(n_bins: int, include_geo_time: bool = True) -> int:
    metadata_values = 3 if include_geo_time else 0
    return int(n_bins + metadata_values) * 32


def fiesta_raw_int16_bits(n_bins: int, include_geo_time: bool = True) -> int:
    # PSD values are commonly quantized before upload; geo/time metadata remains 32-bit here.
    metadata_bits = 3 * 32 if include_geo_time else 0
    return int(n_bins) * 16 + metadata_bits


def fiesta_semantic_event_bits(n_events: int, header_bits: int = 160, per_event_bits: int = 48) -> int:
    # event: start bin, end bin, peak/mean power, confidence/flags.
    return int(header_bits) + int(n_events) * int(per_event_bits)


def summarize_fiesta_file(
    file: FiestaSpectrumFile,
    max_rows: int | None = None,
    threshold_margin_db: float = 8.0,
    min_bins: int = 2,
) -> list[FiestaFrameSummary]:
    data = load_fiesta_file(file, max_rows=max_rows)
    summaries: list[FiestaFrameSummary] = []
    for row in data:
        latitude, longitude, timestamp = float(row[0]), float(row[1]), int(row[2])
        psd = row[3:]
        events = extract_fiesta_events(
            psd,
            center_hz=file.center_hz,
            bandwidth_hz=file.bandwidth_hz,
            threshold_margin_db=threshold_margin_db,
            min_bins=min_bins,
        )
        summaries.append(
            FiestaFrameSummary(
                device=file.device,
                center_hz=file.center_hz,
                bandwidth_hz=file.bandwidth_hz,
                timestamp_s=timestamp,
                latitude=latitude,
                longitude=longitude,
                n_bins=file.n_bins,
                n_events=len(events),
                raw_float32_bits=fiesta_raw_float32_bits(file.n_bins),
                raw_int16_bits=fiesta_raw_int16_bits(file.n_bins),
                semantic_event_bits=fiesta_semantic_event_bits(len(events)),
            )
        )
    return summaries
