from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage

from .preprocessing import STFTResult
from .types import SignalBox


def enhance_stft_for_detection(power_db: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Return a detection-oriented STFT map.

    Modes:
    - raw: original dB power.
    - freq_median: remove the per-frequency median over time. This highlights
      bursts and OFDM symbols against stationary spectral structure.
    - time_median: remove the per-time median over frequency. This highlights
      narrowband structures against wideband time bursts.
    - local_median: subtract a local 2-D median background.
    - local_zscore: local median subtraction divided by local robust scale.
    """

    power = np.asarray(power_db, dtype=np.float32)
    mode = mode.lower()
    if mode == "raw":
        return power
    if mode == "freq_median":
        return power - np.median(power, axis=1, keepdims=True)
    if mode == "time_median":
        return power - np.median(power, axis=0, keepdims=True)
    if mode == "local_median":
        background = ndimage.median_filter(power, size=(17, 17), mode="nearest")
        return power - background
    if mode == "local_zscore":
        background = ndimage.median_filter(power, size=(17, 17), mode="nearest")
        residual = power - background
        scale = ndimage.median_filter(np.abs(residual), size=(17, 17), mode="nearest")
        return residual / (1.4826 * scale + 1e-3)
    raise ValueError(f"Unknown detection enhancement mode: {mode}")


def robust_threshold_db(power_db: np.ndarray, sigma: float = 6.0) -> float:
    """Median/MAD threshold for noisy STFT power maps."""

    median = float(np.median(power_db))
    mad = float(np.median(np.abs(power_db - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(power_db))
    return median + sigma * max(robust_std, 1e-6)


def _axis_edges(values: np.ndarray) -> np.ndarray:
    """Convert axis centers to approximate bin edges."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    if values.size == 1:
        return np.array([values[0], values[0]], dtype=np.float64)
    diffs = np.diff(values)
    left = values[0] - diffs[0] / 2.0
    mids = values[:-1] + diffs / 2.0
    right = values[-1] + diffs[-1] / 2.0
    return np.concatenate([[left], mids, [right]])


def energy_detector(
    stft: STFTResult,
    threshold_db: float | None = None,
    min_cells: int = 8,
    label: str = "occupied",
) -> List[SignalBox]:
    """A simple threshold-based baseline on STFT power.

    It returns one coarse time-frequency box covering all active cells. This is
    intentionally simple: it is a sanity baseline, not the final detector.
    """

    power = stft.power_db
    if threshold_db is None:
        threshold_db = float(np.median(power) + 6.0)
    active = power > threshold_db
    if int(active.sum()) < min_cells:
        return []

    freq_idx, time_idx = np.where(active)
    f0 = float(stft.freqs_hz[max(0, freq_idx.min())])
    f1 = float(stft.freqs_hz[min(len(stft.freqs_hz) - 1, freq_idx.max())])
    t0 = float(stft.times_s[max(0, time_idx.min())])
    t1 = float(stft.times_s[min(len(stft.times_s) - 1, time_idx.max())])
    return [
        SignalBox(
            label=label,
            t_start_s=min(t0, t1),
            t_end_s=max(t0, t1),
            f_low_hz=min(f0, f1),
            f_high_hz=max(f0, f1),
            confidence=1.0,
        )
    ]


def connected_component_energy_detector(
    stft: STFTResult,
    threshold_db: float | None = None,
    threshold_sigma: float = 6.0,
    enhancement: str = "raw",
    enhanced_power: np.ndarray | None = None,
    min_cells: int = 12,
    min_time_bins: int = 1,
    min_freq_bins: int = 1,
    max_boxes: int = 64,
    binary_opening: bool = False,
    label: str = "occupied",
) -> List[SignalBox]:
    """Detect multiple time-frequency regions by thresholding STFT power.

    This is the first serious non-learning baseline. It is intentionally
    transparent: threshold the STFT, find connected components, convert each
    component to a time-frequency box, and sort boxes by component energy.
    """

    power = enhance_stft_for_detection(stft.power_db, mode=enhancement) if enhanced_power is None else np.asarray(enhanced_power)
    if threshold_db is None:
        threshold_db = robust_threshold_db(power, sigma=threshold_sigma)

    active = power > threshold_db
    if binary_opening:
        active = ndimage.binary_opening(active, structure=np.ones((2, 2), dtype=bool))

    structure = np.ones((3, 3), dtype=np.int8)
    labeled, n_components = ndimage.label(active, structure=structure)
    if n_components <= 0:
        return []

    component_slices = ndimage.find_objects(labeled)
    component_counts = np.bincount(labeled.ravel(), minlength=n_components + 1)
    freq_edges = _axis_edges(stft.freqs_hz)
    time_edges = _axis_edges(stft.times_s)
    boxes_with_scores: list[tuple[float, SignalBox]] = []

    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        cell_count = int(component_counts[component_id])
        if cell_count < min_cells:
            continue

        f_slice, t_slice = component_slice
        f_min = int(f_slice.start)
        f_max = int(f_slice.stop - 1)
        t_min = int(t_slice.start)
        t_max = int(t_slice.stop - 1)
        if (f_max - f_min + 1) < min_freq_bins or (t_max - t_min + 1) < min_time_bins:
            continue

        local_labels = labeled[component_slice]
        local_power = power[component_slice]
        component_power = local_power[local_labels == component_id]
        score = float(np.mean(component_power - threshold_db))
        confidence = float(1.0 / (1.0 + np.exp(-score / 6.0)))
        box = SignalBox(
            label=label,
            t_start_s=float(time_edges[t_min]),
            t_end_s=float(time_edges[t_max + 1]),
            f_low_hz=float(freq_edges[f_min]),
            f_high_hz=float(freq_edges[f_max + 1]),
            confidence=confidence,
        )
        boxes_with_scores.append((score * cell_count, box))

    boxes_with_scores.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes_with_scores[:max_boxes]]


def binary_mask_to_boxes(
    active: np.ndarray,
    stft: STFTResult,
    min_cells: int = 12,
    min_time_bins: int = 1,
    min_freq_bins: int = 1,
    max_boxes: int = 128,
    label: str = "occupied",
) -> List[SignalBox]:
    """Convert a binary STFT occupancy mask to time-frequency boxes."""

    active = np.asarray(active, dtype=bool)
    if active.shape != stft.power_db.shape:
        raise ValueError(f"active shape {active.shape} does not match STFT shape {stft.power_db.shape}")
    labeled, n_components = ndimage.label(active, structure=np.ones((3, 3), dtype=np.int8))
    if n_components <= 0:
        return []

    component_slices = ndimage.find_objects(labeled)
    component_counts = np.bincount(labeled.ravel(), minlength=n_components + 1)
    freq_edges = _axis_edges(stft.freqs_hz)
    time_edges = _axis_edges(stft.times_s)
    boxes_with_scores: list[tuple[int, SignalBox]] = []

    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        cell_count = int(component_counts[component_id])
        if cell_count < min_cells:
            continue
        f_slice, t_slice = component_slice
        f_min = int(f_slice.start)
        f_max = int(f_slice.stop - 1)
        t_min = int(t_slice.start)
        t_max = int(t_slice.stop - 1)
        if (f_max - f_min + 1) < min_freq_bins or (t_max - t_min + 1) < min_time_bins:
            continue

        boxes_with_scores.append(
            (
                cell_count,
                SignalBox(
                    label=label,
                    t_start_s=float(time_edges[t_min]),
                    t_end_s=float(time_edges[t_max + 1]),
                    f_low_hz=float(freq_edges[f_min]),
                    f_high_hz=float(freq_edges[f_max + 1]),
                    confidence=1.0,
                ),
            )
        )

    boxes_with_scores.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes_with_scores[:max_boxes]]
