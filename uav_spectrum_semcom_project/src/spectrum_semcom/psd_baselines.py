"""Classical PSD summaries used by the stage-2 fair-link comparison.

The routines intentionally operate on the available spectrogram intensity
proxy.  They are not presented as calibrated receiver PSD in dBm.
"""
from __future__ import annotations

import numpy as np


def high_resolution_channel_psd(image: np.ndarray, n_channels: int, bins_per_channel: int, axis: str) -> np.ndarray:
    """Return mean-energy sub-bins, ordered channel by channel."""
    if n_channels <= 0 or bins_per_channel <= 0:
        raise ValueError("n_channels and bins_per_channel must be positive")
    value = np.asarray(image, dtype=np.float32) / 255.0
    dimension = 0 if axis == "y" else 1
    channels = np.array_split(value, n_channels, axis=dimension)
    summaries: list[float] = []
    for channel in channels:
        parts = np.array_split(channel, bins_per_channel, axis=dimension)
        summaries.extend(float(np.mean(part)) for part in parts)
    return np.asarray(summaries, dtype=np.float32)


def reduce_high_resolution_psd(values: np.ndarray, n_channels: int, bins_per_channel: int, reducer: str) -> np.ndarray:
    """Map transmitted sub-bin energies back to a resource-selection score."""
    array = np.asarray(values, dtype=np.float32)
    if array.size != n_channels * bins_per_channel:
        raise ValueError("PSD vector length does not match declared resolution")
    grouped = array.reshape(n_channels, bins_per_channel)
    if reducer == "mean":
        return np.mean(grouped, axis=1)
    if reducer == "max":
        return np.max(grouped, axis=1)
    if reducer == "p75":
        return np.quantile(grouped, 0.75, axis=1).astype(np.float32)
    raise ValueError("reducer must be mean, max, or p75")


def cfar_occupancy_summary(values: np.ndarray, n_channels: int, bins_per_channel: int, threshold_sigma: float) -> np.ndarray:
    """A robust CA-CFAR-like binary occupancy fraction per candidate channel.

    Noise is estimated from the lower half of the frame's sub-bin energies,
    preventing a strong occupied band from setting its own threshold.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.size != n_channels * bins_per_channel:
        raise ValueError("PSD vector length does not match declared resolution")
    floor = np.sort(array)[: max(1, array.size // 2)]
    median = float(np.median(floor))
    mad = float(np.median(np.abs(floor - median)))
    sigma = max(1.4826 * mad, 1e-4)
    flags = array > median + threshold_sigma * sigma
    return flags.reshape(n_channels, bins_per_channel).mean(axis=1).astype(np.float32)
