"""Utilities for aligned multi-receiver LoRaIQ SigMF observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


def load_cf32_sigmf(data_path: str | Path, meta_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a single-channel little-endian complex-float SigMF recording."""

    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    global_meta = metadata.get("global", {})
    if global_meta.get("core:datatype") != "cf32_le":
        raise ValueError("LoRaIQ pilot expects core:datatype=cf32_le")
    if int(global_meta.get("core:num_channels", 1)) != 1:
        raise ValueError("LoRaIQ pilot expects one SigMF channel")
    iq = np.fromfile(Path(data_path), dtype=np.dtype("<c8"))
    if iq.ndim != 1 or iq.size == 0 or not np.all(np.isfinite(iq)):
        raise ValueError("SigMF data must contain finite one-dimensional complex samples")
    return iq.astype(np.complex64, copy=False), metadata


def channel_band_occupancy(
    f_low_hz: float,
    f_high_hz: float,
    sample_rate_hz: float,
    n_channels: int,
) -> np.ndarray:
    """Return fractional overlap of a labelled frequency band with equal channels."""

    if sample_rate_hz <= 0 or n_channels < 1 or f_high_hz <= f_low_hz:
        raise ValueError("invalid frequency-band occupancy inputs")
    edges = np.linspace(-sample_rate_hz / 2.0, sample_rate_hz / 2.0, n_channels + 1)
    widths = np.diff(edges)
    overlap = np.maximum(
        0.0,
        np.minimum(float(f_high_hz), edges[1:]) - np.maximum(float(f_low_hz), edges[:-1]),
    )
    return np.clip(overlap / widths, 0.0, 1.0).astype(np.float32)


def welch_channel_excess_db(
    iq: np.ndarray,
    frame_start: int,
    frame_count: int,
    sample_rate_hz: float,
    n_channels: int,
    nperseg: int = 1024,
    guard_samples: int = 1024,
) -> np.ndarray:
    """Estimate per-channel in-frame excess power relative to off-frame samples."""

    values = np.asarray(iq, dtype=np.complex64)
    start = int(frame_start)
    count = int(frame_count)
    end = start + count
    if values.ndim != 1 or start < 0 or count <= 0 or end > values.size:
        raise ValueError("frame interval must be inside the IQ recording")
    if sample_rate_hz <= 0 or n_channels < 1 or nperseg < 8 or guard_samples < 0:
        raise ValueError("invalid Welch detector settings")
    before = values[: max(0, start - int(guard_samples))]
    after = values[min(values.size, end + int(guard_samples)) :]
    noise = np.concatenate([before, after])
    frame = values[start:end]
    segment = min(int(nperseg), frame.size, noise.size)
    if segment < 8:
        raise ValueError("insufficient off-frame samples for a noise reference")

    def spectrum(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        freqs, power = signal.welch(
            samples,
            fs=float(sample_rate_hz),
            window="hann",
            nperseg=segment,
            noverlap=segment // 2,
            nfft=segment,
            return_onesided=False,
            scaling="density",
        )
        return np.fft.fftshift(freqs), np.fft.fftshift(power)

    freqs, frame_power = spectrum(frame)
    _, noise_power = spectrum(noise)
    edges = np.linspace(-sample_rate_hz / 2.0, sample_rate_hz / 2.0, n_channels + 1)
    excess = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError("Welch resolution is too small for the requested channel count")
        ratio = (float(np.mean(frame_power[mask])) + 1e-20) / (
            float(np.mean(noise_power[mask])) + 1e-20
        )
        excess.append(10.0 * np.log10(ratio))
    return np.asarray(excess, dtype=np.float32)


def excess_db_to_probability(
    excess_db: np.ndarray,
    threshold_db: float,
    temperature_db: float,
) -> np.ndarray:
    """Map detector evidence to a bounded soft occupancy semantic vector."""

    if temperature_db <= 0:
        raise ValueError("temperature_db must be positive")
    logits = (np.asarray(excess_db, dtype=np.float64) - float(threshold_db)) / float(temperature_db)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)


def quantize_probability(values: np.ndarray, bits: int) -> np.ndarray:
    """Uniformly quantize occupancy probabilities to the declared payload precision."""

    if bits < 1 or bits > 16:
        raise ValueError("probability quantization bits must be in [1, 16]")
    levels = (1 << int(bits)) - 1
    return (np.rint(np.clip(values, 0.0, 1.0) * levels) / levels).astype(np.float32)
