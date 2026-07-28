"""Controlled same-source I/Q replay utilities for multi-receiver diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SignalBox


@dataclass(frozen=True)
class ReplayChannel:
    sensing_snr_db: float
    gain_db: float = 0.0
    frequency_offset_hz: float = 0.0
    multipath_taps: tuple[complex, ...] = (1.0 + 0.0j,)


def replay_iq_view(
    iq: np.ndarray,
    sample_rate_hz: float,
    channel: ReplayChannel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replay one source recording through gain, CFO, multipath, and complex AWGN."""

    source = np.asarray(iq, dtype=np.complex64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("iq must be a non-empty one-dimensional array")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    taps = np.asarray(channel.multipath_taps, dtype=np.complex64)
    if taps.ndim != 1 or taps.size == 0 or not np.all(np.isfinite(taps)):
        raise ValueError("multipath_taps must be a finite non-empty vector")
    filtered = np.convolve(source, taps, mode="full")[: source.size]
    time = np.arange(source.size, dtype=np.float64) / float(sample_rate_hz)
    shifted = filtered * np.exp(2j * np.pi * float(channel.frequency_offset_hz) * time)
    scaled = shifted * (10.0 ** (float(channel.gain_db) / 20.0))
    signal_power = max(float(np.mean(np.abs(scaled) ** 2)), 1e-12)
    noise_power = signal_power / (10.0 ** (float(channel.sensing_snr_db) / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=source.size) + 1j * rng.normal(size=source.size)
    )
    return (scaled + noise).astype(np.complex64)


def frequency_channel_occupancy(
    boxes: list[SignalBox],
    sample_rate_hz: float,
    frame_duration_s: float,
    n_channels: int,
) -> np.ndarray:
    """Map time-frequency boxes to fractional occupancy of equal frequency channels."""

    if sample_rate_hz <= 0 or frame_duration_s <= 0 or n_channels < 1:
        raise ValueError("sample rate, frame duration, and channel count must be positive")
    edges = np.linspace(-sample_rate_hz / 2.0, sample_rate_hz / 2.0, n_channels + 1)
    occupancy = np.zeros(n_channels, dtype=np.float64)
    for box in boxes:
        time_fraction = np.clip((float(box.t_end_s) - float(box.t_start_s)) / frame_duration_s, 0.0, 1.0)
        for channel in range(n_channels):
            overlap = max(0.0, min(float(box.f_high_hz), edges[channel + 1]) - max(float(box.f_low_hz), edges[channel]))
            occupancy[channel] += time_fraction * overlap / max(edges[channel + 1] - edges[channel], 1e-12)
    return np.clip(occupancy, 0.0, 1.0).astype(np.float32)


def reference_rf_energy_j(
    transmitted_bits: int,
    attempts: int,
    duration_s: float,
    symbol_rate_overhead_s: float,
    transmit_power_w: float = 1.0,
) -> float:
    """Reference RF energy using link airtime after removing non-RF fixed delays."""

    if transmitted_bits < 0 or attempts < 0 or duration_s < 0 or symbol_rate_overhead_s < 0 or transmit_power_w < 0:
        raise ValueError("energy inputs must be non-negative")
    if transmitted_bits == 0:
        return 0.0
    airtime = max(0.0, float(duration_s) - int(attempts) * float(symbol_rate_overhead_s))
    return float(transmit_power_w * airtime)
