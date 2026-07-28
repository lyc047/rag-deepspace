"""Deterministic measurement-derived spectrum stress trajectories for Stage 5."""

from __future__ import annotations

import numpy as np


def _site_positions(site_ids: np.ndarray) -> list[np.ndarray]:
    sites = np.asarray(site_ids).astype(str)
    return [np.flatnonzero(sites == site) for site in sorted(set(sites))]


def apply_spectrum_regime(
    channel_power_dbm: np.ndarray,
    site_ids: np.ndarray,
    *,
    regime: str,
    parameters: dict,
) -> np.ndarray:
    """Return a transformed copy without changing scene or site membership."""
    power = np.asarray(channel_power_dbm, dtype=np.float64)
    sites = np.asarray(site_ids)
    if power.ndim != 2 or sites.shape != (power.shape[0],):
        raise ValueError("power and site arrays have incompatible shapes")
    if not np.all(np.isfinite(power)):
        raise ValueError("power contains non-finite values")
    output = power.copy()
    if regime == "observed":
        return output
    n_channels = power.shape[1]
    if regime == "gradual_drift":
        amplitude = float(parameters["maximum_edge_offset_db"])
        edge = np.linspace(-amplitude, amplitude, n_channels)
        for positions in _site_positions(sites):
            denominator = max(len(positions) - 1, 1)
            for local, position in enumerate(positions):
                output[position] += (local / denominator) * edge
        return output
    if regime == "abrupt_band_hop":
        period = int(parameters["period_scenes"])
        shift = int(parameters["shift_channels"])
        if period < 1 or shift < 1:
            raise ValueError("invalid abrupt-hop parameters")
        for positions in _site_positions(sites):
            for local, position in enumerate(positions):
                epoch = local // period
                output[position] = np.roll(
                    power[position],
                    (epoch * shift) % n_channels,
                )
        return output
    if regime == "burst_block_interference":
        period = int(parameters["period_scenes"])
        duration = int(parameters["duration_scenes"])
        width = int(parameters["block_width_channels"])
        amplitude = float(parameters["interference_db"])
        stride = int(parameters["block_start_stride"])
        if (
            period < 1
            or not 1 <= duration <= period
            or not 1 <= width <= n_channels
            or stride < 1
        ):
            raise ValueError("invalid burst-interference parameters")
        candidates = n_channels - width + 1
        for positions in _site_positions(sites):
            for local, position in enumerate(positions):
                if local % period >= duration:
                    continue
                window = local // period
                start = (window * stride) % candidates
                output[position, start : start + width] += amplitude
        return output
    raise ValueError(f"unknown spectrum stress regime: {regime}")

