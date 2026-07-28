"""Pure helpers for the frozen AERPAW final-inference orchestration."""

from __future__ import annotations

import numpy as np

from .multigranular_semantics import SemanticQuality


def power_block_regret_db(belief: np.ndarray, channel_power_dbm: np.ndarray, demand_channels: int) -> float:
    """Return the received-power gap of the block selected from ``belief``."""

    estimate = np.asarray(belief, dtype=np.float64).reshape(-1)
    power = np.asarray(channel_power_dbm, dtype=np.float64).reshape(-1)
    if estimate.shape != power.shape or not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(power)):
        raise ValueError("belief and channel power must be equally shaped and finite")
    if np.any((estimate < 0) | (estimate > 1)):
        raise ValueError("belief must be in [0, 1]")
    demand = int(demand_channels)
    if not 1 <= demand <= estimate.size:
        raise ValueError("invalid demand_channels")
    kernel = np.ones(demand, dtype=np.float64) / demand
    selected = int(np.argmin(np.convolve(estimate, kernel, mode="valid")))
    costs = np.convolve(power, kernel, mode="valid")
    return float(max(0.0, costs[selected] - np.min(costs)))


def occupancy_quality(values: np.ndarray) -> SemanticQuality:
    """Create the frozen, deployable quality payload without truth access."""

    occupancy = np.asarray(values, dtype=np.float64).reshape(-1)
    if occupancy.size < 1 or not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("occupancy must be a non-empty finite vector in [0, 1]")
    clipped = np.clip(occupancy, 1e-12, 1 - 1e-12)
    entropy = -np.mean(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
    confidence = np.mean(np.abs(occupancy - 0.5) * 2.0)
    return SemanticQuality(0.0, float(confidence), float(entropy), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
