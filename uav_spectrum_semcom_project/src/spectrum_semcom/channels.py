from __future__ import annotations

import numpy as np


def awgn(x: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add AWGN to a real or complex signal."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    power = float(np.mean(np.abs(x) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    if np.iscomplexobj(x):
        noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(x.shape)
    return x + noise.astype(x.dtype, copy=False)


def binary_erasure(bits: np.ndarray, erasure_prob: float, seed: int | None = None, erasure_value: int = -1) -> np.ndarray:
    """Apply a simple packet/bit erasure model to integer bits."""

    rng = np.random.default_rng(seed)
    bits = np.asarray(bits).copy()
    mask = rng.random(bits.shape) < erasure_prob
    bits[mask] = erasure_value
    return bits

