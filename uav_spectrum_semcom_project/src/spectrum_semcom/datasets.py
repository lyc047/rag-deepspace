from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .preprocessing import STFTResult
from .types import SignalBox


def boxes_to_occupancy_mask(stft: STFTResult, boxes: Iterable[SignalBox]) -> np.ndarray:
    """Rasterize time-frequency boxes onto an STFT grid."""

    mask = np.zeros_like(stft.power_db, dtype=np.float32)
    freqs = stft.freqs_hz
    times = stft.times_s
    if len(freqs) == 0 or len(times) == 0:
        return mask

    for box in boxes:
        f_low, f_high = sorted([box.f_low_hz, box.f_high_hz])
        t_start, t_end = sorted([box.t_start_s, box.t_end_s])
        f_idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        t_idx = np.where((times >= t_start) & (times <= t_end))[0]
        if f_idx.size == 0 or t_idx.size == 0:
            continue
        mask[f_idx.min() : f_idx.max() + 1, t_idx.min() : t_idx.max() + 1] = 1.0
    return mask


def normalize_stft_patch(power_db: np.ndarray) -> np.ndarray:
    """Robustly normalize a power patch for CNN input."""

    x = np.asarray(power_db, dtype=np.float32)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad if mad > 1e-6 else np.std(x) + 1e-6
    return np.clip((x - median) / scale, -8.0, 8.0).astype(np.float32)


@dataclass(frozen=True)
class PatchSet:
    x: np.ndarray
    y: np.ndarray
    positions: List[Tuple[int, int]]


def sample_stft_patches(
    power_db: np.ndarray,
    mask: np.ndarray,
    patch_shape: tuple[int, int] = (128, 128),
    max_patches: int = 96,
    positive_fraction: float = 0.6,
    seed: int = 0,
) -> PatchSet:
    """Sample balanced STFT patches for a small occupancy detector."""

    rng = np.random.default_rng(seed)
    height, width = power_db.shape
    ph, pw = patch_shape
    if height < ph or width < pw:
        raise ValueError(f"patch_shape={patch_shape} larger than power map shape={power_db.shape}")

    positions: List[Tuple[int, int]] = []
    positive_cells = np.argwhere(mask > 0.5)
    n_pos = int(max_patches * positive_fraction) if positive_cells.size else 0
    n_neg = max_patches - n_pos

    for _ in range(n_pos):
        f, t = positive_cells[rng.integers(0, len(positive_cells))]
        f0 = int(np.clip(f - ph // 2, 0, height - ph))
        t0 = int(np.clip(t - pw // 2, 0, width - pw))
        positions.append((f0, t0))

    attempts = 0
    while len(positions) < max_patches and attempts < max_patches * 20:
        attempts += 1
        f0 = int(rng.integers(0, height - ph + 1))
        t0 = int(rng.integers(0, width - pw + 1))
        patch_mask = mask[f0 : f0 + ph, t0 : t0 + pw]
        if len(positions) < n_pos or patch_mask.mean() < 0.2:
            positions.append((f0, t0))

    xs = []
    ys = []
    for f0, t0 in positions:
        patch = normalize_stft_patch(power_db[f0 : f0 + ph, t0 : t0 + pw])
        target = mask[f0 : f0 + ph, t0 : t0 + pw].astype(np.float32)
        xs.append(patch[None, :, :])
        ys.append(target[None, :, :])

    return PatchSet(x=np.stack(xs), y=np.stack(ys), positions=positions)

