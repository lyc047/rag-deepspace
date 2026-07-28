"""Traceable construction of controlled multi-target scenes from frozen frames."""

from __future__ import annotations

import numpy as np


def grouped_source_indices(n_sources: int, sources_per_scene: int) -> list[np.ndarray]:
    """Partition source indices without replacement; discard an incomplete tail."""

    if n_sources < 1:
        return []
    if sources_per_scene < 1:
        raise ValueError("sources_per_scene must be positive")
    usable = n_sources - n_sources % sources_per_scene
    return [np.arange(start, start + sources_per_scene, dtype=np.int64) for start in range(0, usable, sources_per_scene)]


def sliding_source_indices(n_sources: int, sources_per_scene: int, stride: int = 1) -> list[np.ndarray]:
    """Build traceable temporally overlapping scenes without source reuse across splits."""

    if sources_per_scene < 1:
        raise ValueError("sources_per_scene must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if n_sources < sources_per_scene:
        return []
    return [
        np.arange(start, start + sources_per_scene, dtype=np.int64)
        for start in range(0, n_sources - sources_per_scene + 1, stride)
    ]


def compose_max_hold_scene(images: list[np.ndarray]) -> np.ndarray:
    """Combine aligned source spectrograms into one explicitly synthetic scene.

    Max-hold preserves narrowband targets from every source.  The caller must
    retain the source-frame IDs and union the corresponding truth boxes.
    """

    if not images:
        raise ValueError("at least one source image is required")
    arrays = [np.asarray(image, dtype=np.uint8) for image in images]
    shape = arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays):
        raise ValueError("all source images must be equally shaped grayscale arrays")
    return np.maximum.reduce(arrays).astype(np.uint8)
