"""Data primitives for frozen-detector Gate A caches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def parse_namespaced_frame_id(frame_id: str) -> tuple[str, str]:
    if not isinstance(frame_id, str) or frame_id.count(":") != 1:
        raise ValueError("frame id must use the source_split:stem namespace")
    split, stem = frame_id.split(":", 1)
    if split not in {"train", "val", "test"} or not stem or "/" in stem or "\\" in stem:
        raise ValueError("invalid source split or frame stem")
    return split, stem


def resolve_raddet_paths(root: str | Path, frame_id: str) -> tuple[Path, Path]:
    split, stem = parse_namespaced_frame_id(frame_id)
    root_path = Path(root)
    return root_path / "images" / split / f"{stem}.png", root_path / "labels" / split / f"{stem}.txt"


def pool_probability_mask(mask: np.ndarray, n_channels: int, axis: str) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("mask must be a finite 2-D probability array in [0, 1]")
    channels = int(n_channels)
    dimension = values.shape[0] if axis == "y" else values.shape[1] if axis == "x" else 0
    if channels < 1 or dimension == 0 or dimension % channels:
        raise ValueError("axis must be x/y and its dimension divisible by n_channels")
    if axis == "y":
        return values.reshape(channels, dimension // channels, values.shape[1]).mean(axis=(1, 2))
    return values.reshape(values.shape[0], channels, dimension // channels).mean(axis=(0, 2))


def validate_cache_arrays(
    scene_ids: Sequence[str],
    base_occupancy: np.ndarray,
    truth_occupancy: np.ndarray,
    detector_confidence: np.ndarray,
    n_channels: int,
) -> list[str]:
    errors: list[str] = []
    count = len(scene_ids)
    if count < 1 or len(set(str(value) for value in scene_ids)) != count:
        errors.append("scene_ids must be non-empty and unique")
    expected = (count, int(n_channels))
    for name, values in (("base_occupancy", base_occupancy), ("truth_occupancy", truth_occupancy)):
        array = np.asarray(values)
        if array.shape != expected or not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            errors.append(f"{name} must have shape {expected} with finite values in [0, 1]")
    confidence = np.asarray(detector_confidence)
    if confidence.shape != (count,) or not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        errors.append("detector_confidence must contain one finite [0, 1] value per scene")
    return errors


def validate_multinode_cache_arrays(
    scene_ids: Sequence[str],
    node_occupancy: np.ndarray,
    truth_occupancy: np.ndarray,
    node_quality: np.ndarray,
    quality_availability: np.ndarray,
    n_nodes: int,
    n_channels: int,
) -> list[str]:
    errors: list[str] = []
    count = len(scene_ids)
    occupancy = np.asarray(node_occupancy); truth = np.asarray(truth_occupancy); quality = np.asarray(node_quality)
    if occupancy.shape != (count, n_nodes, n_channels) or not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0)|(occupancy > 1)):
        errors.append("node_occupancy has invalid shape or values")
    if truth.shape != (count, n_channels) or not np.all(np.isfinite(truth)) or np.any((truth < 0)|(truth > 1)):
        errors.append("truth_occupancy has invalid shape or values")
    if quality.ndim != 3 or quality.shape[:2] != (count, n_nodes) or not np.all(np.isfinite(quality)):
        errors.append("node_quality has invalid shape or values")
    availability = np.asarray(quality_availability)
    if quality.ndim == 3 and (availability.shape != (quality.shape[2],) or availability.dtype != np.bool_):
        errors.append("quality_availability must be one boolean per feature")
    return errors
