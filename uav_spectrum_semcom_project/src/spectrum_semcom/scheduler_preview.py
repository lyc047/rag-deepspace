"""Auditable sparse residual preview used only for C2 scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from zlib import crc32

import numpy as np


_MAGIC = 0xB6
_VERSION = 1


def _uint_bits(value: int, width: int) -> np.ndarray:
    if width < 1 or not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit scheduler preview field")
    return np.asarray([(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1): value = (value << 1) | int(bit)
    return value


def _quantize(value: float, low: float, high: float, width: int) -> int:
    return int(np.rint((np.clip(float(value), low, high) - low) / (high - low) * (2**width - 1)))


def _dequantize(code: int, low: float, high: float, width: int) -> float:
    return float(low + int(code) / (2**width - 1) * (high - low))


@dataclass(frozen=True)
class SchedulerPreview:
    node_id: int
    scene_id: str
    hard_occupancy: np.ndarray
    residual_indices: tuple[int, ...]
    residual_magnitudes: tuple[float, ...]
    sensing_snr_db: float
    prediction_confidence: float
    clipping_ratio: float
    age_s: float
    magnitude_bits: int = 3

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.hard_occupancy, dtype=np.float64).reshape(-1)
        if not 0 <= int(self.node_id) <= 4095 or not self.scene_id or not 2 <= occupancy.size <= 255:
            raise ValueError("invalid scheduler preview identity or occupancy width")
        if np.any((occupancy != 0) & (occupancy != 1)):
            raise ValueError("scheduler preview fusion occupancy must be one bit")
        if len(self.residual_indices) != len(self.residual_magnitudes) or len(self.residual_indices) > 7:
            raise ValueError("invalid residual sketch length")
        if len(set(self.residual_indices)) != len(self.residual_indices) or any(not 0 <= index < occupancy.size for index in self.residual_indices):
            raise ValueError("residual indices must be unique and in range")
        if any(not np.isfinite(value) or not 0 <= value <= .5 for value in self.residual_magnitudes):
            raise ValueError("residual magnitudes must lie in [0, 0.5]")
        if self.magnitude_bits not in (2, 3, 4):
            raise ValueError("residual magnitude bits must be 2, 3, or 4")
        if not np.isfinite(self.sensing_snr_db) or not 0 <= self.prediction_confidence <= 1 or not 0 <= self.clipping_ratio <= 1 or not 0 <= self.age_s <= 1.275:
            raise ValueError("invalid compact quality summary")
        object.__setattr__(self, "hard_occupancy", occupancy)

    @property
    def scheduler_report(self) -> np.ndarray:
        values = self.hard_occupancy.astype(np.float64, copy=True)
        for index, magnitude in zip(self.residual_indices, self.residual_magnitudes):
            values[index] += magnitude if values[index] == 0 else -magnitude
        return np.clip(values, 0, 1)


@dataclass(frozen=True)
class EncodedSchedulerPreview:
    bits: np.ndarray

    @property
    def application_bits(self) -> int:
        return int(self.bits.size)


def build_scheduler_preview(soft_occupancy: np.ndarray, node_id: int, scene_id: str, sensing_snr_db: float, prediction_confidence: float, clipping_ratio: float, age_s: float, top_k: int = 2, magnitude_bits: int = 3) -> SchedulerPreview:
    soft = np.asarray(soft_occupancy, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(soft)) or np.any((soft < 0) | (soft > 1)) or not 0 <= top_k <= min(7, soft.size):
        raise ValueError("invalid soft occupancy or top_k")
    hard = np.rint(soft).astype(np.float64)
    residual = np.abs(soft - hard)
    order = np.lexsort((np.arange(soft.size), -residual))[:top_k]
    levels = 2**magnitude_bits - 1
    magnitudes = tuple(float(np.rint(residual[index] / .5 * levels) / levels * .5) for index in order)
    return SchedulerPreview(node_id, scene_id, hard, tuple(int(index) for index in order), magnitudes, sensing_snr_db, prediction_confidence, clipping_ratio, age_s, magnitude_bits)


def encode_scheduler_preview(preview: SchedulerPreview) -> EncodedSchedulerPreview:
    width = int(ceil(log2(preview.hard_occupancy.size)))
    parts = [_uint_bits(_MAGIC, 8), _uint_bits(_VERSION, 3), _uint_bits(preview.node_id, 12), _uint_bits(crc32(preview.scene_id.encode("utf-8")) & 0xFFFFFFFF, 32), _uint_bits(preview.hard_occupancy.size, 8), _uint_bits(len(preview.residual_indices), 3), _uint_bits(preview.magnitude_bits - 2, 2), preview.hard_occupancy.astype(np.uint8), _uint_bits(_quantize(preview.sensing_snr_db, -16, 15.5, 6), 6), _uint_bits(_quantize(preview.prediction_confidence, 0, 1, 6), 6), _uint_bits(_quantize(preview.clipping_ratio, 0, 1, 5), 5), _uint_bits(_quantize(preview.age_s, 0, 1.275, 8), 8)]
    for index, magnitude in zip(preview.residual_indices, preview.residual_magnitudes):
        parts.extend((_uint_bits(index, width), _uint_bits(_quantize(magnitude, 0, .5, preview.magnitude_bits), preview.magnitude_bits)))
    return EncodedSchedulerPreview(np.concatenate(parts).astype(np.uint8))


def decode_scheduler_preview(bits: np.ndarray) -> SchedulerPreview:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1); offset = 0
    if np.any((data != 0) & (data != 1)):
        raise ValueError("scheduler preview bitstream must be binary")
    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size: raise ValueError("truncated scheduler preview")
        value = _bits_uint(data[offset:offset + width]); offset += width; return value
    if take(8) != _MAGIC or take(3) != _VERSION: raise ValueError("invalid scheduler preview header")
    node = take(12); tag = take(32); n_channels = take(8); count = take(3); magnitude_bits = take(2) + 2
    if n_channels < 2: raise ValueError("invalid scheduler preview channel count")
    hard = np.asarray([take(1) for _ in range(n_channels)], dtype=np.float64)
    snr = _dequantize(take(6), -16, 15.5, 6); confidence = _dequantize(take(6), 0, 1, 6); clipping = _dequantize(take(5), 0, 1, 5); age = _dequantize(take(8), 0, 1.275, 8)
    width = int(ceil(log2(n_channels))); indices=[]; magnitudes=[]
    for _ in range(count): indices.append(take(width)); magnitudes.append(_dequantize(take(magnitude_bits), 0, .5, magnitude_bits))
    if offset != data.size: raise ValueError("scheduler preview has trailing bits")
    return SchedulerPreview(node, f"tag:{tag}", hard, tuple(indices), tuple(magnitudes), snr, confidence, clipping, age, magnitude_bits)
