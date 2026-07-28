"""Compact cumulative ACK codec for Stage-5 state confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_MAGIC = 0xA7
_VERSION = 1
_ACK_BITS = 24


@dataclass(frozen=True)
class CumulativeAck:
    node_id: int
    epoch: int


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit ACK field")
    return np.asarray(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


def encode_cumulative_ack(*, node_id: int, epoch: int) -> np.ndarray:
    if not 0 <= int(node_id) <= 63 or not 0 <= int(epoch) <= 255:
        raise ValueError("invalid cumulative ACK field")
    return np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 2),
            _uint_to_bits(node_id, 6),
            _uint_to_bits(epoch, 8),
        ]
    )


def decode_cumulative_ack(bits: np.ndarray) -> CumulativeAck:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size != _ACK_BITS or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid cumulative ACK bitstream")
    if _bits_to_uint(data[:8]) != _MAGIC or _bits_to_uint(data[8:10]) != _VERSION:
        raise ValueError("invalid cumulative ACK identity")
    return CumulativeAck(
        node_id=_bits_to_uint(data[10:16]),
        epoch=_bits_to_uint(data[16:24]),
    )


def cumulative_ack_payload_bits() -> int:
    return _ACK_BITS

