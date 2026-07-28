"""Real codec for one packet carrying optimal indices for multiple queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_MAGIC = 0xE6
_VERSION = 1
_MESSAGE_TYPE = 0
_MINIMUM_HEADER_BITS = 40


@dataclass(frozen=True)
class DecodedQueryBundle:
    node_id: int
    epoch: int
    target_starts: dict[int, int]


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if width < 0 or not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit requested width")
    return np.asarray(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


def query_bundle_payload_width(
    n_channels: int,
    demand_channels: tuple[int, ...] | list[int],
) -> int:
    demands = tuple(int(value) for value in demand_channels)
    if (
        n_channels < 1
        or not demands
        or len(set(demands)) != len(demands)
        or any(not 1 <= demand <= n_channels for demand in demands)
    ):
        raise ValueError("invalid channel count or query set")
    return int(
        sum(
            math.ceil(math.log2(n_channels - demand + 1))
            for demand in demands
        )
    )


def encode_query_bundle(
    target_starts: dict[int, int],
    *,
    n_channels: int,
    demand_order: tuple[int, ...] | list[int],
    node_id: int,
    epoch: int,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> np.ndarray:
    demands = tuple(int(value) for value in demand_order)
    query_bundle_payload_width(n_channels, demands)
    if (
        set(target_starts) != set(demands)
        or application_header_bits < _MINIMUM_HEADER_BITS
        or not 0 <= int(node_id) <= 255
        or not 0 <= int(epoch) <= 255
        or len(demands) > 7
    ):
        raise ValueError("invalid bundle header or query targets")
    payload_parts = []
    for demand in demands:
        candidates = int(n_channels) - demand + 1
        target = int(target_starts[demand])
        if not 0 <= target < candidates:
            raise ValueError("bundle target is outside its candidate set")
        width = int(math.ceil(math.log2(candidates)))
        payload_parts.append(_uint_to_bits(target, width))
    fixed = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_MESSAGE_TYPE, 2),
            _uint_to_bits(int(node_id), 8),
            _uint_to_bits(len(demands), 3),
            _uint_to_bits(int(epoch), 8),
            np.zeros(8, dtype=np.uint8),
        ]
    )
    padding = np.zeros(int(application_header_bits) - fixed.size, dtype=np.uint8)
    payload = (
        np.concatenate(payload_parts)
        if payload_parts
        else np.zeros(0, dtype=np.uint8)
    )
    return np.concatenate([fixed, padding, payload])


def decode_query_bundle(
    bits: np.ndarray,
    *,
    n_channels: int,
    demand_order: tuple[int, ...] | list[int],
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> DecodedQueryBundle:
    demands = tuple(int(value) for value in demand_order)
    payload_width = query_bundle_payload_width(n_channels, demands)
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if (
        application_header_bits < _MINIMUM_HEADER_BITS
        or data.size != int(application_header_bits) + payload_width
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError("invalid query bundle bitstream")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated query bundle")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if (
        take(8) != _MAGIC
        or take(3) != _VERSION
        or take(2) != _MESSAGE_TYPE
    ):
        raise ValueError("invalid query bundle identity")
    node_id = take(8)
    query_count = take(3)
    epoch = take(8)
    reserved = take(8)
    if query_count != len(demands) or reserved != 0:
        raise ValueError("query bundle schema mismatch")
    offset = int(application_header_bits)
    targets = {}
    for demand in demands:
        candidates = int(n_channels) - demand + 1
        width = int(math.ceil(math.log2(candidates)))
        target = take(width)
        if target >= candidates:
            raise ValueError("decoded bundle target is outside candidate set")
        targets[demand] = target
    if offset != data.size:
        raise ValueError("query bundle has trailing bits")
    return DecodedQueryBundle(node_id, epoch, targets)

