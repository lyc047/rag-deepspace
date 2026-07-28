"""Fixed-width context probes for detecting silent receiver resets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_REQUEST_MAGIC = 0xEA
_RESPONSE_MAGIC = 0xEB
_VERSION = 1
_FRAME_BITS = 32


@dataclass(frozen=True)
class ContextProbeRequest:
    node_id: int
    codebook_epoch: int
    expected_update_epoch: int


@dataclass(frozen=True)
class ContextProbeResponse:
    node_id: int
    codebook_epoch: int
    current_update_epoch: int


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= int(value) < 2**width:
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


def _encode_frame(
    *,
    magic: int,
    node_id: int,
    codebook_epoch: int,
    update_epoch: int,
) -> np.ndarray:
    if (
        not 0 <= int(node_id) <= 63
        or not 0 <= int(codebook_epoch) <= 255
        or not 0 <= int(update_epoch) <= 255
    ):
        raise ValueError("context probe field is outside the fixed schema")
    return np.concatenate(
        [
            _uint_to_bits(magic, 8),
            _uint_to_bits(_VERSION, 2),
            _uint_to_bits(node_id, 6),
            _uint_to_bits(codebook_epoch, 8),
            _uint_to_bits(update_epoch, 8),
        ]
    )


def _decode_frame(bits: np.ndarray, *, expected_magic: int) -> tuple[int, int, int]:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size != _FRAME_BITS:
        raise ValueError("context probe must contain exactly 32 bits")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("invalid context probe bits")
    magic = _bits_to_uint(data[:8])
    version = _bits_to_uint(data[8:10])
    if magic != expected_magic or version != _VERSION:
        raise ValueError("invalid context probe identity")
    return (
        _bits_to_uint(data[10:16]),
        _bits_to_uint(data[16:24]),
        _bits_to_uint(data[24:32]),
    )


def encode_context_probe_request(
    *,
    node_id: int,
    codebook_epoch: int,
    expected_update_epoch: int,
) -> np.ndarray:
    return _encode_frame(
        magic=_REQUEST_MAGIC,
        node_id=node_id,
        codebook_epoch=codebook_epoch,
        update_epoch=expected_update_epoch,
    )


def decode_context_probe_request(bits: np.ndarray) -> ContextProbeRequest:
    node_id, codebook_epoch, update_epoch = _decode_frame(
        bits, expected_magic=_REQUEST_MAGIC
    )
    return ContextProbeRequest(node_id, codebook_epoch, update_epoch)


def encode_context_probe_response(
    *,
    node_id: int,
    codebook_epoch: int,
    current_update_epoch: int,
) -> np.ndarray:
    return _encode_frame(
        magic=_RESPONSE_MAGIC,
        node_id=node_id,
        codebook_epoch=codebook_epoch,
        update_epoch=current_update_epoch,
    )


def decode_context_probe_response(bits: np.ndarray) -> ContextProbeResponse:
    node_id, codebook_epoch, update_epoch = _decode_frame(
        bits, expected_magic=_RESPONSE_MAGIC
    )
    return ContextProbeResponse(node_id, codebook_epoch, update_epoch)

