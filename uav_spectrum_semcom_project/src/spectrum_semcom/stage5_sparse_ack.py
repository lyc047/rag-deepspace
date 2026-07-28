"""Primitives for Stage-5 on-demand and piggybacked state confirmation."""

from __future__ import annotations

import hashlib

import numpy as np


_REQUEST_BITS = 1
_PIGGYBACK_EPOCH_BITS = 8


def encode_ack_request(request_latest_epoch: bool = True) -> np.ndarray:
    """Encode the one-bit request carried in a typed control packet."""
    return np.asarray([int(bool(request_latest_epoch))], dtype=np.uint8)


def decode_ack_request(bits: np.ndarray) -> bool:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size != _REQUEST_BITS or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid ACK-request bitstream")
    return bool(data[0])


def ack_request_payload_bits() -> int:
    return _REQUEST_BITS


def piggyback_epoch_bits() -> int:
    return _PIGGYBACK_EPOCH_BITS


def deterministic_piggyback_opportunity(
    scene_id: str,
    probability: float,
    *,
    salt: str = "stage5-sparse-ack-v1",
) -> bool:
    """Return a stable nested opportunity sample for protocol boundary tests."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    digest = hashlib.sha256(f"{salt}:{scene_id}".encode("utf-8")).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(2**64)
    return bool(uniform < float(probability))
