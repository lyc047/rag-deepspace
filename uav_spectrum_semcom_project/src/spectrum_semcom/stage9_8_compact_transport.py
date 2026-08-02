"""Stage-9.8 fixed-schema 8-bit UDP framing."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from spectrum_semcom.stage9_6_process_protocol import bits_to_wire, wire_to_bits
from spectrum_semcom.stage9_7_udp_transport import DecodedUdpDatagram


COMPACT_HEADER_BITS = 8
COMPACT_MAGIC = 0xA
COMPACT_VERSION = 1
KIND_TO_ID = {
    "reset_nack": 0,
    "activation": 1,
    "full_install": 2,
    "update": 3,
}
ID_TO_KIND = {value: key for key, value in KIND_TO_ID.items()}
ALLOWED_PROTOCOL_BITS = {
    "reset_nack": (32,),
    "activation": (80,),
    "full_install": (419,),
    "update": (42, 56),
}
REGISTERED_COMPACT_UDP_PAYLOAD_BITS = {
    "reset_nack": 40,
    "activation": 88,
    "full_install": 432,
    "update": 56,
}


def compact_payload_bits(kind: str, protocol_bits: int) -> int:
    if kind not in ALLOWED_PROTOCOL_BITS or int(protocol_bits) not in ALLOWED_PROTOCOL_BITS[kind]:
        raise ValueError("protocol length is outside compact schema")
    return COMPACT_HEADER_BITS + 8 * ((int(protocol_bits) + 7) // 8)


def encode_compact_udp_datagram(kind: str, wire: dict[str, Any]) -> bytes:
    if kind not in KIND_TO_ID:
        raise ValueError("unknown compact UDP frame kind")
    bits = wire_to_bits(wire)
    if int(bits.size) not in ALLOWED_PROTOCOL_BITS[kind]:
        raise ValueError("protocol length is outside compact schema")
    header = bytes(
        [
            (COMPACT_MAGIC << 4)
            | (COMPACT_VERSION << 2)
            | KIND_TO_ID[kind]
        ]
    )
    datagram = header + bytes.fromhex(str(wire["payload_hex"]))
    if 8 * len(datagram) != compact_payload_bits(kind, int(bits.size)):
        raise AssertionError("compact UDP width invariant failed")
    return datagram


def decode_compact_udp_datagram(datagram: bytes) -> DecodedUdpDatagram:
    raw = bytes(datagram)
    if len(raw) < 2:
        raise ValueError("truncated compact UDP datagram")
    header = int(raw[0])
    magic = header >> 4
    version = (header >> 2) & 0x03
    kind_id = header & 0x03
    if magic != COMPACT_MAGIC or version != COMPACT_VERSION:
        raise ValueError("invalid compact UDP identity")
    kind = ID_TO_KIND[kind_id]
    payload_bytes = len(raw) - 1
    matches = tuple(
        width
        for width in ALLOWED_PROTOCOL_BITS[kind]
        if (int(width) + 7) // 8 == payload_bytes
    )
    if len(matches) != 1:
        raise ValueError("compact UDP length is invalid or ambiguous")
    protocol_bits = int(matches[0])
    wire = {"bit_length": protocol_bits, "payload_hex": raw[1:].hex()}
    bits = wire_to_bits(wire)
    if int(bits.size) != protocol_bits:
        raise ValueError("compact UDP protocol length mismatch")
    return DecodedUdpDatagram(
        kind=kind,
        wire=wire,
        protocol_bits=protocol_bits,
        udp_payload_bits=8 * len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def corrupt_compact_protocol_payload(datagram: bytes) -> bytes:
    raw = bytearray(datagram)
    if len(raw) < 2:
        raise ValueError("cannot corrupt truncated compact datagram")
    raw[1] ^= 0x80
    return bytes(raw)


def escape_roundtrip_self_check() -> bool:
    wire = bits_to_wire(np.zeros(56, dtype=np.uint8))
    decoded = decode_compact_udp_datagram(
        encode_compact_udp_datagram("update", wire)
    )
    return bool(decoded.kind == "update" and decoded.wire == wire)
