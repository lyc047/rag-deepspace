"""Stage-9.7 UDP application framing for the frozen semantic protocol."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

from spectrum_semcom.stage9_6_process_protocol import (
    REGISTERED_WIDTHS,
    wire_to_bits,
)


TRANSPORT_HEADER_BITS = 32
TRANSPORT_MAGIC = 0xD7
TRANSPORT_VERSION = 1
KIND_TO_ID = {
    "reset_nack": 1,
    "activation": 2,
    "full_install": 3,
    "update": 4,
}
ID_TO_KIND = {value: key for key, value in KIND_TO_ID.items()}
REGISTERED_UDP_PAYLOAD_BITS = {
    kind: TRANSPORT_HEADER_BITS + 8 * ((width + 7) // 8)
    for kind, width in REGISTERED_WIDTHS.items()
}


@dataclass(frozen=True)
class DecodedUdpDatagram:
    kind: str
    wire: dict[str, Any]
    protocol_bits: int
    udp_payload_bits: int
    sha256: str


def encode_udp_datagram(kind: str, wire: dict[str, Any]) -> bytes:
    if kind not in KIND_TO_ID:
        raise ValueError("unknown UDP transport frame kind")
    bits = wire_to_bits(wire)
    if int(bits.size) != int(REGISTERED_WIDTHS[kind]):
        raise ValueError("protocol frame width does not match registered kind")
    payload = bytes.fromhex(str(wire["payload_hex"]))
    header = struct.pack(
        "!BBH",
        TRANSPORT_MAGIC,
        (TRANSPORT_VERSION << 4) | KIND_TO_ID[kind],
        int(bits.size),
    )
    datagram = header + payload
    if 8 * len(datagram) != REGISTERED_UDP_PAYLOAD_BITS[kind]:
        raise AssertionError("UDP payload width invariant failed")
    return datagram


def decode_udp_datagram(datagram: bytes) -> DecodedUdpDatagram:
    raw = bytes(datagram)
    if len(raw) < 5:
        raise ValueError("truncated UDP application datagram")
    magic, version_kind, bit_length = struct.unpack("!BBH", raw[:4])
    version = version_kind >> 4
    kind_id = version_kind & 0x0F
    if magic != TRANSPORT_MAGIC or version != TRANSPORT_VERSION:
        raise ValueError("invalid UDP application identity")
    if kind_id not in ID_TO_KIND:
        raise ValueError("unknown UDP application frame kind")
    kind = ID_TO_KIND[kind_id]
    wire = {"bit_length": int(bit_length), "payload_hex": raw[4:].hex()}
    bits = wire_to_bits(wire)
    if int(bits.size) != int(REGISTERED_WIDTHS[kind]):
        raise ValueError("UDP datagram protocol width mismatch")
    if 8 * len(raw) != int(REGISTERED_UDP_PAYLOAD_BITS[kind]):
        raise ValueError("UDP application payload width mismatch")
    return DecodedUdpDatagram(
        kind=kind,
        wire=wire,
        protocol_bits=int(bits.size),
        udp_payload_bits=8 * len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def corrupt_protocol_payload(datagram: bytes) -> bytes:
    raw = bytearray(datagram)
    if len(raw) < 5:
        raise ValueError("cannot corrupt truncated datagram")
    raw[4] ^= 0x80
    return bytes(raw)
