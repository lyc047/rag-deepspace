from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage9_6_process_protocol import SenderEndpoint, bits_to_wire
from spectrum_semcom.stage9_8_compact_transport import (
    REGISTERED_COMPACT_UDP_PAYLOAD_BITS,
    corrupt_compact_protocol_payload,
    decode_compact_udp_datagram,
    encode_compact_udp_datagram,
    escape_roundtrip_self_check,
)


def test_compact_normal_update_roundtrip_and_width():
    frame = SenderEndpoint().make_update(symbol=1, update_epoch=4)
    datagram = encode_compact_udp_datagram("update", frame["wire"])
    decoded = decode_compact_udp_datagram(datagram)
    assert decoded.wire == frame["wire"]
    assert decoded.udp_payload_bits == REGISTERED_COMPACT_UDP_PAYLOAD_BITS["update"]


def test_compact_escape_width_roundtrip():
    assert escape_roundtrip_self_check()
    wire = bits_to_wire(np.zeros(56, dtype=np.uint8))
    assert decode_compact_udp_datagram(
        encode_compact_udp_datagram("update", wire)
    ).protocol_bits == 56


def test_compact_rejects_invalid_length_and_preserves_inner_corruption():
    frame = SenderEndpoint().make_update(symbol=1, update_epoch=4)
    datagram = encode_compact_udp_datagram("update", frame["wire"])
    corrupted = decode_compact_udp_datagram(
        corrupt_compact_protocol_payload(datagram)
    )
    assert corrupted.wire != frame["wire"]
    with pytest.raises(ValueError):
        decode_compact_udp_datagram(datagram[:-1])
