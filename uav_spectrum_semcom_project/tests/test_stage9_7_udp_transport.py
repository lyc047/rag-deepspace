from __future__ import annotations

import pytest

from spectrum_semcom.stage9_6_process_protocol import SenderEndpoint
from spectrum_semcom.stage9_7_udp_transport import (
    REGISTERED_UDP_PAYLOAD_BITS,
    corrupt_protocol_payload,
    decode_udp_datagram,
    encode_udp_datagram,
)


def _frames():
    sender = SenderEndpoint()
    update = sender.make_update(symbol=1, update_epoch=4)
    return [("update", update["wire"])]


def test_udp_application_frame_roundtrip_and_width():
    for kind, wire in _frames():
        datagram = encode_udp_datagram(kind, wire)
        decoded = decode_udp_datagram(datagram)
        assert decoded.kind == kind
        assert decoded.wire == wire
        assert decoded.udp_payload_bits == REGISTERED_UDP_PAYLOAD_BITS[kind]


def test_udp_application_frame_preserves_corruption_for_endpoint_and_rejects_truncation():
    kind, wire = _frames()[0]
    datagram = encode_udp_datagram(kind, wire)
    corrupted = decode_udp_datagram(corrupt_protocol_payload(datagram))
    assert corrupted.kind == kind
    assert corrupted.wire != wire
    with pytest.raises(ValueError):
        decode_udp_datagram(datagram[:-1])
