import numpy as np
import pytest

from spectrum_semcom.stage6_context_heartbeat import (
    decode_context_probe_request,
    decode_context_probe_response,
    encode_context_probe_request,
    encode_context_probe_response,
)


def test_context_probe_request_round_trip_is_exactly_32_bits() -> None:
    bits = encode_context_probe_request(
        node_id=17,
        codebook_epoch=201,
        expected_update_epoch=99,
    )
    assert bits.size == 32
    decoded = decode_context_probe_request(bits)
    assert decoded.node_id == 17
    assert decoded.codebook_epoch == 201
    assert decoded.expected_update_epoch == 99


def test_context_probe_response_round_trip_is_exactly_32_bits() -> None:
    bits = encode_context_probe_response(
        node_id=17,
        codebook_epoch=201,
        current_update_epoch=100,
    )
    assert bits.size == 32
    decoded = decode_context_probe_response(bits)
    assert decoded.node_id == 17
    assert decoded.codebook_epoch == 201
    assert decoded.current_update_epoch == 100


def test_probe_request_cannot_be_decoded_as_response() -> None:
    bits = encode_context_probe_request(
        node_id=1,
        codebook_epoch=1,
        expected_update_epoch=1,
    )
    with pytest.raises(ValueError, match="identity"):
        decode_context_probe_response(bits)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(31, dtype=np.uint8),
        np.zeros(33, dtype=np.uint8),
        np.full(32, 2, dtype=np.uint8),
    ],
)
def test_context_probe_rejects_invalid_length_or_bits(
    invalid: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        decode_context_probe_request(invalid)


def test_context_probe_rejects_node_outside_six_bits() -> None:
    with pytest.raises(ValueError):
        encode_context_probe_request(
            node_id=64,
            codebook_epoch=1,
            expected_update_epoch=1,
        )
