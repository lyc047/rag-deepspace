import numpy as np
import pytest

from spectrum_semcom.stage5_cumulative_ack import (
    cumulative_ack_payload_bits,
    decode_cumulative_ack,
    encode_cumulative_ack,
)


def test_cumulative_ack_codec_round_trip() -> None:
    bits = encode_cumulative_ack(node_id=17, epoch=255)
    assert bits.size == 24
    assert cumulative_ack_payload_bits() == 24
    decoded = decode_cumulative_ack(bits)
    assert decoded.node_id == 17
    assert decoded.epoch == 255


def test_cumulative_ack_codec_fails_closed() -> None:
    bits = encode_cumulative_ack(node_id=0, epoch=0)
    damaged = bits.copy()
    damaged[0] ^= 1
    with pytest.raises(ValueError):
        decode_cumulative_ack(damaged)
    with pytest.raises(ValueError):
        encode_cumulative_ack(node_id=64, epoch=0)
    with pytest.raises(ValueError):
        decode_cumulative_ack(np.zeros(23, dtype=np.uint8))

