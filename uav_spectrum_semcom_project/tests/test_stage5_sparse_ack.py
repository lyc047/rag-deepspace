import numpy as np
import pytest

from spectrum_semcom.stage5_sparse_ack import (
    ack_request_payload_bits,
    decode_ack_request,
    deterministic_piggyback_opportunity,
    encode_ack_request,
    piggyback_epoch_bits,
)


def test_ack_request_codec_and_declared_field_sizes() -> None:
    encoded = encode_ack_request()
    assert np.array_equal(encoded, np.asarray([1], dtype=np.uint8))
    assert decode_ack_request(encoded) is True
    assert decode_ack_request(np.asarray([0], dtype=np.uint8)) is False
    assert ack_request_payload_bits() == 1
    assert piggyback_epoch_bits() == 8


def test_ack_request_rejects_invalid_bitstreams() -> None:
    with pytest.raises(ValueError):
        decode_ack_request(np.asarray([1, 0], dtype=np.uint8))
    with pytest.raises(ValueError):
        decode_ack_request(np.asarray([2], dtype=np.uint8))


def test_piggyback_opportunities_are_stable_and_nested() -> None:
    scene_ids = [f"scene-{index}" for index in range(200)]
    p25 = {
        scene_id
        for scene_id in scene_ids
        if deterministic_piggyback_opportunity(scene_id, 0.25)
    }
    p50 = {
        scene_id
        for scene_id in scene_ids
        if deterministic_piggyback_opportunity(scene_id, 0.5)
    }
    p100 = {
        scene_id
        for scene_id in scene_ids
        if deterministic_piggyback_opportunity(scene_id, 1.0)
    }
    assert p25 < p50 < p100
    assert p100 == set(scene_ids)
    assert deterministic_piggyback_opportunity("scene-7", 0.5) == (
        deterministic_piggyback_opportunity("scene-7", 0.5)
    )


def test_piggyback_probability_is_validated() -> None:
    with pytest.raises(ValueError):
        deterministic_piggyback_opportunity("scene", -0.1)
    with pytest.raises(ValueError):
        deterministic_piggyback_opportunity("scene", 1.1)
