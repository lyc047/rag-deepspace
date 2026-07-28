import numpy as np
import pytest

from spectrum_semcom.gate_a_data import parse_namespaced_frame_id, pool_probability_mask, validate_cache_arrays, validate_multinode_cache_arrays


def test_namespaced_frame_id_keeps_original_split_identity() -> None:
    assert parse_namespaced_frame_id("train:0001") == ("train", "0001")
    assert parse_namespaced_frame_id("val:0001") == ("val", "0001")
    with pytest.raises(ValueError, match="namespace"):
        parse_namespaced_frame_id("0001")


def test_probability_pool_matches_declared_axis() -> None:
    mask = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert np.allclose(pool_probability_mask(mask, 2, "y"), [0.0, 1.0])
    assert np.allclose(pool_probability_mask(mask.T, 2, "x"), [0.0, 1.0])


def test_cache_validation_rejects_out_of_range_occupancy() -> None:
    errors = validate_cache_arrays(["scene"], np.asarray([[1.2, 0.0]]), np.zeros((1, 2)), np.asarray([0.5]), 2)
    assert any("base_occupancy" in error for error in errors)


def test_multinode_cache_validation_checks_all_axes() -> None:
    errors = validate_multinode_cache_arrays(["s"], np.zeros((1,2,4)), np.zeros((1,4)), np.zeros((1,2,3)), np.asarray([True,False,True]), 2, 4)
    assert errors == []
