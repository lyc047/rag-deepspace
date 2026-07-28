import numpy as np
import pytest

from spectrum_semcom.stage6r_context_adapter import (
    leave_one_context_novelty_threshold,
    nearest_context,
    normalized_feature_distance,
    robust_spectral_shape_features,
)


def test_shape_features_ignore_common_power_offset():
    values = np.asarray([[-90.0, -80.0, -70.0], [-85.0, -75.0, -65.0]])
    shifted = values + 17.0
    assert np.allclose(
        robust_spectral_shape_features(values),
        robust_spectral_shape_features(shifted),
    )


def test_nearest_context_and_training_only_threshold():
    prototypes = {
        "left": np.asarray([-1.0, 0.0, 1.0]),
        "near_left": np.asarray([-0.8, 0.0, 0.8]),
        "right": np.asarray([1.0, 0.0, -1.0]),
    }
    name, distance = nearest_context(
        np.asarray([-0.9, 0.0, 0.9]), prototypes
    )
    assert name == "left"
    assert distance == pytest.approx(
        normalized_feature_distance(
            np.asarray([-0.9, 0.0, 0.9]), prototypes["left"]
        )
    )
    assert leave_one_context_novelty_threshold(prototypes) > 0.0


def test_invalid_feature_inputs_fail_closed():
    with pytest.raises(ValueError):
        robust_spectral_shape_features(np.asarray([[1.0]]))
    with pytest.raises(ValueError):
        leave_one_context_novelty_threshold({"only": np.zeros(3)})

