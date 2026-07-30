import numpy as np

from spectrum_semcom.stage7_lightweight_models import (
    expand_discrete_survival_rows,
    fit_discrete_survival_logistic,
    model_matrix,
)


def test_model_matrix_uses_registered_features_without_site_identity() -> None:
    dataset = {
        "features": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "feature_names": ("power_mean_dbm", "current_codeword_id"),
        "previous_symbol": np.asarray([-1, 2]),
        "action_dwell_scenes": np.asarray([1, 5]),
    }
    matrix, numeric_count, names = model_matrix(
        dataset,
        numeric_names=("power_mean_dbm",),
        categorical_names=(
            "current_codeword_id",
            "previous_symbol",
            "action_dwell_bin",
        ),
    )
    assert numeric_count == 1
    assert names == (
        "power_mean_dbm",
        "current_codeword_id",
        "previous_symbol",
        "action_dwell_bin",
    )
    assert matrix.shape == (2, 4)


def test_survival_expansion_stops_after_first_event() -> None:
    rows, labels, sources = expand_discrete_survival_rows(
        np.asarray([[1.0], [2.0], [3.0]]),
        np.asarray([2, -1, 7]),
        maximum_horizon=5,
    )
    assert rows.shape == (12, 2)
    assert labels.tolist() == [0, 1] + [0] * 10
    assert sources.tolist() == [0, 0] + [1] * 5 + [2] * 5
    assert rows[:2, -1].tolist() == [1.0, 2.0]


def test_discrete_survival_prediction_is_monotonic_by_horizon() -> None:
    features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    model = fit_discrete_survival_logistic(
        features,
        np.asarray([-1, 3, -1, 1, 2, 4]),
        numeric_count=1,
        maximum_horizon=5,
        regularization_c=0.3,
        class_weight=None,
        maximum_iterations=500,
        random_seed=7,
    )
    risk = model.predict_risk(features, (1, 3, 5))
    assert risk.shape == (6, 3)
    assert np.all(risk[:, 1] >= risk[:, 0])
    assert np.all(risk[:, 2] >= risk[:, 1])
