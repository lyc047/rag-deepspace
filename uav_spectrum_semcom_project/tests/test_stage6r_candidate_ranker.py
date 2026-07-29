import numpy as np

from spectrum_semcom.stage6r_candidate_ranker import (
    candidate_feature_matrix,
    deterministic_training_subset,
    future_coverage_labels,
    pooled_context_features,
    select_ranked_codebook,
)


def test_candidate_features_have_fixed_width_and_no_future_input():
    power = np.arange(32, dtype=float).reshape(2, 16)
    context = pooled_context_features(power, output_bins=8)
    regrets = np.asarray([[0.0, 0.4], [0.1, 0.2]])
    features = candidate_feature_matrix(
        calibration_regrets=regrets,
        actions=[(0, 0, 0), (1, 1, 1)],
        action_maxima=[4, 4, 4],
        mean_optimal_actions=[0.5, 0.5, 0.5],
        context_features=context,
        epsilon_db=0.2,
    )
    assert features.shape == (2, 19)
    assert np.all(np.isfinite(features))


def test_labels_and_subset_are_deterministic():
    regrets = np.asarray(
        [[0.0, 0.3, 0.1, 0.5], [0.1, 0.4, 0.2, 0.0]]
    )
    labels = future_coverage_labels(regrets, epsilon_db=0.2)
    assert np.allclose(labels, [1.0, 0.0, 1.0, 0.5])
    first = deterministic_training_subset(
        calibration_coverage=labels,
        future_coverage=labels[::-1],
        maximum_rows=8,
    )
    second = deterministic_training_subset(
        calibration_coverage=labels,
        future_coverage=labels[::-1],
        maximum_rows=8,
    )
    assert np.array_equal(first, second)


def test_ranked_selection_returns_distinct_safe_candidates():
    regrets = np.asarray(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    selected = select_ranked_codebook(
        calibration_regrets=regrets,
        predicted_future_coverage=np.asarray([0.9, 0.8, 0.1]),
        actions=[(0,), (1,), (2,)],
        epsilon_db=0.2,
        codeword_count=2,
    )
    assert len(set(selected.tolist())) == 2
    covered = np.any(regrets[:, selected] <= 0.2, axis=1)
    assert np.all(covered)

