import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage7_causal_baselines import (
    build_causal_task_risk_dataset,
    dwell_bin,
    quantile_lookup_probabilities,
    smoothed_lookup_probabilities,
)


def _dataset(last_row: np.ndarray | None = None) -> tuple:
    powers = np.asarray(
        [
            [-100.0, -90.0, -80.0],
            [-99.0, -91.0, -80.0],
            [-98.0, -92.0, -80.0],
            [-80.0, -90.0, -100.0],
            [-81.0, -90.0, -99.0],
            [-82.0, -90.0, -98.0],
            [-83.0, -90.0, -97.0],
        ]
    )
    if last_row is not None:
        powers[-1] = last_row
    states = [
        build_task_state(row, (SpectrumTaskQuery(1),), epsilon_db=0.2)
        for row in powers
    ]
    codebook = fit_greedy_task_codebook(states)
    return powers, states, codebook


def test_causal_features_do_not_change_when_only_future_power_changes() -> None:
    powers, states, codebook = _dataset()
    first = build_causal_task_risk_dataset(
        powers,
        states,
        ["a"] * len(states),
        codebook,
        horizons=(1, 3),
    )
    changed_power, changed_states, _ = _dataset(
        np.asarray([-10.0, -20.0, -30.0])
    )
    second = build_causal_task_risk_dataset(
        changed_power,
        changed_states,
        ["a"] * len(changed_states),
        codebook,
        horizons=(1, 3),
    )
    assert np.array_equal(first["source_indices"], second["source_indices"])
    assert np.allclose(first["features"][0], second["features"][0])


def test_causal_dataset_keeps_full_horizon_inside_group() -> None:
    powers, states, codebook = _dataset()
    result = build_causal_task_risk_dataset(
        powers,
        states,
        ["a", "a", "a", "a", "b", "b", "b"],
        codebook,
        horizons=(1, 2),
    )
    assert result["groups"].tolist() == ["a", "a", "b"]
    assert result["source_indices"].tolist() == [0, 1, 4]
    assert result["task_time_to_failure"].shape == (3,)


def test_lookup_baselines_use_training_prevalence_for_unseen_keys() -> None:
    scores = smoothed_lookup_probabilities(
        ["a", "a", "b"],
        np.asarray([1, 0, 0]),
        ["a", "missing"],
        prior_strength=2.0,
    )
    assert 0.0 < scores[0] < 1.0
    assert scores[1] == np.mean([1, 0, 0])
    continuous = quantile_lookup_probabilities(
        np.arange(10),
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        np.asarray([1.0, 8.0]),
        bin_count=4,
    )
    assert continuous[1] > continuous[0]
    assert dwell_bin(np.asarray([1, 2, 3, 5, 9])).tolist() == [0, 1, 2, 3, 4]
