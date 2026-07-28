import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_temporal_hazard import (
    build_temporal_hazard_dataset,
    temporal_hazard_feature_names,
)


def test_temporal_hazard_features_are_causal_and_reset_at_group_boundary() -> None:
    powers = np.asarray(
        [
            [-100.0, -90.0, -80.0],
            [-99.0, -91.0, -80.0],
            [-70.0, -80.0, -100.0],
            [-71.0, -79.0, -100.0],
        ]
    )
    query = SpectrumTaskQuery(1)
    states = [
        build_task_state(row, (query,), epsilon_db=0.2)
        for row in powers
    ]
    codebook = fit_greedy_task_codebook(states)
    result = build_temporal_hazard_dataset(
        powers,
        states,
        np.asarray(
            [
                "2023-01-01T00:00:00",
                "2023-01-01T00:00:10",
                "2023-01-01T01:00:00",
                "2023-01-01T01:00:10",
            ]
        ),
        np.asarray(["a", "a", "b", "b"]),
        codebook,
    )
    assert result["source_indices"].tolist() == [0, 2]
    names = result["feature_names"]
    history = names.index("history_available")
    change = names.index("past_change_mean_db")
    assert np.all(result["features"][:, history] == 0.0)
    assert np.all(result["features"][:, change] == 0.0)


def test_temporal_hazard_label_is_next_scene_reuse_violation() -> None:
    powers = np.asarray(
        [
            [-100.0, -90.0, -80.0],
            [-80.0, -90.0, -100.0],
        ]
    )
    query = SpectrumTaskQuery(1)
    states = [
        build_task_state(row, (query,), epsilon_db=0.2)
        for row in powers
    ]
    codebook = fit_greedy_task_codebook(states)
    result = build_temporal_hazard_dataset(
        powers,
        states,
        np.asarray(
            ["2023-01-01T00:00:00", "2023-01-01T00:00:10"]
        ),
        np.asarray(["a", "a"]),
        codebook,
    )
    assert result["labels"].tolist() == [1]
    assert result["next_regret_db"][0] > 0.2


def test_temporal_hazard_feature_schema_scales_with_queries() -> None:
    assert len(temporal_hazard_feature_names(3)) == 19
