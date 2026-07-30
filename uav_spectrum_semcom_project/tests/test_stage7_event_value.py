import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage7_event_value import (
    build_task_event_value_labels,
    maximum_true_run,
)


def test_maximum_true_run_counts_consecutive_events() -> None:
    assert maximum_true_run(np.asarray([0, 1, 1, 0, 1])) == 2
    assert maximum_true_run(np.zeros(4)) == 0


def test_event_value_labels_are_group_bounded_and_costed() -> None:
    powers = np.asarray(
        [
            [-100.0, -80.0],
            [-99.0, -81.0],
            [-80.0, -100.0],
            [-81.0, -99.0],
            [-82.0, -98.0],
            [-100.0, -80.0],
            [-99.0, -81.0],
            [-80.0, -100.0],
            [-81.0, -99.0],
        ]
    )
    states = [
        build_task_state(
            row, (SpectrumTaskQuery(1),), epsilon_db=0.2
        )
        for row in powers
    ]
    codebook = fit_greedy_task_codebook(states)
    result = build_task_event_value_labels(
        states,
        ["a"] * 5 + ["b"] * 4,
        codebook,
        horizon=3,
        persistent_run_lengths=(2, 3),
        refresh_offset=1,
        incremental_bits=66,
        clean_scene_value_bits=(32, 128),
    )
    assert result["source_indices"].tolist() == [0, 1, 5]
    assert result["groups"].tolist() == ["a", "a", "b"]
    assert result["incremental_bits"] == 66
    assert result["persistent_failure"]["2"].shape == (3,)
    assert result["net_positive_by_clean_scene_value_bits"]["128"].shape == (
        3,
    )
