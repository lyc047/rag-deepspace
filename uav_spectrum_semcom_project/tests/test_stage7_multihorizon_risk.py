import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
)
from spectrum_semcom.stage7_multihorizon_risk import (
    build_multihorizon_risk_labels,
    perfect_reset_oracle_trigger_mask,
)


def _states() -> list:
    powers = np.asarray(
        [
            [-100.0, -90.0, -80.0],
            [-100.0, -90.0, -80.0],
            [-80.0, -90.0, -100.0],
            [-80.0, -90.0, -100.0],
            [-100.0, -90.0, -80.0],
            [-80.0, -90.0, -100.0],
        ]
    )
    return [
        build_task_state(row, (SpectrumTaskQuery(1),), epsilon_db=0.2)
        for row in powers
    ]


def test_multihorizon_labels_detect_future_task_and_context_failures() -> None:
    states = _states()
    actions = [state.optimal_actions for state in states]
    context = np.asarray([False, False, True, False, False, False])
    result = build_multihorizon_risk_labels(
        states,
        actions,
        ["a", "a", "a", "a", "b", "b"],
        horizons=(1, 3),
        context_failure_events=context,
    )
    assert result["observed"][0].tolist() == [True, True]
    assert result["task_failure"][0].tolist() == [0, 1]
    assert result["context_failure"][0].tolist() == [0, 1]
    assert result["any_time_to_failure"][0] == 2
    assert result["task_time_to_failure"][0] == 2
    assert result["context_time_to_failure"][0] == 2


def test_multihorizon_labels_never_cross_group_boundaries() -> None:
    states = _states()
    actions = [state.optimal_actions for state in states]
    result = build_multihorizon_risk_labels(
        states,
        actions,
        ["a", "a", "a", "a", "b", "b"],
        horizons=(1, 3),
    )
    assert result["observed"][3].tolist() == [False, False]
    assert result["task_failure"][3].tolist() == [0, 0]
    assert result["observed"][4].tolist() == [True, False]


def test_multihorizon_labels_reject_unsorted_or_censored_schema() -> None:
    states = _states()
    actions = [state.optimal_actions for state in states]
    with pytest.raises(ValueError, match="horizons"):
        build_multihorizon_risk_labels(
            states, actions, ["a"] * len(states), horizons=(3, 1)
        )
    with pytest.raises(ValueError, match="context"):
        build_multihorizon_risk_labels(
            states,
            actions,
            ["a"] * len(states),
            context_failure_events=[False],
        )


def test_perfect_reset_oracle_uses_only_registered_reset_stream() -> None:
    random = np.asarray(
        [[0.1, 0.9], [0.8, 0.2], [0.4, 0.7]], dtype=np.float64
    )
    assert perfect_reset_oracle_trigger_mask(
        random, reset_probability=0.5, reset_stream_index=0
    ).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        perfect_reset_oracle_trigger_mask(
            random, reset_probability=1.1, reset_stream_index=0
        )
