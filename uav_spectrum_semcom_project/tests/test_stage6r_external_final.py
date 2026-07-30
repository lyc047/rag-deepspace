import numpy as np

from spectrum_semcom.stage6r_external_final import (
    hierarchical_mean_ci,
    synthetic_six_hour_timestamps,
    task_queries,
)


def test_synthetic_timestamps_span_six_hours():
    values = synthetic_six_hour_timestamps(3)
    assert values[0].endswith("+00:00")
    assert "03:00:00" in values[1]
    assert "06:00:00" in values[2]


def test_task_queries_follow_frozen_ratios():
    assert [query.demand_channels for query in task_queries(
        16, [0.25, 0.5, 0.75]
    )] == [4, 8, 12]


def test_hierarchical_ci_respects_site_weights():
    values = np.asarray([[0.0, 0.0], [10.0, 10.0]])
    result = hierarchical_mean_ci(
        values,
        seed=1,
        replicates=200,
        confidence=0.95,
        site_weights=np.asarray([1.0, 3.0]),
    )
    assert result["mean"] == 7.5
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]
