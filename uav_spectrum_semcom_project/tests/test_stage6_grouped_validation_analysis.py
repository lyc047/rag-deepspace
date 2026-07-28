from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_stage6_grouped_validation import (  # noqa: E402
    _bootstrap_plan,
    _paired_savings,
    _row_for,
)


def _metric_row(value: float, scene_count: int = 10) -> dict:
    return {
        "evaluated_scene_count": scene_count,
        "trajectory_metrics": [
            {
                "resource_equivalent_bits_per_scene": value,
                "clean_rate": 0.95,
            },
            {
                "resource_equivalent_bits_per_scene": value,
                "clean_rate": 0.95,
            },
        ],
    }


def test_hierarchical_paired_savings_preserves_exact_ratio() -> None:
    semantic = [_metric_row(5.0, 10), _metric_row(10.0, 20)]
    exact = [_metric_row(10.0, 10), _metric_row(20.0, 20)]
    groups, trajectories = _bootstrap_plan(2, 2, 200, 17)
    result = _paired_savings(
        semantic,
        exact,
        "resource_equivalent_bits_per_scene",
        groups,
        trajectories,
        0.025,
    )
    assert result["point_estimate"] == 50.0
    assert result["confidence_interval_lower"] == 50.0
    assert result["confidence_interval_upper"] == 50.0


def test_bootstrap_plan_is_deterministic() -> None:
    left = _bootstrap_plan(3, 5, 20, 99)
    right = _bootstrap_plan(3, 5, 20, 99)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_row_lookup_respects_frozen_workpoint_identity() -> None:
    semantic = {
        "candidate_id": "n8_controller_01",
        "deployment_mode": "online_install_empty_context",
        "requested_session_scene_count": 20,
    }
    exact = {
        "task_packet_open_loop_attempts": 2,
        "requested_session_scene_count": 20,
    }
    group = {
        "semantic_workpoints": [semantic],
        "exact_workpoints": [exact],
    }
    assert (
        _row_for(
            group,
            method="semantic",
            scene_count=20,
            candidate_id="n8_controller_01",
            deployment_mode="online_install_empty_context",
        )
        is semantic
    )
    assert (
        _row_for(
            group,
            method="exact",
            scene_count=20,
            attempts=2,
        )
        is exact
    )
