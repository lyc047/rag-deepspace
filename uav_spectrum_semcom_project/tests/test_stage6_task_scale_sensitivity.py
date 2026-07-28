from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectrum_semcom.stage6_task_scale_sensitivity import (
    amortized_bits_per_scene,
    codebook_install_break_even_scenes,
    codeword_usage_metrics,
    registered_grid_points,
    rounded_demands,
    validate_complete_query_lattice,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (
            PROJECT_DIR
            / "configs"
            / "stage6_task_scale_sensitivity_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_registered_grid_is_complete_and_unique() -> None:
    protocol = _protocol()
    validate_complete_query_lattice(protocol)
    points = registered_grid_points(protocol)
    assert len(points) == 4 * 5 * 7
    assert len({value.grid_point_id for value in points}) == len(points)


def test_rounding_produces_expected_demands() -> None:
    assert rounded_demands(8, (0.25, 0.5, 0.75)) == (2, 4, 6)
    assert rounded_demands(64, (0.25, 0.5, 0.75)) == (16, 32, 48)
    with pytest.raises(ValueError):
        rounded_demands(2, (0.25, 0.5))


def test_codeword_usage_keeps_escape_separate() -> None:
    value = codeword_usage_metrics(
        (0, 0, 1, 3), codeword_count=3, escape_symbol=3
    )
    assert value["escape_rate"] == pytest.approx(0.25)
    assert value["used_codeword_count"] == 2
    assert value["codeword_utilization_rate"] == pytest.approx(2.0 / 3.0)
    assert value["effective_codeword_perplexity"] > 1.0


def test_amortization_and_break_even_are_consistent() -> None:
    assert amortized_bits_per_scene(40.0, 800, 800) == pytest.approx(41.0)
    assert codebook_install_break_even_scenes(40.0, 50.0, 800) == 80.0
    assert codebook_install_break_even_scenes(50.0, 50.0, 800) is None
