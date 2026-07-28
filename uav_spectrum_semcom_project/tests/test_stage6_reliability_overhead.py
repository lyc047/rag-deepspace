from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_stage6_reliability_overhead import (  # noqa: E402
    _count_rate,
    _savings,
    _weighted_metric,
)


def _row(scenes: int, bits: float, count: float) -> dict:
    return {
        "evaluated_scene_count": scenes,
        "summary": {
            "actual_bits_per_scene": bits,
            "context_install_count": count,
        },
    }


def test_weighted_metric_uses_evaluated_scene_count() -> None:
    rows = [_row(10, 10.0, 1.0), _row(30, 30.0, 3.0)]
    assert _weighted_metric(rows, "actual_bits_per_scene") == 25.0


def test_count_rate_sums_counts_over_scenes() -> None:
    rows = [_row(10, 10.0, 1.0), _row(30, 30.0, 3.0)]
    assert _count_rate(rows, "context_install_count") == 0.1


def test_savings_sign_is_explicit() -> None:
    assert _savings(100.0, 80.0) == 20.0
    assert _savings(100.0, 120.0) == -20.0
