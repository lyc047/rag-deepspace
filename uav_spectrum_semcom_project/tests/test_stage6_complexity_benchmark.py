import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from run_stage6_complexity_benchmark import (  # noqa: E402
    _measure,
    _scaling_exponent,
)


def test_measure_reports_positive_quantiles() -> None:
    result = _measure(
        lambda: sum(range(20)),
        repetitions=5,
        warmups=1,
        divisor=1.0,
        unit_scale=1.0,
    )
    assert result["minimum"] > 0.0
    assert result["minimum"] <= result["median"] <= result["maximum"]
    assert result["p25"] <= result["p75"]


def test_scaling_exponent_recovers_quadratic_sequence() -> None:
    exponent = _scaling_exponent(
        [8, 16, 32, 64],
        [64.0, 256.0, 1024.0, 4096.0],
    )
    assert abs(exponent - 2.0) < 1e-12


def test_complexity_protocol_is_environment_sensitive_and_final_closed() -> None:
    protocol = json.loads(
        (
            PROJECT / "configs/stage6_complexity_benchmark_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        protocol["reproducibility_rule"]["classification"]
        == "environment_sensitive"
    )
    assert protocol["reproducibility_rule"][
        "timing_values_must_not_be_compared_for_exact_equality"
    ]
    assert not protocol["governance"][
        "frozen_algorithm_or_parameters_may_be_modified"
    ]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]
