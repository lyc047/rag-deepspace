import torch
from torch import nn

from spectrum_semcom.complexity import benchmark_callable, linear_macs, parameter_summary


def test_linear_complexity_and_parameter_counts() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert linear_macs(model) == 20
    assert linear_macs(model, 5) == 100
    summary = parameter_summary(model)
    assert summary["total_parameters"] == 26
    assert summary["parameter_bytes_fp32"] == 104


def test_benchmark_callable_returns_positive_summary() -> None:
    result = benchmark_callable(lambda: 1 + 1, warmup=1, repetitions=5)
    assert result["repetitions"] == 5
    assert result["median_ms"] >= 0
