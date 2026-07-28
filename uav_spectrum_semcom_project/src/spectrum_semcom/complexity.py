"""Reproducible model-size and linear-MAC accounting helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch
from torch import nn


def parameter_summary(model: nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "total_parameters": sum(x.numel() for x in parameters),
        "trainable_parameters": sum(x.numel() for x in parameters if x.requires_grad),
        "parameter_bytes_fp32": sum(x.numel() * x.element_size() for x in parameters),
    }


def linear_macs(model: nn.Module, batch_items: int = 1) -> int:
    """Count multiply-accumulates in Linear layers, excluding activations."""

    if batch_items < 1:
        raise ValueError("batch_items must be positive")
    return int(batch_items * sum(module.in_features * module.out_features for module in model.modules() if isinstance(module, nn.Linear)))


def benchmark_callable(function: Callable[[], object], *, warmup: int = 100, repetitions: int = 1000) -> dict[str, float | int]:
    if warmup < 0 or repetitions < 2:
        raise ValueError("warmup must be non-negative and repetitions at least two")
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(samples)
    return {
        "repetitions": repetitions,
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[int(0.95 * (repetitions - 1))]),
        "mean_ms": float(statistics.fmean(ordered)),
    }
