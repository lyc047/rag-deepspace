"""Development-only calibration-shift robustness helpers for C1-v2."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def calibration_shift_views(
    occupancy: torch.Tensor,
    specs: Sequence[Mapping[str, float]],
    *,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    if occupancy.ndim != 2 or not occupancy.is_floating_point() or torch.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("occupancy must be a floating [batch, channels] tensor in [0, 1]")
    clipped = torch.clamp(occupancy, 1e-5, 1 - 1e-5)
    logit = torch.logit(clipped)
    views = []
    for spec in specs:
        temperature = float(spec.get("temperature", 1.0)); bias = float(spec.get("bias", 0.0)); noise_std = float(spec.get("noise_std", 0.0))
        if temperature <= 0 or noise_std < 0:
            raise ValueError("view temperature must be positive and noise non-negative")
        shifted = torch.sigmoid(temperature * logit + bias)
        if noise_std:
            noise = torch.randn(shifted.shape, generator=generator, dtype=shifted.dtype, device=shifted.device)
            shifted = shifted + noise_std * noise
        views.append(torch.clamp(shifted, 1e-5, 1 - 1e-5))
    if not views:
        raise ValueError("at least one robustness view is required")
    return views
