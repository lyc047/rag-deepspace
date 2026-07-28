"""Controlled-training helpers shared by all Gate A variants."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .digital_link import DigitalLinkConfig, expected_transmitted_bits_awgn, nominal_transmitted_bits
from .gate_a_model import VariableRateOccupancyHead
from .resource_losses import discrete_occupancy_regret, empirical_cvar


def load_gate_a_cache(path: str | Path) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(Path(path), allow_pickle=False) as data:
        scene_ids = [str(value) for value in data["scene_ids"].tolist()]
        base = torch.as_tensor(data["base_occupancy"], dtype=torch.float32)
        truth = torch.as_tensor(data["truth_occupancy"], dtype=torch.float32)
        confidence = torch.as_tensor(data["detector_confidence"], dtype=torch.float32)
    if base.shape != truth.shape or base.ndim != 2 or len(scene_ids) != base.shape[0] or confidence.shape != (base.shape[0],):
        raise ValueError("invalid Gate A cache shapes")
    return scene_ids, base, truth, confidence


def precision_link_costs(model: VariableRateOccupancyHead, config: DigitalLinkConfig) -> torch.Tensor:
    application = [model.application_overhead_bits + model.n_channels * bits for bits in model.precision_bits]
    return torch.tensor([nominal_transmitted_bits(value, config) for value in application], dtype=torch.float32)


@torch.no_grad()
def evaluate_gate_a_head(
    model: VariableRateOccupancyHead,
    base: torch.Tensor,
    truth: torch.Tensor,
    demand_channels: int,
    link_config: DigitalLinkConfig,
    bit_budget: float,
    clean_threshold: float = 0.02,
) -> dict[str, Any]:
    model.eval()
    reconstructed, selected_bits, application_bits = model.infer(base)
    regrets = discrete_occupancy_regret(reconstructed, truth, demand_channels, reduction="none")
    costs = torch.tensor([nominal_transmitted_bits(int(value), link_config) for value in application_bits])
    predicted_blocks = torch.nn.functional.avg_pool1d(reconstructed[:, None], demand_channels, stride=1).squeeze(1)
    truth_blocks = torch.nn.functional.avg_pool1d(truth[:, None], demand_channels, stride=1).squeeze(1)
    chosen = torch.argmin(predicted_blocks, dim=1)
    chosen_truth = truth_blocks.gather(1, chosen[:, None]).squeeze(1)
    counts = Counter(int(value) for value in selected_bits.tolist())
    return {
        "mean_discrete_regret": float(regrets.mean()),
        "cvar_0_9_regret": float(empirical_cvar(regrets, 0.9)),
        "clean_resource_rate": float((chosen_truth <= clean_threshold).float().mean()),
        "brier": float(torch.square(reconstructed - truth).mean()),
        "mean_application_bits": float(application_bits.float().mean()),
        "mean_nominal_transmitted_bits": float(costs.float().mean()),
        "budget_violation": max(0.0, float(costs.float().mean()) / float(bit_budget) - 1.0),
        "precision_counts": {str(bits): counts.get(bits, 0) for bits in model.precision_bits},
    }
