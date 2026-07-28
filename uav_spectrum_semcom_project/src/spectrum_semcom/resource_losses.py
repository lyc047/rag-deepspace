"""Resource-selection-aware losses and diagnostics for stage-4 Gate A."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


def _validate_occupancy_pair(predicted: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[1] < 1:
        raise ValueError("predicted and truth occupancy must have the same [batch, channels] shape")
    if not predicted.is_floating_point() or not truth.is_floating_point():
        raise ValueError("occupancy tensors must be floating point")
    if not torch.all(torch.isfinite(predicted)) or not torch.all(torch.isfinite(truth)):
        raise ValueError("occupancy tensors must be finite")
    if torch.any((predicted < 0) | (predicted > 1)) or torch.any((truth < 0) | (truth > 1)):
        raise ValueError("occupancy tensors must lie in [0, 1]")
    return predicted, truth


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def contiguous_block_occupancy(channel_occupancy: torch.Tensor, demand_channels: int) -> torch.Tensor:
    """Average occupancy of every contiguous resource block."""

    values = channel_occupancy
    if values.ndim != 2 or values.shape[1] < 1 or not values.is_floating_point():
        raise ValueError("channel_occupancy must be a floating [batch, channels] tensor")
    demand = int(demand_channels)
    if not 1 <= demand <= values.shape[1]:
        raise ValueError("demand_channels must be between one and the channel count")
    if demand == 1:
        return values
    return F.avg_pool1d(values.unsqueeze(1), kernel_size=demand, stride=1).squeeze(1)


def discrete_occupancy_regret(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    demand_channels: int,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """True occupancy gap of the discrete predicted choice to the oracle."""

    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    predicted_blocks = contiguous_block_occupancy(predicted, demand_channels)
    truth_blocks = contiguous_block_occupancy(truth, demand_channels)
    chosen = torch.argmin(predicted_blocks, dim=1)
    chosen_truth = truth_blocks.gather(1, chosen[:, None]).squeeze(1)
    regret = torch.clamp_min(chosen_truth - torch.min(truth_blocks, dim=1).values, 0.0)
    return _reduce(regret, reduction)


def soft_resource_distribution(
    predicted_occupancy: torch.Tensor,
    demand_channels: int,
    inverse_temperature: float,
) -> torch.Tensor:
    """Differentiable distribution over contiguous resource-block choices."""

    if inverse_temperature <= 0:
        raise ValueError("inverse_temperature must be positive")
    predicted_blocks = contiguous_block_occupancy(predicted_occupancy, demand_channels)
    return torch.softmax(-float(inverse_temperature) * predicted_blocks, dim=1)


def soft_occupancy_regret(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    demand_channels: int,
    inverse_temperature: float = 20.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Expected true occupancy under soft selection minus oracle occupancy."""

    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    probabilities = soft_resource_distribution(predicted, demand_channels, inverse_temperature)
    truth_blocks = contiguous_block_occupancy(truth, demand_channels)
    expected_truth = torch.sum(probabilities * truth_blocks, dim=1)
    regret = torch.clamp_min(expected_truth - torch.min(truth_blocks, dim=1).values, 0.0)
    return _reduce(regret, reduction)


def contiguous_block_ranking_loss(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    demand_channels: int,
    margin: float = 0.02,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Rank the truth-optimal block below every alternative predicted block."""

    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    if margin < 0:
        raise ValueError("ranking margin must be non-negative")
    predicted_blocks = contiguous_block_occupancy(predicted, demand_channels)
    truth_blocks = contiguous_block_occupancy(truth, demand_channels)
    oracle = torch.argmin(truth_blocks, dim=1)
    predicted_oracle = predicted_blocks.gather(1, oracle[:, None])
    truth_oracle = truth_blocks.gather(1, oracle[:, None])
    truth_gap = torch.clamp_min(truth_blocks - truth_oracle, 0.0)
    violations = torch.relu(float(margin) + predicted_oracle - predicted_blocks)
    mask = torch.ones_like(violations); mask.scatter_(1, oracle[:, None], 0.0)
    per_scene = torch.sum(violations * truth_gap * mask, dim=1) / torch.clamp_min(torch.sum(mask, dim=1), 1.0)
    return _reduce(per_scene, reduction)


def tail_contiguous_block_ranking_loss(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    demand_channels: int,
    margin: float = 0.02,
    cvar_alpha: float = 0.9,
) -> torch.Tensor:
    """CVaR of per-scene block-ranking violations for direct tail control."""

    per_scene = contiguous_block_ranking_loss(
        predicted_occupancy,
        truth_occupancy,
        demand_channels=demand_channels,
        margin=margin,
        reduction="none",
    )
    return empirical_cvar(per_scene, alpha=cvar_alpha)


def missed_occupancy_loss(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Positive-class log loss that penalizes underestimating occupied channels."""

    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    per_channel = -truth * torch.log(torch.clamp(predicted, min=1e-7, max=1.0))
    per_scene = per_channel.mean(dim=1)
    return _reduce(per_scene, reduction)


def empirical_cvar(losses: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """Mean of the worst empirical ``1-alpha`` fraction of scene losses."""

    values = losses.reshape(-1)
    if values.numel() == 0 or not values.is_floating_point() or not torch.all(torch.isfinite(values)):
        raise ValueError("losses must be a non-empty finite floating tensor")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    tail_count = max(1, int(ceil((1.0 - float(alpha)) * values.numel())))
    return torch.topk(values, k=tail_count, largest=True).values.mean()


def brier_occupancy_loss(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    per_scene = torch.square(predicted - truth).mean(dim=1)
    return _reduce(per_scene, reduction)


@dataclass(frozen=True)
class GateALossConfig:
    name: str
    detection_weight: float = 1.0
    rate_weight: float = 0.0
    resource_weight: float = 0.0
    miss_weight: float = 0.0
    tail_weight: float = 0.0
    calibration_weight: float = 0.0
    inverse_temperature: float = 20.0
    cvar_alpha: float = 0.9

    def __post_init__(self) -> None:
        weights = (
            self.detection_weight,
            self.rate_weight,
            self.resource_weight,
            self.miss_weight,
            self.tail_weight,
            self.calibration_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("Gate A loss weights must be non-negative")
        if self.inverse_temperature <= 0 or not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("invalid inverse temperature or CVaR alpha")


@dataclass(frozen=True)
class GateALossOutput:
    total: torch.Tensor
    detection: torch.Tensor
    rate_violation: torch.Tensor
    soft_regret: torch.Tensor
    missed_occupancy: torch.Tensor
    tail_regret: torch.Tensor
    calibration: torch.Tensor


def gate_a_loss_configs(
    *,
    rate_weight: float = 1.0,
    resource_weight: float = 1.0,
    miss_weight: float = 0.5,
    tail_weight: float = 0.5,
    calibration_weight: float = 0.25,
    inverse_temperature: float = 20.0,
    cvar_alpha: float = 0.9,
) -> dict[str, GateALossConfig]:
    """Four preregistered Gate A variants sharing all non-loss settings."""

    shared = {"inverse_temperature": inverse_temperature, "cvar_alpha": cvar_alpha}
    return {
        "detection_only": GateALossConfig("detection_only", **shared),
        "detection_plus_rate": GateALossConfig("detection_plus_rate", rate_weight=rate_weight, **shared),
        "detection_plus_resource": GateALossConfig(
            "detection_plus_resource", resource_weight=resource_weight, **shared
        ),
        "full_joint_loss": GateALossConfig(
            "full_joint_loss",
            rate_weight=rate_weight,
            resource_weight=resource_weight,
            miss_weight=miss_weight,
            tail_weight=tail_weight,
            calibration_weight=calibration_weight,
            **shared,
        ),
    }


def update_rate_dual_multiplier(
    current_multiplier: float,
    mean_expected_bits: float,
    bit_budget: float,
    step_size: float,
    maximum_multiplier: float | None = None,
) -> float:
    """One projected dual-ascent update for the normalized bit constraint."""

    values = (current_multiplier, mean_expected_bits, bit_budget, step_size)
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("dual update inputs must be finite")
    if current_multiplier < 0 or mean_expected_bits < 0 or bit_budget <= 0 or step_size <= 0:
        raise ValueError("invalid non-negative cost, positive budget, or positive step size")
    updated = max(0.0, float(current_multiplier) + float(step_size) * (float(mean_expected_bits) / bit_budget - 1.0))
    if maximum_multiplier is not None:
        if maximum_multiplier <= 0:
            raise ValueError("maximum_multiplier must be positive")
        updated = min(updated, float(maximum_multiplier))
    return updated


def compute_gate_a_loss(
    predicted_occupancy: torch.Tensor,
    truth_occupancy: torch.Tensor,
    expected_bits: torch.Tensor,
    bit_budget: float,
    config: GateALossConfig,
    demand_channels: int = 1,
    detection_loss: torch.Tensor | None = None,
) -> GateALossOutput:
    """Compute one Gate A objective with explicit, reportable components."""

    predicted, truth = _validate_occupancy_pair(predicted_occupancy, truth_occupancy)
    bits = expected_bits.reshape(-1)
    if bits.numel() not in {1, predicted.shape[0]} or not bits.is_floating_point() or not torch.all(torch.isfinite(bits)):
        raise ValueError("expected_bits must be one finite float per scene or a scalar")
    if torch.any(bits < 0) or bit_budget <= 0:
        raise ValueError("expected bits must be non-negative and bit_budget positive")
    if detection_loss is None:
        detection = F.binary_cross_entropy(torch.clamp(predicted, 1e-7, 1.0 - 1e-7), truth)
    else:
        if detection_loss.numel() != 1 or not torch.isfinite(detection_loss):
            raise ValueError("detection_loss must be one finite scalar")
        detection = detection_loss.reshape(())

    scene_regret = soft_occupancy_regret(
        predicted,
        truth,
        demand_channels=demand_channels,
        inverse_temperature=config.inverse_temperature,
        reduction="none",
    )
    soft_regret = scene_regret.mean()
    tail_regret = empirical_cvar(scene_regret, config.cvar_alpha)
    missed = missed_occupancy_loss(predicted, truth)
    calibration = brier_occupancy_loss(predicted, truth)
    rate_violation = bits.mean() / float(bit_budget) - 1.0
    total = (
        config.detection_weight * detection
        + config.rate_weight * rate_violation
        + config.resource_weight * soft_regret
        + config.miss_weight * missed
        + config.tail_weight * tail_regret
        + config.calibration_weight * calibration
    )
    return GateALossOutput(total, detection, rate_violation, soft_regret, missed, tail_regret, calibration)


@dataclass(frozen=True)
class ResourceTaskDiagnostics:
    oracle_clean_rate: float
    mean_random_oracle_gap: float
    mean_oracle_margin: float
    ambiguous_oracle_rate: float


@torch.no_grad()
def resource_task_diagnostics(
    truth_occupancy: torch.Tensor,
    demand_channels: int,
    clean_threshold: float,
    ambiguity_tolerance: float = 1e-6,
) -> ResourceTaskDiagnostics:
    """Detect a saturated or decision-ambiguous resource task before training."""

    if truth_occupancy.ndim != 2 or not truth_occupancy.is_floating_point():
        raise ValueError("truth_occupancy must be a floating [batch, channels] tensor")
    blocks = contiguous_block_occupancy(truth_occupancy, demand_channels)
    oracle = torch.min(blocks, dim=1).values
    random_gap = torch.mean(blocks, dim=1) - oracle
    if blocks.shape[1] == 1:
        margin = torch.zeros_like(oracle)
    else:
        margin = torch.topk(blocks, k=2, dim=1, largest=False).values[:, 1] - oracle
    return ResourceTaskDiagnostics(
        oracle_clean_rate=float(torch.mean((oracle <= clean_threshold).float()).item()),
        mean_random_oracle_gap=float(torch.mean(random_gap).item()),
        mean_oracle_margin=float(torch.mean(margin).item()),
        ambiguous_oracle_rate=float(torch.mean((margin <= ambiguity_tolerance).float()).item()),
    )
