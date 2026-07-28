"""Deployable C2 counterfactual value-per-bit prediction.

The feature interface intentionally cannot receive the candidate high-precision
report or occupancy truth: those quantities exist only while constructing
training labels and offline evaluation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Standardization:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float


def build_deployable_value_features(
    *,
    current_belief: np.ndarray,
    candidate_current_report: np.ndarray,
    candidate_quality: np.ndarray,
    quality_availability: np.ndarray,
    current_state: tuple[int, ...] | np.ndarray,
    candidate_node: int,
    target_granularity: int,
    expected_transmitted_bits: float,
) -> np.ndarray:
    belief = np.asarray(current_belief, dtype=np.float32).reshape(-1)
    report = np.asarray(candidate_current_report, dtype=np.float32).reshape(-1)
    quality = np.asarray(candidate_quality, dtype=np.float32).reshape(-1)
    availability = np.asarray(quality_availability, dtype=bool).reshape(-1)
    state = np.asarray(current_state, dtype=np.int64).reshape(-1)
    node = int(candidate_node)
    target = int(target_granularity)
    if belief.shape != report.shape or not np.all(np.isfinite(belief)) or not np.all(np.isfinite(report)):
        raise ValueError("belief and current report must be equally shaped and finite")
    if quality.shape != availability.shape or not np.all(np.isfinite(quality)):
        raise ValueError("quality values and availability mask must align")
    if not 0 <= node < state.size or np.any((state < 1) | (state > 3)) or target not in (2, 3):
        raise ValueError("invalid state, node, or target granularity")
    if target <= state[node] or not np.isfinite(expected_transmitted_bits) or expected_transmitted_bits <= 0:
        raise ValueError("candidate must be an affordable granularity upgrade")
    state_one_hot = np.eye(3, dtype=np.float32)[state - 1].reshape(-1)
    node_one_hot = np.eye(state.size, dtype=np.float32)[node]
    target_one_hot = np.asarray([target == 2, target == 3], dtype=np.float32)
    visible_quality = np.where(availability, quality, 0.0).astype(np.float32)
    return np.concatenate(
        [belief, report, visible_quality, availability.astype(np.float32), state_one_hot, node_one_hot, target_one_hot, np.asarray([np.log1p(expected_transmitted_bits)], dtype=np.float32)]
    )


def fit_standardization(features: np.ndarray, targets: np.ndarray) -> Standardization:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("finite 2-D features and aligned targets are required")
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    target_scale = max(float(y.std()), 1e-12)
    return Standardization(x.mean(axis=0).astype(np.float32), feature_scale.astype(np.float32), float(y.mean()), target_scale)


class CounterfactualValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("positive input and hidden dimensions are required")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.SiLU(), nn.Dropout(float(dropout))))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def grouped_value_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    no_action_standardized: float,
    huber_delta: float = 1.0,
    listwise_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predictions.shape != targets.shape or valid.shape != predictions.shape:
        raise ValueError("prediction, target, and group mask shapes must match")
    regression = F.huber_loss(predictions[valid], targets[valid], delta=float(huber_delta))
    no_action = torch.full((predictions.shape[0], 1), float(no_action_standardized), device=predictions.device)
    prediction_logits = torch.cat((predictions, no_action), dim=1)
    target_logits = torch.cat((targets, no_action), dim=1)
    extended_valid = torch.cat((valid, torch.ones((valid.shape[0], 1), dtype=torch.bool, device=valid.device)), dim=1)
    prediction_logits = prediction_logits.masked_fill(~extended_valid, -torch.inf)
    target_logits = target_logits.masked_fill(~extended_valid, -torch.inf)
    target_distribution = torch.softmax(target_logits, dim=1)
    log_prediction = torch.log_softmax(prediction_logits, dim=1)
    cross_entropy_terms = torch.where(extended_valid, target_distribution * log_prediction, 0.0)
    ranking = -cross_entropy_terms.sum(dim=1).mean()
    total = regression + float(listwise_weight) * ranking
    return total, {"regression": regression.detach(), "listwise": ranking.detach()}
