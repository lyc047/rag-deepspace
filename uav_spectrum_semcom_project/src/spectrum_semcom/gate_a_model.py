"""Shared variable-rate occupancy head for the four Gate A objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GateAHeadOutput:
    refined_occupancy: torch.Tensor
    reconstructed_occupancy: torch.Tensor
    precision_probabilities: torch.Tensor
    expected_application_bits: torch.Tensor


def ste_quantize_probabilities(probabilities: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in {1, 2, 4, 8}:
        raise ValueError("bits must be one of 1, 2, 4, or 8")
    if not probabilities.is_floating_point() or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be floating values in [0, 1]")
    levels = float(2**bits - 1)
    hard = torch.round(probabilities * levels) / levels
    return probabilities + (hard - probabilities).detach()


class VariableRateOccupancyHead(nn.Module):
    """Refine frozen detector occupancy and choose a G2 probability precision."""

    precision_bits = (1, 2, 4, 8)

    def __init__(self, n_channels: int, hidden_dim: int = 32, g2_header_and_quality_bits: int = 151):
        super().__init__()
        if n_channels < 2 or hidden_dim < 1 or g2_header_and_quality_bits < 1:
            raise ValueError("invalid channel count, hidden dimension, or application overhead")
        self.n_channels = int(n_channels)
        self.application_overhead_bits = int(g2_header_and_quality_bits)
        self.shared = nn.Sequential(nn.Linear(n_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.residual = nn.Linear(hidden_dim, n_channels)
        self.rate_logits = nn.Linear(hidden_dim, len(self.precision_bits))
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.rate_logits.weight); nn.init.zeros_(self.rate_logits.bias)

    def forward(self, base_occupancy: torch.Tensor, rate_temperature: float = 1.0) -> GateAHeadOutput:
        if base_occupancy.ndim != 2 or base_occupancy.shape[1] != self.n_channels:
            raise ValueError(f"base_occupancy must have shape [batch, {self.n_channels}]")
        if rate_temperature <= 0:
            raise ValueError("rate_temperature must be positive")
        clipped = torch.clamp(base_occupancy, 1e-5, 1.0 - 1e-5)
        hidden = self.shared(clipped)
        refined = torch.sigmoid(torch.logit(clipped) + self.residual(hidden))
        precision_probabilities = torch.softmax(self.rate_logits(hidden) / float(rate_temperature), dim=1)
        candidates = torch.stack([ste_quantize_probabilities(refined, bits) for bits in self.precision_bits], dim=1)
        reconstructed = torch.sum(precision_probabilities[:, :, None] * candidates, dim=1)
        bit_options = torch.as_tensor(
            [self.application_overhead_bits + self.n_channels * bits for bits in self.precision_bits],
            dtype=base_occupancy.dtype,
            device=base_occupancy.device,
        )
        expected_bits = torch.sum(precision_probabilities * bit_options[None, :], dim=1)
        return GateAHeadOutput(refined, reconstructed, precision_probabilities, expected_bits)

    @torch.no_grad()
    def infer(self, base_occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(base_occupancy)
        selected = torch.argmax(output.precision_probabilities, dim=1)
        bit_tensor = torch.as_tensor(self.precision_bits, device=base_occupancy.device)
        selected_bits = bit_tensor[selected]
        reconstructed = torch.stack(
            [ste_quantize_probabilities(output.refined_occupancy[index], int(selected_bits[index])) for index in range(base_occupancy.shape[0])]
        )
        application_bits = self.application_overhead_bits + self.n_channels * selected_bits
        return reconstructed, selected_bits, application_bits


class FixedPrecisionOccupancyHead(nn.Module):
    """C1-v2 head with a shared precision policy for strict bit matching."""

    def __init__(self, n_channels: int, hidden_dim: int = 32, probability_bits: int = 2, g2_header_and_quality_bits: int = 151):
        super().__init__()
        if n_channels < 2 or hidden_dim < 1 or probability_bits not in {1, 2, 4, 8}:
            raise ValueError("invalid fixed-precision head settings")
        self.n_channels = int(n_channels)
        self.probability_bits = int(probability_bits)
        self.application_bits = int(g2_header_and_quality_bits) + self.n_channels * self.probability_bits
        self.shared = nn.Sequential(nn.Linear(n_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.residual = nn.Linear(hidden_dim, n_channels)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, base_occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if base_occupancy.ndim != 2 or base_occupancy.shape[1] != self.n_channels:
            raise ValueError(f"base_occupancy must have shape [batch, {self.n_channels}]")
        clipped = torch.clamp(base_occupancy, 1e-5, 1 - 1e-5)
        refined = torch.sigmoid(torch.logit(clipped) + self.residual(self.shared(clipped)))
        return refined, ste_quantize_probabilities(refined, self.probability_bits)

    @torch.no_grad()
    def infer(self, base_occupancy: torch.Tensor) -> torch.Tensor:
        return self.forward(base_occupancy)[1]
