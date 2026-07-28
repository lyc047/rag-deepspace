"""Interpretable task-ambiguity and disagreement based C2 scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class MonotonicValueCalibrator:
    score_knots: np.ndarray
    value_knots: np.ndarray

    def predict(self, score: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(score, dtype=np.float64), self.score_knots, self.value_knots)


@dataclass(frozen=True)
class CandidateAction:
    node: int
    target_granularity: int
    predicted_value: float
    expected_bits: float

    @property
    def value_per_bit(self) -> float:
        return self.predicted_value / max(self.expected_bits, 1e-12)


def train_margin_temperature(beliefs: np.ndarray, demand_channels: int) -> float:
    margins = []
    for belief in np.asarray(beliefs, dtype=np.float64):
        blocks = np.convolve(belief, np.ones(demand_channels) / demand_channels, mode="valid")
        ordered = np.sort(blocks)
        margins.append(float(ordered[1] - ordered[0]))
    positive = np.asarray([value for value in margins if value > 1e-12])
    return max(float(np.median(positive)) if positive.size else 0.05, 1e-3)


def analytic_raw_score(current_belief: np.ndarray, candidate_current_report: np.ndarray, candidate_quality: np.ndarray, demand_channels: int, target_granularity: int, margin_temperature: float, freshness_tau_s: float = 0.5) -> float:
    belief = np.asarray(current_belief, dtype=np.float64).reshape(-1)
    report = np.asarray(candidate_current_report, dtype=np.float64).reshape(-1)
    quality = np.asarray(candidate_quality, dtype=np.float64).reshape(-1)
    if belief.shape != report.shape or quality.size != 10 or margin_temperature <= 0 or freshness_tau_s <= 0:
        raise ValueError("invalid analytic scheduler inputs")
    demand = int(demand_channels)
    blocks = np.convolve(belief, np.ones(demand) / demand, mode="valid")
    order = np.argsort(blocks)
    best, second = int(order[0]), int(order[1])
    margin = float(blocks[second] - blocks[best])
    ambiguity = float(np.exp(-margin / margin_temperature))
    mask = np.zeros_like(belief, dtype=bool)
    mask[best : best + demand] = True
    mask[second : second + demand] = True
    local_disagreement = float(np.mean(np.abs(report[mask] - belief[mask])))
    report_blocks = np.convolve(report, np.ones(demand) / demand, mode="valid")
    contrast_shift = abs(float((report_blocks[best] - report_blocks[second]) - (blocks[best] - blocks[second])))
    sensing = 1.0 / (1.0 + np.exp(-quality[0] / 4.0))
    reliability = sensing * np.clip(quality[1], 0, 1) * np.clip(quality[8], 0, 1) * np.exp(-max(0.0, quality[7]) / freshness_tau_s) * (1.0 - np.clip(quality[3], 0, 1))
    gain = {2: 0.9375, 3: 0.99609375}.get(int(target_granularity))
    if gain is None:
        raise ValueError("target granularity must be G2 or G3")
    return float(ambiguity * (local_disagreement + contrast_shift) * reliability * gain)


def fit_monotonic_value_calibrator(scores: np.ndarray, values: np.ndarray, n_bins: int = 10) -> MonotonicValueCalibrator:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("scores and values must be aligned finite vectors")
    if float(np.ptp(x)) <= 1e-15:
        center = float(x[0]); mean_value = float(np.mean(y))
        return MonotonicValueCalibrator(np.asarray([center - 1e-12, center + 1e-12]), np.asarray([mean_value, mean_value]))
    order = np.argsort(x, kind="stable")
    chunks = [chunk for chunk in np.array_split(order, min(int(n_bins), x.size)) if chunk.size]
    blocks = [[float(np.mean(x[chunk])), float(np.mean(y[chunk])), float(chunk.size)] for chunk in chunks]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][1] <= blocks[index + 1][1]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[2] + right[2]
        blocks[index:index + 2] = [[(left[0] * left[2] + right[0] * right[2]) / weight, (left[1] * left[2] + right[1] * right[2]) / weight, weight]]
        index = max(0, index - 1)
    score_knots = np.asarray([block[0] for block in blocks])
    value_knots = np.asarray([block[1] for block in blocks])
    if score_knots.size == 1:
        score_knots = np.asarray([score_knots[0] - 1e-12, score_knots[0] + 1e-12])
        value_knots = np.repeat(value_knots, 2)
    return MonotonicValueCalibrator(score_knots, value_knots)


def greedy_actions(candidates: list[CandidateAction], budget_bits: float, maximum_actions: int) -> tuple[CandidateAction, ...]:
    selected = []
    used_nodes: set[int] = set()
    remaining = float(budget_bits)
    for candidate in sorted(candidates, key=lambda item: (item.value_per_bit, item.predicted_value), reverse=True):
        if candidate.predicted_value <= 0 or candidate.node in used_nodes or candidate.expected_bits > remaining + 1e-12:
            continue
        selected.append(candidate); used_nodes.add(candidate.node); remaining -= candidate.expected_bits
        if len(selected) >= int(maximum_actions):
            break
    return tuple(selected)


def exact_multiple_choice_knapsack(candidates: list[CandidateAction], budget_bits: float, maximum_actions: int) -> tuple[CandidateAction, ...]:
    by_node: dict[int, list[CandidateAction]] = {}
    for candidate in candidates:
        by_node.setdefault(candidate.node, []).append(candidate)
    best_value = 0.0
    best_bits = 0.0
    best: tuple[CandidateAction, ...] = ()
    for choices in product(*[(None, *by_node[node]) for node in sorted(by_node)]):
        selected = tuple(item for item in choices if item is not None)
        bits = sum(item.expected_bits for item in selected)
        value = sum(item.predicted_value for item in selected)
        if len(selected) <= int(maximum_actions) and bits <= float(budget_bits) + 1e-12 and value > 0 and (value > best_value + 1e-15 or (abs(value - best_value) <= 1e-15 and bits < best_bits)):
            best_value, best_bits, best = value, bits, selected
    return best
