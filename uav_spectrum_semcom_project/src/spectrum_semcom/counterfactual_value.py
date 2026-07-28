"""Truth-audited counterfactual node--granularity value labels for C2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from .digital_link import DigitalLinkConfig, expected_transmitted_bits_awgn, nominal_transmitted_bits
from .multigranular_semantics import SemanticMessage, SemanticQuality, encode_semantic_message, quantize_probabilities


GRANULARITY_BITS = {1: 1, 2: 4, 3: 8}


@dataclass(frozen=True)
class CounterfactualValueLabel:
    scene_id: str
    current_state: tuple[int, ...]
    candidate_node: int
    target_granularity: int
    current_belief: tuple[float, ...]
    candidate_current_report: tuple[float, ...]
    candidate_report: tuple[float, ...]
    candidate_quality: tuple[float, ...]
    baseline_regret: float
    upgraded_regret: float
    marginal_value: float
    application_bits: int
    nominal_transmitted_bits: int
    expected_transmitted_bits: float
    value_per_expected_bit: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OneStepOracleResult:
    baseline_regret: float
    best_upgraded_regret: float
    regret_reduction: float
    best_node: int
    best_target_granularity: int


def discrete_scene_regret(belief: np.ndarray, truth: np.ndarray, demand_channels: int) -> float:
    estimated = np.asarray(belief, dtype=np.float64).reshape(-1)
    actual = np.asarray(truth, dtype=np.float64).reshape(-1)
    if estimated.shape != actual.shape or not np.all(np.isfinite(estimated)) or np.any((estimated < 0) | (estimated > 1)) or np.any((actual < 0) | (actual > 1)):
        raise ValueError("belief and truth must be equally shaped finite occupancy vectors in [0, 1]")
    demand = int(demand_channels)
    if not 1 <= demand <= estimated.size:
        raise ValueError("invalid demand_channels")
    estimated_blocks = np.convolve(estimated, np.ones(demand) / demand, mode="valid")
    truth_blocks = np.convolve(actual, np.ones(demand) / demand, mode="valid")
    chosen = int(np.argmin(estimated_blocks))
    return float(max(0.0, truth_blocks[chosen] - np.min(truth_blocks)))


def admissible_current_states(n_nodes: int, max_upgraded_nodes: int = 1) -> list[tuple[int, ...]]:
    if n_nodes < 1 or max_upgraded_nodes < 0:
        raise ValueError("invalid node count or state depth")
    states = []
    for state in product((1, 2, 3), repeat=n_nodes):
        if sum(value > 1 for value in state) <= max_upgraded_nodes:
            states.append(tuple(int(value) for value in state))
    return states


def quantized_node_reports(node_occupancy: np.ndarray, preview_probability_bits: int = 1) -> dict[int, np.ndarray]:
    values = np.asarray(node_occupancy, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("node_occupancy must be [nodes, channels] in [0, 1]")
    if preview_probability_bits not in (1, 2):
        raise ValueError("G1 preview_probability_bits must be 1 or 2")
    bits_by_granularity = {**GRANULARITY_BITS, 1: int(preview_probability_bits)}
    return {granularity: np.stack([quantize_probabilities(row, bits)[1] for row in values]) for granularity, bits in bits_by_granularity.items()}


def fuse_state(reports: dict[int, np.ndarray], state: tuple[int, ...]) -> np.ndarray:
    if not state or any(value not in reports for value in state):
        raise ValueError("state references an unavailable granularity")
    return np.mean(np.stack([reports[granularity][node] for node, granularity in enumerate(state)]), axis=0)


def candidate_message_cost(
    scene_id: str,
    node: int,
    target: int,
    occupancy: np.ndarray,
    quality: SemanticQuality,
    link_config: DigitalLinkConfig,
    probability_bits: int | None = None,
) -> tuple[int, int, float]:
    name = f"G{target}"
    bits = GRANULARITY_BITS[target] if probability_bits is None else int(probability_bits)
    if target != 1 and bits != GRANULARITY_BITS[target]:
        raise ValueError("G2/G3 candidate bits are frozen for this diagnostic")
    message = SemanticMessage(name, node, scene_id, occupancy, probability_bits=bits, quality=None if target == 1 else quality, evidence=())
    application = encode_semantic_message(message).application_bits
    return application, nominal_transmitted_bits(application, link_config), expected_transmitted_bits_awgn(application, link_config)


def generate_counterfactual_labels(
    *,
    split_role: str,
    scene_id: str,
    node_occupancy: np.ndarray,
    truth_occupancy: np.ndarray,
    qualities: tuple[SemanticQuality, ...],
    link_configs: tuple[DigitalLinkConfig, ...],
    demand_channels: int,
    max_upgraded_nodes: int = 1,
    preview_probability_bits: int = 1,
) -> list[CounterfactualValueLabel]:
    """Generate labels only where training truth access is explicitly authorized."""

    if split_role != "train":
        raise PermissionError("counterfactual supervision labels may access truth on train only")
    reports = quantized_node_reports(node_occupancy, preview_probability_bits)
    if len(qualities) != node_occupancy.shape[0]:
        raise ValueError("one quality vector is required per node")
    if len(link_configs) != node_occupancy.shape[0]:
        raise ValueError("one reporting-link configuration is required per node")
    labels: list[CounterfactualValueLabel] = []
    for state in admissible_current_states(node_occupancy.shape[0], max_upgraded_nodes):
        current = fuse_state(reports, state)
        baseline_regret = discrete_scene_regret(current, truth_occupancy, demand_channels)
        for node, current_granularity in enumerate(state):
            for target in (2, 3):
                if target <= current_granularity:
                    continue
                upgraded_state = list(state); upgraded_state[node] = target
                upgraded = fuse_state(reports, tuple(upgraded_state))
                upgraded_regret = discrete_scene_regret(upgraded, truth_occupancy, demand_channels)
                value = baseline_regret - upgraded_regret
                application, transmitted, expected = candidate_message_cost(scene_id, node, target, reports[target][node], qualities[node], link_configs[node])
                quality_values = tuple(float(value) for value in qualities[node].__dict__.values())
                labels.append(CounterfactualValueLabel(scene_id, state, node, target, tuple(float(x) for x in current), tuple(float(x) for x in reports[current_granularity][node]), tuple(float(x) for x in reports[target][node]), quality_values, baseline_regret, upgraded_regret, value, application, transmitted, expected, value / max(expected, 1)))
    return labels


def evaluate_one_step_oracle(node_occupancy: np.ndarray, truth_occupancy: np.ndarray, demand_channels: int, preview_probability_bits: int = 1) -> OneStepOracleResult:
    """Aggregate evaluation helper; it does not expose or serialize supervision rows."""
    reports=quantized_node_reports(node_occupancy, preview_probability_bits); state=(1,)*node_occupancy.shape[0]; current=fuse_state(reports,state); baseline=discrete_scene_regret(current,truth_occupancy,demand_channels)
    best=(baseline,-1,-1)
    for node in range(node_occupancy.shape[0]):
        for target in (2,3):
            upgraded=list(state); upgraded[node]=target; regret=discrete_scene_regret(fuse_state(reports,tuple(upgraded)),truth_occupancy,demand_channels)
            if regret < best[0]: best=(regret,node,target)
    return OneStepOracleResult(baseline,best[0],baseline-best[0],best[1],best[2])
