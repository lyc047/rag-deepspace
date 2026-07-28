"""Deterministic helpers for Stage-6 task-scale sensitivity experiments.

The module contains no data access and no Final access mechanism.  It only
validates the registered task grid and computes descriptive rate metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TaskScaleGridPoint:
    n_channels: int
    epsilon_db: float
    query_set_id: str
    demand_ratios: tuple[float, ...]
    demands: tuple[int, ...]

    @property
    def grid_point_id(self) -> str:
        epsilon = f"{self.epsilon_db:.3f}".rstrip("0").rstrip(".")
        epsilon = epsilon.replace(".", "p")
        return f"n{self.n_channels}_e{epsilon}_{self.query_set_id}"


def rounded_demands(
    n_channels: int, demand_ratios: Iterable[float]
) -> tuple[int, ...]:
    """Map registered demand ratios to distinct valid channel counts."""

    n_value = int(n_channels)
    ratios = tuple(float(value) for value in demand_ratios)
    demands = tuple(int(round(n_value * value)) for value in ratios)
    if (
        n_value < 1
        or not ratios
        or any(not 0.0 < value <= 1.0 for value in ratios)
        or any(value < 1 or value > n_value for value in demands)
        or len(demands) != len(set(demands))
    ):
        raise ValueError("invalid or duplicate rounded demands")
    return demands


def registered_grid_points(protocol: dict) -> tuple[TaskScaleGridPoint, ...]:
    """Expand and validate the complete preregistered sensitivity grid."""

    task = protocol["task_grid"]
    query_sets = tuple(task["query_sets"])
    identifiers = [str(value["query_set_id"]) for value in query_sets]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("query-set identifiers must be unique")
    points = []
    for n_channels in task["n_channels"]:
        for epsilon_db in task["epsilon_db"]:
            if float(epsilon_db) < 0.0 or not np.isfinite(epsilon_db):
                raise ValueError("epsilon must be finite and non-negative")
            for query_set in query_sets:
                ratios = tuple(
                    float(value) for value in query_set["demand_ratios"]
                )
                points.append(
                    TaskScaleGridPoint(
                        n_channels=int(n_channels),
                        epsilon_db=float(epsilon_db),
                        query_set_id=str(query_set["query_set_id"]),
                        demand_ratios=ratios,
                        demands=rounded_demands(int(n_channels), ratios),
                    )
                )
    identities = [value.grid_point_id for value in points]
    if len(identities) != len(set(identities)):
        raise ValueError("sensitivity grid identities are not unique")
    return tuple(points)


def expected_nonempty_query_sets(base_ratios: Sequence[float]) -> set[tuple[float, ...]]:
    """Return every non-empty ordered subset of the registered base ratios."""

    ratios = tuple(float(value) for value in base_ratios)
    return {
        tuple(items)
        for width in range(1, len(ratios) + 1)
        for items in combinations(ratios, width)
    }


def validate_complete_query_lattice(protocol: dict) -> None:
    """Require the registered query sets to cover the complete subset lattice."""

    task = protocol["task_grid"]
    expected = expected_nonempty_query_sets(task["base_demand_ratios"])
    observed = {
        tuple(float(item) for item in value["demand_ratios"])
        for value in task["query_sets"]
    }
    if observed != expected:
        raise ValueError("query sets do not form the complete non-empty lattice")


def codeword_usage_metrics(
    symbol_ids: Iterable[int],
    *,
    codeword_count: int,
    escape_symbol: int,
) -> dict[str, float | int]:
    """Summarize compact-codeword use while retaining escape observations."""

    values = np.asarray(tuple(int(value) for value in symbol_ids), dtype=np.int64)
    count = int(codeword_count)
    if (
        values.size < 1
        or count < 1
        or int(escape_symbol) != count
        or np.any(values < 0)
        or np.any(values > count)
    ):
        raise ValueError("invalid symbol sequence")
    compact = values[values < count]
    counts = np.bincount(compact, minlength=count).astype(np.float64)
    used = int(np.count_nonzero(counts))
    if compact.size:
        probabilities = counts[counts > 0.0] / float(compact.size)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0
    return {
        "scene_count": int(values.size),
        "compact_scene_count": int(compact.size),
        "escape_scene_count": int(np.sum(values == count)),
        "escape_rate": float(np.mean(values == count)),
        "used_codeword_count": used,
        "codeword_utilization_rate": float(used / count),
        "codeword_usage_entropy_bits": entropy,
        "effective_codeword_perplexity": float(2.0**entropy),
    }


def amortized_bits_per_scene(
    operational_bits_per_scene: float,
    context_install_bits: int,
    horizon_scenes: int,
) -> float:
    """Add one durable codebook installation amortized over a deployment."""

    if (
        float(operational_bits_per_scene) < 0.0
        or int(context_install_bits) < 0
        or int(horizon_scenes) < 1
    ):
        raise ValueError("invalid amortization input")
    return float(
        float(operational_bits_per_scene)
        + int(context_install_bits) / int(horizon_scenes)
    )


def codebook_install_break_even_scenes(
    operational_bits_per_scene: float,
    exact_bits_per_scene: float,
    context_install_bits: int,
) -> float | None:
    """Return the minimum continuous deployment horizon that repays install bits."""

    semantic = float(operational_bits_per_scene)
    exact = float(exact_bits_per_scene)
    install = int(context_install_bits)
    if semantic < 0.0 or exact <= 0.0 or install < 0:
        raise ValueError("invalid break-even input")
    margin = exact - semantic
    if margin <= 0.0:
        return None
    return float(math.ceil(install / margin))
