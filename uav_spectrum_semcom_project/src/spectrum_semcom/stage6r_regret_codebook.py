"""Safe fixed-action codebook feasibility tools for Stage-6R."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .stage6_task_codebook import SpectrumTaskState


@dataclass(frozen=True)
class SelectedActionCodebook:
    actions: tuple[tuple[int, ...], ...]
    newly_covered_weight: tuple[float, ...]
    newly_covered_count: tuple[int, ...]


def _validate_states(
    states: Sequence[SpectrumTaskState],
) -> tuple[SpectrumTaskState, ...]:
    values = tuple(states)
    if not values:
        raise ValueError("at least one task state is required")
    reference = values[0]
    for state in values[1:]:
        if (
            state.n_channels != reference.n_channels
            or state.schema != reference.schema
            or abs(state.epsilon_db - reference.epsilon_db) > 1e-12
        ):
            raise ValueError("task states must share one schema")
    return values


def exhaustive_action_tuples(
    states: Sequence[SpectrumTaskState],
) -> tuple[tuple[int, ...], ...]:
    values = _validate_states(states)
    alphabets = [profile.allowed_starts for profile in values[0].profiles]
    return tuple(tuple(map(int, row)) for row in product(*alphabets))


def optimal_action_candidates(
    states: Sequence[SpectrumTaskState],
) -> tuple[tuple[int, ...], ...]:
    values = _validate_states(states)
    return tuple(sorted({state.optimal_actions for state in values}))


def admissible_union_candidates(
    states: Sequence[SpectrumTaskState],
) -> tuple[tuple[int, ...], ...]:
    values = _validate_states(states)
    candidates: set[tuple[int, ...]] = set()
    for state in values:
        candidates.update(
            tuple(map(int, row))
            for row in product(
                *(profile.epsilon_optimal_starts for profile in state.profiles)
            )
        )
    return tuple(sorted(candidates))


def regret_matrix(
    states: Sequence[SpectrumTaskState],
    actions: Sequence[Sequence[int]],
) -> np.ndarray:
    values = _validate_states(states)
    action_array = np.asarray(actions, dtype=np.int32)
    if (
        action_array.ndim != 2
        or action_array.shape[0] < 1
        or action_array.shape[1] != len(values[0].profiles)
    ):
        raise ValueError("candidate actions do not match task schema")
    output = np.empty((len(values), action_array.shape[0]), dtype=np.float32)
    for state_index, state in enumerate(values):
        query_regrets = []
        for query_index, profile in enumerate(state.profiles):
            starts = action_array[:, query_index]
            if np.any(starts < 0) or np.any(
                starts >= len(profile.candidate_costs_dbm)
            ):
                raise ValueError("candidate action is outside a query alphabet")
            costs = np.asarray(profile.candidate_costs_dbm, dtype=np.float64)
            query_regrets.append(
                np.maximum(0.0, costs[starts] - profile.optimum_cost_dbm)
            )
        output[state_index] = np.max(
            np.stack(query_regrets, axis=1), axis=1
        ).astype(np.float32)
    return output


def campaign_balanced_weights(campaign_ids: Sequence[str]) -> np.ndarray:
    campaigns = np.asarray(campaign_ids).astype(str).reshape(-1)
    if campaigns.size < 1:
        raise ValueError("campaign IDs are required")
    unique, counts = np.unique(campaigns, return_counts=True)
    count_by_campaign = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / count_by_campaign[value] for value in campaigns],
        dtype=np.float64,
    )
    return weights


def greedy_select_actions(
    *,
    regrets: np.ndarray,
    actions: Sequence[Sequence[int]],
    epsilon_db: float,
    weights: np.ndarray,
    max_codewords: int,
) -> SelectedActionCodebook:
    regret = np.asarray(regrets, dtype=np.float64)
    action_tuple = tuple(tuple(map(int, row)) for row in actions)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        regret.ndim != 2
        or regret.shape != (weight.size, len(action_tuple))
        or weight.size < 1
        or not np.all(np.isfinite(regret))
        or not np.all(np.isfinite(weight))
        or np.any(weight <= 0.0)
        or not np.isfinite(epsilon_db)
        or epsilon_db < 0.0
        or max_codewords < 1
    ):
        raise ValueError("invalid regret matrix, weights, epsilon, or K")
    safe = regret <= float(epsilon_db) + 1e-12
    remaining = np.ones(weight.size, dtype=bool)
    scores = safe.T @ weight
    selected_indices: list[int] = []
    covered_weights: list[float] = []
    covered_counts: list[int] = []

    for _ in range(int(max_codewords)):
        if selected_indices:
            scores[np.asarray(selected_indices, dtype=int)] = -np.inf
        maximum = float(np.max(scores))
        if not np.isfinite(maximum) or maximum <= 0.0:
            break
        tied = np.flatnonzero(np.isclose(scores, maximum, rtol=1e-12, atol=1e-12))
        tie_rows = []
        for candidate in tied:
            mask = remaining & safe[:, candidate]
            mean_regret = float(
                np.average(regret[mask, candidate], weights=weight[mask])
            )
            tie_rows.append(
                (mean_regret, action_tuple[int(candidate)], int(candidate))
            )
        _, _, chosen = min(tie_rows)
        covered = remaining & safe[:, chosen]
        selected_indices.append(chosen)
        covered_weights.append(float(np.sum(weight[covered])))
        covered_counts.append(int(np.count_nonzero(covered)))
        remaining[covered] = False
        if np.any(covered):
            scores -= safe[covered].T @ weight[covered]
        if not np.any(remaining):
            break
    return SelectedActionCodebook(
        actions=tuple(action_tuple[index] for index in selected_indices),
        newly_covered_weight=tuple(covered_weights),
        newly_covered_count=tuple(covered_counts),
    )


def _cvar(values: np.ndarray, alpha: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size < 1:
        return 0.0
    start = min(array.size - 1, int(math.floor(float(alpha) * array.size)))
    return float(np.mean(array[start:]))


def evaluate_selected_actions(
    *,
    regrets: np.ndarray,
    selected_candidate_indices: Sequence[int],
    epsilon_db: float,
    exact_action_payload_bits: int,
    cvar_alpha: float,
) -> dict:
    regret = np.asarray(regrets, dtype=np.float64)
    selected = np.asarray(selected_candidate_indices, dtype=int).reshape(-1)
    if (
        regret.ndim != 2
        or selected.size < 1
        or np.any(selected < 0)
        or np.any(selected >= regret.shape[1])
        or exact_action_payload_bits < 1
        or not 0.0 < cvar_alpha < 1.0
    ):
        raise ValueError("invalid evaluation input")
    selected_regret = regret[:, selected]
    best_position = np.argmin(selected_regret, axis=1)
    best_regret = selected_regret[
        np.arange(selected_regret.shape[0]), best_position
    ]
    covered = best_regret <= float(epsilon_db) + 1e-12
    usage = np.bincount(
        best_position[covered], minlength=selected.size
    ).astype(np.int64)
    used = usage[usage > 0]
    probabilities = used / max(1, int(np.sum(used)))
    entropy = float(
        -np.sum(probabilities * np.log2(probabilities))
        if probabilities.size
        else 0.0
    )
    codeword_count = int(selected.size)
    symbol_width = max(1, int(math.ceil(math.log2(codeword_count + 1))))
    escape_rate = float(1.0 - np.mean(covered))
    covered_regret = best_regret[covered]
    return {
        "scene_count": int(regret.shape[0]),
        "codeword_count": codeword_count,
        "symbol_width_bits": symbol_width,
        "covered_count": int(np.count_nonzero(covered)),
        "coverage_rate": float(np.mean(covered)),
        "escape_count": int(np.count_nonzero(~covered)),
        "escape_rate": escape_rate,
        "used_codeword_count": int(used.size),
        "codeword_utilization_rate": float(used.size / codeword_count),
        "codeword_usage_entropy_bits": entropy,
        "covered_mean_max_regret_db": float(
            np.mean(covered_regret) if covered_regret.size else 0.0
        ),
        "covered_cvar_max_regret_db": _cvar(covered_regret, cvar_alpha),
        "covered_maximum_max_regret_db": float(
            np.max(covered_regret) if covered_regret.size else 0.0
        ),
        "exact_action_payload_bits": int(exact_action_payload_bits),
        "expected_semantic_payload_bits_per_scene": float(
            symbol_width + escape_rate * exact_action_payload_bits
        ),
        "exact_payload_bits_per_scene": int(exact_action_payload_bits),
        "payload_savings_percentage": float(
            100.0
            * (
                1.0
                - (
                    symbol_width
                    + escape_rate * exact_action_payload_bits
                )
                / exact_action_payload_bits
            )
        ),
        "unsafe_coded_action_count": 0,
        "usage_counts": usage.tolist(),
    }


__all__ = [
    "SelectedActionCodebook",
    "admissible_union_candidates",
    "campaign_balanced_weights",
    "evaluate_selected_actions",
    "exhaustive_action_tuples",
    "greedy_select_actions",
    "optimal_action_candidates",
    "regret_matrix",
]
