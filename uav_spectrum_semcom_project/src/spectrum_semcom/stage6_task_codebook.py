"""Task-loss profiles and an analytic epsilon-regret spectrum codebook.

This module is development-only.  It does not read data, choose epsilon from
evaluation results, or interact with any Final access mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import count
from typing import Iterable, Sequence

import numpy as np

from spectrum_semcom.stage5_event_semantics import contiguous_block_costs


@dataclass(frozen=True)
class SpectrumTaskQuery:
    """One contiguous-spectrum allocation query."""

    demand_channels: int
    allowed_starts: tuple[int, ...] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if int(self.demand_channels) < 1:
            raise ValueError("demand_channels must be positive")
        if self.allowed_starts is not None:
            starts = tuple(int(value) for value in self.allowed_starts)
            if (
                not starts
                or any(value < 0 for value in starts)
                or len(starts) != len(set(starts))
                or starts != tuple(sorted(starts))
            ):
                raise ValueError(
                    "allowed_starts must be unique, sorted, and non-negative"
                )


@dataclass(frozen=True)
class QueryCostProfile:
    """Task costs and epsilon-admissible actions for one query."""

    query: SpectrumTaskQuery
    candidate_costs_dbm: tuple[float, ...]
    optimum_start: int
    optimum_cost_dbm: float
    epsilon_optimal_starts: tuple[int, ...]
    runner_up_gap_db: float

    def regret_db(self, selected_start: int) -> float:
        starts = self.allowed_starts
        if int(selected_start) not in starts:
            return float("inf")
        return float(
            max(
                0.0,
                self.candidate_costs_dbm[int(selected_start)]
                - self.optimum_cost_dbm,
            )
        )

    @property
    def allowed_starts(self) -> tuple[int, ...]:
        if self.query.allowed_starts is not None:
            return self.query.allowed_starts
        return tuple(range(len(self.candidate_costs_dbm)))


@dataclass(frozen=True)
class SpectrumTaskState:
    """Multi-query task semantics derived from one spectrum observation."""

    n_channels: int
    epsilon_db: float
    profiles: tuple[QueryCostProfile, ...]

    @property
    def queries(self) -> tuple[SpectrumTaskQuery, ...]:
        return tuple(profile.query for profile in self.profiles)

    @property
    def optimal_actions(self) -> tuple[int, ...]:
        return tuple(profile.optimum_start for profile in self.profiles)

    @property
    def schema(
        self,
    ) -> tuple[tuple[int, tuple[int, ...] | None, str], ...]:
        return tuple(
            (
                profile.query.demand_channels,
                profile.query.allowed_starts,
                profile.query.name,
            )
            for profile in self.profiles
        )

    def regrets_db(self, actions: Sequence[int]) -> tuple[float, ...]:
        values = tuple(int(value) for value in actions)
        if len(values) != len(self.profiles):
            raise ValueError("action tuple does not match query schema")
        return tuple(
            profile.regret_db(action)
            for profile, action in zip(self.profiles, values)
        )

    def max_regret_db(self, actions: Sequence[int]) -> float:
        return float(max(self.regrets_db(actions), default=0.0))

    def accepts(self, actions: Sequence[int], tolerance: float = 1e-12) -> bool:
        return self.max_regret_db(actions) <= self.epsilon_db + float(tolerance)


@dataclass(frozen=True)
class TaskCodeword:
    codeword_id: int
    decoder_actions: tuple[int, ...]
    training_coverage_count: int


@dataclass(frozen=True)
class GreedyTaskCodebook:
    """Ordered codewords fitted only from development states."""

    n_channels: int
    queries: tuple[SpectrumTaskQuery, ...]
    epsilon_db: float
    codewords: tuple[TaskCodeword, ...]
    training_scene_count: int
    exact_action_tuple_count: int
    training_covered_count: int

    @property
    def symbol_width_bits(self) -> int:
        """Fixed width including one reserved escape symbol."""

        return max(1, int(math.ceil(math.log2(len(self.codewords) + 1))))

    @property
    def escape_symbol(self) -> int:
        return len(self.codewords)


@dataclass(frozen=True)
class CodewordDecision:
    symbol_id: int
    decoder_actions: tuple[int, ...]
    regret_by_query_db: tuple[float, ...]
    max_regret_db: float
    uses_fallback: bool
    payload_bits: int


def build_task_state(
    channel_power_dbm: np.ndarray,
    queries: Iterable[SpectrumTaskQuery],
    *,
    epsilon_db: float,
) -> SpectrumTaskState:
    """Build deterministic multi-query task-loss profiles."""

    powers = np.asarray(channel_power_dbm, dtype=np.float64).reshape(-1)
    query_tuple = tuple(queries)
    if (
        powers.size < 1
        or not np.all(np.isfinite(powers))
        or not query_tuple
        or len({query.demand_channels for query in query_tuple})
        != len(query_tuple)
        or not np.isfinite(epsilon_db)
        or float(epsilon_db) < 0.0
    ):
        raise ValueError("invalid spectrum, query set, or epsilon")

    profiles = []
    for query in query_tuple:
        costs = contiguous_block_costs(powers, query.demand_channels)
        allowed = (
            tuple(range(costs.size))
            if query.allowed_starts is None
            else query.allowed_starts
        )
        if any(start >= costs.size for start in allowed):
            raise ValueError("allowed start is outside the candidate set")
        allowed_costs = np.asarray([costs[start] for start in allowed])
        optimum_position = int(np.argmin(allowed_costs))
        optimum_start = int(allowed[optimum_position])
        optimum_cost = float(costs[optimum_start])
        epsilon_starts = tuple(
            int(start)
            for start in allowed
            if float(costs[start] - optimum_cost) <= float(epsilon_db) + 1e-12
        )
        sorted_costs = np.sort(allowed_costs)
        runner_up_gap = (
            float(max(0.0, sorted_costs[1] - sorted_costs[0]))
            if sorted_costs.size >= 2
            else float("inf")
        )
        profiles.append(
            QueryCostProfile(
                query=query,
                candidate_costs_dbm=tuple(float(value) for value in costs),
                optimum_start=optimum_start,
                optimum_cost_dbm=optimum_cost,
                epsilon_optimal_starts=epsilon_starts,
                runner_up_gap_db=runner_up_gap,
            )
        )
    return SpectrumTaskState(
        n_channels=int(powers.size),
        epsilon_db=float(epsilon_db),
        profiles=tuple(profiles),
    )


def _validate_compatible_states(
    states: Sequence[SpectrumTaskState],
) -> tuple[
    int,
    float,
    tuple[SpectrumTaskQuery, ...],
    tuple[tuple[int, tuple[int, ...] | None, str], ...],
]:
    if not states:
        raise ValueError("at least one development state is required")
    reference = states[0]
    for state in states[1:]:
        if (
            state.n_channels != reference.n_channels
            or state.schema != reference.schema
            or abs(state.epsilon_db - reference.epsilon_db) > 1e-12
        ):
            raise ValueError("all states must share channel, query, and epsilon schema")
    return (
        reference.n_channels,
        reference.epsilon_db,
        reference.queries,
        reference.schema,
    )


def fit_greedy_task_codebook(
    states: Sequence[SpectrumTaskState],
    *,
    max_codewords: int | None = None,
) -> GreedyTaskCodebook:
    """Greedily cover development states with epsilon-safe decoder actions.

    Candidate decoder actions are the distinct exact optimum tuples observed in
    development.  Every development state is therefore coverable when
    ``max_codewords`` is not set.  The greedy objective first maximizes newly
    covered states, then minimizes their worst mean regret, then uses
    lexicographic action order for deterministic tie-breaking.
    """

    state_tuple = tuple(states)
    n_channels, epsilon_db, queries, _ = _validate_compatible_states(state_tuple)
    if max_codewords is not None and int(max_codewords) < 1:
        raise ValueError("max_codewords must be positive when provided")
    candidates = tuple(sorted({state.optimal_actions for state in state_tuple}))
    coverage = {
        actions: frozenset(
            index
            for index, state in enumerate(state_tuple)
            if state.accepts(actions)
        )
        for actions in candidates
    }
    remaining = set(range(len(state_tuple)))
    selected: list[TaskCodeword] = []
    limit = len(candidates) if max_codewords is None else int(max_codewords)

    for codeword_id in count():
        if not remaining or codeword_id >= limit:
            break
        scored = []
        for actions in candidates:
            newly_covered = coverage[actions] & remaining
            if not newly_covered:
                continue
            worst_regrets = [
                state_tuple[index].max_regret_db(actions)
                for index in newly_covered
            ]
            scored.append(
                (
                    -len(newly_covered),
                    float(np.mean(worst_regrets)),
                    actions,
                    newly_covered,
                )
            )
        if not scored:
            break
        _, _, actions, newly_covered = min(scored)
        selected.append(
            TaskCodeword(
                codeword_id=codeword_id,
                decoder_actions=actions,
                training_coverage_count=len(newly_covered),
            )
        )
        remaining.difference_update(newly_covered)

    return GreedyTaskCodebook(
        n_channels=n_channels,
        queries=queries,
        epsilon_db=epsilon_db,
        codewords=tuple(selected),
        training_scene_count=len(state_tuple),
        exact_action_tuple_count=len(candidates),
        training_covered_count=len(state_tuple) - len(remaining),
    )


def exact_action_payload_bits(state: SpectrumTaskState) -> int:
    """Payload needed to transmit every query action without a codebook."""

    width = 0
    for profile in state.profiles:
        action_count = len(profile.allowed_starts)
        width += int(math.ceil(math.log2(action_count)))
    return int(width)


def encode_task_state(
    codebook: GreedyTaskCodebook,
    state: SpectrumTaskState,
) -> CodewordDecision:
    """Select the lowest-regret safe codeword or an exact-action fallback."""

    if (
        state.n_channels != codebook.n_channels
        or state.queries != codebook.queries
        or abs(state.epsilon_db - codebook.epsilon_db) > 1e-12
    ):
        raise ValueError("state does not match codebook schema")
    feasible = []
    for codeword in codebook.codewords:
        regrets = state.regrets_db(codeword.decoder_actions)
        maximum = float(max(regrets, default=0.0))
        if maximum <= codebook.epsilon_db + 1e-12:
            feasible.append((maximum, codeword.codeword_id, codeword, regrets))
    if feasible:
        maximum, _, codeword, regrets = min(feasible)
        return CodewordDecision(
            symbol_id=codeword.codeword_id,
            decoder_actions=codeword.decoder_actions,
            regret_by_query_db=regrets,
            max_regret_db=maximum,
            uses_fallback=False,
            payload_bits=codebook.symbol_width_bits,
        )
    actions = state.optimal_actions
    regrets = state.regrets_db(actions)
    return CodewordDecision(
        symbol_id=codebook.escape_symbol,
        decoder_actions=actions,
        regret_by_query_db=regrets,
        max_regret_db=float(max(regrets, default=0.0)),
        uses_fallback=True,
        payload_bits=codebook.symbol_width_bits + exact_action_payload_bits(state),
    )
