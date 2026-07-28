import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    exact_action_payload_bits,
    fit_greedy_task_codebook,
)


def test_task_profile_exposes_epsilon_optimal_actions_and_regret() -> None:
    state = build_task_state(
        np.array([-100.0, -99.9, -95.0]),
        (SpectrumTaskQuery(1),),
        epsilon_db=0.2,
    )
    profile = state.profiles[0]
    assert profile.optimum_start == 0
    assert profile.epsilon_optimal_starts == (0, 1)
    assert profile.regret_db(0) == 0.0
    assert profile.regret_db(1) == pytest.approx(0.1)
    assert profile.regret_db(2) == pytest.approx(5.0)


def test_greedy_codebook_groups_different_exact_optima_safely() -> None:
    query = (SpectrumTaskQuery(1),)
    states = [
        build_task_state(np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2),
        build_task_state(np.array([-99.9, -100.0, -95.0]), query, epsilon_db=0.2),
        build_task_state(np.array([-95.0, -95.0, -100.0]), query, epsilon_db=0.2),
    ]
    assert len({state.optimal_actions for state in states}) == 3
    codebook = fit_greedy_task_codebook(states)
    assert codebook.exact_action_tuple_count == 3
    assert len(codebook.codewords) == 2
    assert codebook.training_covered_count == 3
    decisions = [encode_task_state(codebook, state) for state in states]
    assert all(not decision.uses_fallback for decision in decisions)
    assert max(decision.max_regret_db for decision in decisions) <= 0.2


def test_unseen_state_uses_explicit_exact_fallback() -> None:
    query = (SpectrumTaskQuery(1),)
    fitted = [
        build_task_state(np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2),
        build_task_state(np.array([-99.9, -100.0, -95.0]), query, epsilon_db=0.2),
    ]
    codebook = fit_greedy_task_codebook(fitted)
    unseen = build_task_state(
        np.array([-95.0, -95.0, -100.0]),
        query,
        epsilon_db=0.2,
    )
    decision = encode_task_state(codebook, unseen)
    assert decision.uses_fallback
    assert decision.decoder_actions == unseen.optimal_actions
    assert decision.max_regret_db == 0.0
    assert decision.payload_bits == (
        codebook.symbol_width_bits + exact_action_payload_bits(unseen)
    )


def test_multi_query_allowed_start_constraints_are_enforced() -> None:
    queries = (
        SpectrumTaskQuery(1, allowed_starts=(0, 2)),
        SpectrumTaskQuery(2),
    )
    state = build_task_state(
        np.array([-100.0, -90.0, -99.0, -80.0]),
        queries,
        epsilon_db=1.5,
    )
    assert state.profiles[0].epsilon_optimal_starts == (0, 2)
    assert state.accepts((2, state.profiles[1].optimum_start))
    assert not state.accepts((1, state.profiles[1].optimum_start))


def test_invalid_or_mismatched_schemas_fail_closed() -> None:
    with pytest.raises(ValueError):
        SpectrumTaskQuery(0)
    with pytest.raises(ValueError):
        SpectrumTaskQuery(1, allowed_starts=(2, 1))
    with pytest.raises(ValueError):
        build_task_state(
            np.array([-100.0, np.nan]),
            (SpectrumTaskQuery(1),),
            epsilon_db=0.2,
        )
    first = build_task_state(
        np.array([-100.0, -99.0]),
        (SpectrumTaskQuery(1),),
        epsilon_db=0.2,
    )
    second = build_task_state(
        np.array([-100.0, -99.0]),
        (SpectrumTaskQuery(2),),
        epsilon_db=0.2,
    )
    with pytest.raises(ValueError, match="schema"):
        fit_greedy_task_codebook((first, second))
