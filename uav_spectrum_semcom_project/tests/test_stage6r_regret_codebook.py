import math

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
)
from spectrum_semcom.stage6r_regret_codebook import (
    admissible_union_candidates,
    campaign_balanced_weights,
    evaluate_selected_actions,
    exhaustive_action_tuples,
    greedy_select_actions,
    optimal_action_candidates,
    regret_matrix,
)


def _states() -> list:
    queries = (SpectrumTaskQuery(1), SpectrumTaskQuery(2))
    powers = (
        np.asarray([0.0, 0.1, 2.0, 3.0]),
        np.asarray([0.1, 0.0, 2.0, 3.0]),
        np.asarray([3.0, 2.0, 0.0, 0.1]),
    )
    return [
        build_task_state(value, queries, epsilon_db=0.25)
        for value in powers
    ]


def test_candidate_sets_expand_monotonically() -> None:
    states = _states()
    optimal = set(optimal_action_candidates(states))
    admissible = set(admissible_union_candidates(states))
    exhaustive = set(exhaustive_action_tuples(states))
    assert optimal <= admissible <= exhaustive
    assert len(exhaustive) == 4 * 3


def test_campaign_balanced_weights_sum_to_campaign_count() -> None:
    weights = campaign_balanced_weights(["a", "a", "a", "b"])
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1])
    assert math.isclose(float(np.sum(weights)), 2.0)


def test_greedy_selection_and_safe_evaluation() -> None:
    states = _states()
    actions = exhaustive_action_tuples(states)
    regrets = regret_matrix(states, actions)
    selected = greedy_select_actions(
        regrets=regrets,
        actions=actions,
        epsilon_db=0.25,
        weights=np.ones(len(states)),
        max_codewords=2,
    )
    indices = [actions.index(value) for value in selected.actions]
    result = evaluate_selected_actions(
        regrets=regrets,
        selected_candidate_indices=indices,
        epsilon_db=0.25,
        exact_action_payload_bits=4,
        cvar_alpha=0.9,
    )
    assert result["coverage_rate"] == 1.0
    assert result["unsafe_coded_action_count"] == 0
    assert result["escape_count"] == 0
    assert result["symbol_width_bits"] == 2


def test_regret_matrix_rejects_out_of_range_action() -> None:
    with np.testing.assert_raises_regex(ValueError, "outside"):
        regret_matrix(_states(), [(99, 0)])
