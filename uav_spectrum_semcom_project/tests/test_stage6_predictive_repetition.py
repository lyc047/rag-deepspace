import numpy as np

from spectrum_semcom.stage6_predictive_repetition import (
    simulate_predictive_repetition,
)
from spectrum_semcom.stage6_context_codec import (
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook


def _fixture():
    powers = np.asarray(
        [
            [-100.0, -90.0, -80.0],
            [-80.0, -90.0, -100.0],
        ]
    )
    query = SpectrumTaskQuery(1)
    states = [
        build_task_state(row, (query,), epsilon_db=0.2)
        for row in powers
    ]
    session = install_codebook(
        fit_greedy_task_codebook(states), epoch=1
    )
    return powers, states, session


def test_duplicate_rescues_lost_primary_without_changing_action_rule() -> None:
    powers, states, session = _fixture()
    result = simulate_predictive_repetition(
        states,
        np.asarray(
            ["2023-01-01T00:00:00", "2023-01-01T00:00:10"]
        ),
        np.asarray(["a", "a"]),
        np.asarray([0, 1]),
        session=session,
        protection_candidates=np.asarray([True, True]),
        maximum_reservations=2,
        reservation_equivalent_bits=48,
        task_loss_probability=0.5,
        random_values=np.asarray([[0.1, 0.9], [0.1, 0.9]]),
        max_age_minutes=120.0,
        outage_penalty_db=10.0,
    )
    assert result.protected_update_count == 2
    assert result.failed_update_count == 0
    assert result.clean == (True, True)
    assert result.extra_repetition_bits > 0


def test_no_candidate_never_spends_repetition_budget() -> None:
    _, states, session = _fixture()
    result = simulate_predictive_repetition(
        states,
        np.asarray(
            ["2023-01-01T00:00:00", "2023-01-01T00:00:10"]
        ),
        np.asarray(["a", "a"]),
        np.asarray([0, 1]),
        session=session,
        protection_candidates=np.asarray([False, False]),
        maximum_reservations=2,
        reservation_equivalent_bits=48,
        task_loss_probability=0.5,
        random_values=np.asarray([[0.9, 0.9], [0.9, 0.9]]),
        max_age_minutes=120.0,
        outage_penalty_db=10.0,
    )
    assert result.protected_update_count == 0
    assert result.extra_repetition_bits == 0


def test_protection_budget_is_a_hard_cap() -> None:
    _, states, session = _fixture()
    result = simulate_predictive_repetition(
        states,
        np.asarray(
            ["2023-01-01T00:00:00", "2023-01-01T00:00:10"]
        ),
        np.asarray(["a", "a"]),
        np.asarray([0, 1]),
        session=session,
        protection_candidates=np.asarray([True, True]),
        maximum_reservations=1,
        reservation_equivalent_bits=48,
        task_loss_probability=0.5,
        random_values=np.asarray([[0.9, 0.9], [0.9, 0.9]]),
        max_age_minutes=120.0,
        outage_penalty_db=10.0,
    )
    assert result.protected_update_count == 1
    assert result.reservation_count == 1
    assert result.reserved_capacity_bits == 48


def test_worst_case_compact_length_covers_every_real_update() -> None:
    _, states, session = _fixture()
    maximum = maximum_compact_update_bits(session)
    from spectrum_semcom.stage6_context_codec import encode_compact_update

    assert all(
        encode_compact_update(
            state, session, node_id=1, update_epoch=index
        )[0].size
        <= maximum
        for index, state in enumerate(states)
    )
