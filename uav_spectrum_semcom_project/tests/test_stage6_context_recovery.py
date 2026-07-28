import numpy as np
import pytest

from spectrum_semcom.stage6_context_recovery import (
    ReceiverContextHypothesis,
    belief_is_uncertain,
    belief_requires_context_install,
    belief_worst_case_regret_db,
    compact_update_attempt,
    condition_belief_on_cumulative_ack,
    context_install_attempt,
    initial_context_belief,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
)


def test_install_ack_collapses_context_uncertainty() -> None:
    initial = initial_context_belief()
    uncertain = context_install_attempt(
        initial, codebook_epoch=3, ack_received=False
    )
    assert belief_is_uncertain(uncertain)
    assert belief_requires_context_install(uncertain, codebook_epoch=3)
    confirmed = context_install_attempt(
        uncertain, codebook_epoch=3, ack_received=True
    )
    assert not belief_is_uncertain(confirmed)
    assert not belief_requires_context_install(confirmed, codebook_epoch=3)


def test_update_ack_filters_lost_or_delivered_hypotheses() -> None:
    installed = context_install_attempt(
        initial_context_belief(), codebook_epoch=3, ack_received=True
    )
    uncertain = compact_update_attempt(
        installed,
        codebook_epoch=3,
        update_epoch=7,
        decoder_actions=(1,),
        success_timestamp="2023-01-01T00:00:00",
        ack_received=False,
        include_context_loss_on_no_ack=False,
    )
    assert belief_is_uncertain(uncertain)
    filtered, confirmed, accepted = condition_belief_on_cumulative_ack(
        uncertain,
        ack_epoch=7,
        confirmed_epoch=None,
        guard_stale_epochs=True,
    )
    assert accepted
    assert confirmed == 7
    assert {value.decoder_actions for value in filtered} == {(1,)}


def test_stale_duplicate_ack_does_not_roll_back_belief() -> None:
    current = frozenset(
        {
            ReceiverContextHypothesis(
                True,
                3,
                8,
                (2,),
                "2023-01-01T00:01:00",
            )
        }
    )
    filtered, confirmed, accepted = condition_belief_on_cumulative_ack(
        current,
        ack_epoch=7,
        confirmed_epoch=8,
        guard_stale_epochs=True,
    )
    assert not accepted
    assert confirmed == 8
    assert filtered == current


def test_worst_case_regret_forces_recovery_only_when_material() -> None:
    task = build_task_state(
        np.array([-100.0, -99.9, -95.0]),
        (SpectrumTaskQuery(1),),
        epsilon_db=0.2,
    )
    safe = frozenset(
        {
            ReceiverContextHypothesis(
                True, 3, 1, (0,), "2023-01-01T00:00:00"
            ),
            ReceiverContextHypothesis(
                True, 3, 2, (1,), "2023-01-01T00:01:00"
            ),
        }
    )
    assert belief_worst_case_regret_db(
        task, safe, codebook_epoch=3
    ) <= 0.2
    unsafe = safe | {
        ReceiverContextHypothesis(
            True, 3, 3, (2,), "2023-01-01T00:02:00"
        )
    }
    assert belief_worst_case_regret_db(
        task, unsafe, codebook_epoch=3
    ) > 0.2
    assert belief_worst_case_regret_db(
        task, initial_context_belief(), codebook_epoch=3
    ) == float("inf")


def test_missing_ack_can_include_receiver_context_loss() -> None:
    installed = context_install_attempt(
        initial_context_belief(), codebook_epoch=3, ack_received=True
    )
    belief = compact_update_attempt(
        installed,
        codebook_epoch=3,
        update_epoch=1,
        decoder_actions=(0,),
        success_timestamp="2023-01-01T00:00:00",
        ack_received=False,
        include_context_loss_on_no_ack=True,
    )
    assert belief_requires_context_install(belief, codebook_epoch=3)


def test_invalid_hypothesis_fails_closed() -> None:
    with pytest.raises(ValueError):
        ReceiverContextHypothesis(False, 1, None, None, None)
