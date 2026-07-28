import numpy as np
import pytest

from spectrum_semcom.stage5_state_sync import belief_worst_case_force_refresh


def test_belief_recovery_ignores_task_immaterial_uncertainty() -> None:
    costs = np.asarray([-90.0, -89.9, -88.0])
    assert not belief_worst_case_force_refresh(
        state_uncertain=True,
        plausible_selected_starts=[0, 1],
        costs_dbm=costs,
        regret_threshold_db=0.2,
    )


def test_belief_recovery_refreshes_on_worst_case_task_risk() -> None:
    costs = np.asarray([-90.0, -89.9, -88.0])
    assert belief_worst_case_force_refresh(
        state_uncertain=True,
        plausible_selected_starts=[0, 2],
        costs_dbm=costs,
        regret_threshold_db=0.2,
    )
    assert not belief_worst_case_force_refresh(
        state_uncertain=False,
        plausible_selected_starts=[2],
        costs_dbm=costs,
        regret_threshold_db=0.2,
    )


def test_belief_recovery_fails_closed_for_empty_or_invalid_belief() -> None:
    assert belief_worst_case_force_refresh(
        state_uncertain=True,
        plausible_selected_starts=[],
        costs_dbm=np.asarray([-90.0, -89.0]),
        regret_threshold_db=0.2,
    )
    with pytest.raises(ValueError):
        belief_worst_case_force_refresh(
            state_uncertain=True,
            plausible_selected_starts=[2],
            costs_dbm=np.asarray([-90.0, -89.0]),
            regret_threshold_db=0.2,
        )
