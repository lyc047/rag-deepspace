import pytest

from spectrum_semcom.stage5_state_sync import (
    epoch_is_newer,
    hysteresis_force_refresh,
    must_force_refresh,
    should_accept_ack,
)


def test_epoch_comparison_handles_wrap_and_rejects_stale_duplicates() -> None:
    assert epoch_is_newer(11, 10)
    assert epoch_is_newer(0, 255)
    assert not epoch_is_newer(10, 10)
    assert not epoch_is_newer(9, 10)
    assert should_accept_ack(0, 255, guard_stale_epochs=True)
    assert not should_accept_ack(255, 0, guard_stale_epochs=True)
    assert should_accept_ack(9, 10, guard_stale_epochs=False)


def test_uncertainty_forces_refresh_only_for_recovery_policy() -> None:
    assert must_force_refresh(
        state_uncertain=True,
        force_while_uncertain=True,
    )
    assert not must_force_refresh(
        state_uncertain=True,
        force_while_uncertain=False,
    )
    assert not must_force_refresh(
        state_uncertain=False,
        force_while_uncertain=True,
    )
    with pytest.raises(ValueError):
        epoch_is_newer(256, 0)


def test_hysteresis_uses_count_or_age_guard() -> None:
    assert not hysteresis_force_refresh(
        unconfirmed_transmissions=1,
        force_after_count=2,
        state_age_minutes=80.0,
        max_age_minutes=120.0,
        age_guard_fraction=0.8,
    )
    assert hysteresis_force_refresh(
        unconfirmed_transmissions=2,
        force_after_count=2,
        state_age_minutes=10.0,
        max_age_minutes=120.0,
        age_guard_fraction=0.8,
    )
    assert hysteresis_force_refresh(
        unconfirmed_transmissions=1,
        force_after_count=2,
        state_age_minutes=96.0,
        max_age_minutes=120.0,
        age_guard_fraction=0.8,
    )
    with pytest.raises(ValueError):
        hysteresis_force_refresh(
            unconfirmed_transmissions=0,
            force_after_count=0,
            state_age_minutes=None,
            max_age_minutes=120.0,
            age_guard_fraction=None,
        )
