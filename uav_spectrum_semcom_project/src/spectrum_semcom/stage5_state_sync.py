"""ACK epoch and uncertainty primitives for Stage-5 state recovery."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from spectrum_semcom.stage5_event_semantics import block_regret


def epoch_is_newer(candidate: int, reference: int, *, modulus: int = 256) -> bool:
    """RFC-1982-style serial comparison for an even finite epoch space."""
    if (
        modulus < 4
        or modulus % 2
        or not 0 <= int(candidate) < modulus
        or not 0 <= int(reference) < modulus
    ):
        raise ValueError("invalid epoch or modulus")
    distance = (int(candidate) - int(reference)) % modulus
    return 0 < distance < modulus // 2


def should_accept_ack(
    candidate_epoch: int,
    confirmed_epoch: int | None,
    *,
    guard_stale_epochs: bool,
) -> bool:
    """Accept the first ACK; optionally reject duplicate or stale epochs."""
    if not 0 <= int(candidate_epoch) <= 255:
        raise ValueError("ACK epoch must fit eight bits")
    if confirmed_epoch is None:
        return True
    if not 0 <= int(confirmed_epoch) <= 255:
        raise ValueError("confirmed epoch must fit eight bits")
    if not guard_stale_epochs:
        return True
    return epoch_is_newer(candidate_epoch, confirmed_epoch)


def must_force_refresh(
    *,
    state_uncertain: bool,
    force_while_uncertain: bool,
) -> bool:
    return bool(state_uncertain and force_while_uncertain)


def belief_worst_case_force_refresh(
    *,
    state_uncertain: bool,
    plausible_selected_starts: Iterable[int],
    costs_dbm: np.ndarray,
    regret_threshold_db: float,
) -> bool:
    """Refresh only when ACK uncertainty can change the task outcome materially.

    The sender does not need to know whether the data frame or only its ACK was
    lost.  It keeps every receiver state still compatible with its observations
    and retransmits when the worst plausible reuse regret exceeds the already
    frozen task threshold.
    """
    costs = np.asarray(costs_dbm, dtype=np.float64).reshape(-1)
    starts = tuple(int(start) for start in plausible_selected_starts)
    if (
        costs.size < 1
        or not np.all(np.isfinite(costs))
        or float(regret_threshold_db) < 0
        or any(start < 0 or start >= costs.size for start in starts)
    ):
        raise ValueError("invalid belief-risk recovery input")
    if not state_uncertain:
        return False
    if not starts:
        return True
    worst_case_regret = max(block_regret(costs, start) for start in starts)
    return bool(worst_case_regret > float(regret_threshold_db))


def hysteresis_force_refresh(
    *,
    unconfirmed_transmissions: int,
    force_after_count: int | None,
    state_age_minutes: float | None,
    max_age_minutes: float,
    age_guard_fraction: float | None,
) -> bool:
    """Decide whether ACK uncertainty has crossed count or age hysteresis."""
    if unconfirmed_transmissions < 0 or max_age_minutes <= 0:
        raise ValueError("invalid uncertainty count or maximum age")
    if force_after_count is not None and force_after_count < 1:
        raise ValueError("force_after_count must be positive")
    if age_guard_fraction is not None and not 0 < age_guard_fraction <= 1:
        raise ValueError("age_guard_fraction must lie in (0, 1]")
    count_trigger = bool(
        force_after_count is not None
        and unconfirmed_transmissions >= force_after_count
    )
    age_trigger = bool(
        unconfirmed_transmissions > 0
        and age_guard_fraction is not None
        and state_age_minutes is not None
        and state_age_minutes
        >= float(max_age_minutes) * float(age_guard_fraction)
    )
    return count_trigger or age_trigger
