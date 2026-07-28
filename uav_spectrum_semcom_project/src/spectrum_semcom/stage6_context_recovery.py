"""Belief-set primitives for Stage-6 context and ACK recovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from spectrum_semcom.stage5_state_sync import (
    epoch_is_newer,
    should_accept_ack,
)
from spectrum_semcom.stage6_task_codebook import SpectrumTaskState


@dataclass(frozen=True)
class ReceiverContextHypothesis:
    context_installed: bool
    codebook_epoch: int | None
    update_epoch: int | None
    decoder_actions: tuple[int, ...] | None
    success_timestamp: str | None

    def __post_init__(self) -> None:
        if self.context_installed:
            if self.codebook_epoch is None:
                raise ValueError("installed hypothesis requires codebook epoch")
        elif any(
            value is not None
            for value in (
                self.codebook_epoch,
                self.update_epoch,
                self.decoder_actions,
                self.success_timestamp,
            )
        ):
            raise ValueError("uninstalled hypothesis cannot carry semantic state")
        if self.codebook_epoch is not None and not 0 <= self.codebook_epoch <= 255:
            raise ValueError("codebook epoch must fit eight bits")
        if self.update_epoch is not None and not 0 <= self.update_epoch <= 255:
            raise ValueError("update epoch must fit eight bits")
        if (self.decoder_actions is None) != (self.success_timestamp is None):
            raise ValueError("actions and success timestamp must coexist")
        if self.success_timestamp is not None:
            datetime.fromisoformat(self.success_timestamp)


ContextBelief = frozenset[ReceiverContextHypothesis]


def initial_context_belief() -> ContextBelief:
    return frozenset(
        {
            ReceiverContextHypothesis(
                context_installed=False,
                codebook_epoch=None,
                update_epoch=None,
                decoder_actions=None,
                success_timestamp=None,
            )
        }
    )


def context_install_attempt(
    belief: Iterable[ReceiverContextHypothesis],
    *,
    codebook_epoch: int,
    ack_received: bool,
) -> ContextBelief:
    """Propagate delivery/ACK ambiguity for an idempotent context install."""

    hypotheses = frozenset(belief)
    if not hypotheses or not 0 <= int(codebook_epoch) <= 255:
        raise ValueError("invalid context-install belief")
    delivered = set()
    for hypothesis in hypotheses:
        same_context = (
            hypothesis.context_installed
            and hypothesis.codebook_epoch == int(codebook_epoch)
        )
        delivered.add(
            ReceiverContextHypothesis(
                context_installed=True,
                codebook_epoch=int(codebook_epoch),
                update_epoch=hypothesis.update_epoch if same_context else None,
                decoder_actions=(
                    hypothesis.decoder_actions if same_context else None
                ),
                success_timestamp=(
                    hypothesis.success_timestamp if same_context else None
                ),
            )
        )
    if ack_received:
        return frozenset(delivered)
    return frozenset(set(hypotheses) | delivered)


def compact_update_attempt(
    belief: Iterable[ReceiverContextHypothesis],
    *,
    codebook_epoch: int,
    update_epoch: int,
    decoder_actions: tuple[int, ...],
    success_timestamp: str,
    ack_received: bool,
    include_context_loss_on_no_ack: bool,
) -> ContextBelief:
    """Propagate data-loss versus ACK-loss ambiguity for one task update."""

    hypotheses = frozenset(belief)
    datetime.fromisoformat(success_timestamp)
    if (
        not hypotheses
        or not decoder_actions
        or not 0 <= int(codebook_epoch) <= 255
        or not 0 <= int(update_epoch) <= 255
    ):
        raise ValueError("invalid compact-update belief")
    delivered = {
        ReceiverContextHypothesis(
            context_installed=True,
            codebook_epoch=int(codebook_epoch),
            update_epoch=int(update_epoch),
            decoder_actions=tuple(int(value) for value in decoder_actions),
            success_timestamp=success_timestamp,
        )
        for hypothesis in hypotheses
        if hypothesis.context_installed
        and hypothesis.codebook_epoch == int(codebook_epoch)
    }
    if ack_received:
        if not delivered:
            raise ValueError("ACK cannot confirm an undecodable task update")
        return frozenset(delivered)
    possible = set(hypotheses) | delivered
    if include_context_loss_on_no_ack:
        possible.update(initial_context_belief())
    return frozenset(possible)


def condition_belief_on_cumulative_ack(
    belief: Iterable[ReceiverContextHypothesis],
    *,
    ack_epoch: int,
    confirmed_epoch: int | None,
    guard_stale_epochs: bool,
) -> tuple[ContextBelief, int | None, bool]:
    """Filter plausible current states using a cumulative update ACK."""

    hypotheses = frozenset(belief)
    if not hypotheses:
        raise ValueError("context belief cannot be empty")
    accepted = should_accept_ack(
        ack_epoch,
        confirmed_epoch,
        guard_stale_epochs=guard_stale_epochs,
    )
    if not accepted:
        return hypotheses, confirmed_epoch, False
    compatible = frozenset(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.context_installed
        and hypothesis.update_epoch is not None
        and (
            hypothesis.update_epoch == int(ack_epoch)
            or epoch_is_newer(hypothesis.update_epoch, int(ack_epoch))
        )
    )
    if not compatible:
        return hypotheses, confirmed_epoch, False
    return compatible, int(ack_epoch), True


def belief_requires_context_install(
    belief: Iterable[ReceiverContextHypothesis],
    *,
    codebook_epoch: int,
) -> bool:
    hypotheses = frozenset(belief)
    if not hypotheses:
        raise ValueError("context belief cannot be empty")
    return any(
        not hypothesis.context_installed
        or hypothesis.codebook_epoch != int(codebook_epoch)
        for hypothesis in hypotheses
    )


def belief_worst_case_regret_db(
    task_state: SpectrumTaskState,
    belief: Iterable[ReceiverContextHypothesis],
    *,
    codebook_epoch: int,
) -> float:
    """Return infinity when any plausible receiver cannot execute the task."""

    hypotheses = frozenset(belief)
    if not hypotheses:
        raise ValueError("context belief cannot be empty")
    regrets = []
    for hypothesis in hypotheses:
        if (
            not hypothesis.context_installed
            or hypothesis.codebook_epoch != int(codebook_epoch)
            or hypothesis.decoder_actions is None
            or len(hypothesis.decoder_actions) != len(task_state.profiles)
        ):
            return float("inf")
        regrets.append(task_state.max_regret_db(hypothesis.decoder_actions))
    return float(max(regrets, default=float("inf")))


def belief_maximum_age_minutes(
    belief: Iterable[ReceiverContextHypothesis],
    *,
    timestamp_local: str,
    codebook_epoch: int,
) -> float:
    current = datetime.fromisoformat(timestamp_local)
    ages = []
    for hypothesis in frozenset(belief):
        if (
            not hypothesis.context_installed
            or hypothesis.codebook_epoch != int(codebook_epoch)
            or hypothesis.success_timestamp is None
        ):
            return float("inf")
        age = (
            current - datetime.fromisoformat(hypothesis.success_timestamp)
        ).total_seconds() / 60.0
        if age < 0:
            raise ValueError("current timestamp precedes belief state")
        ages.append(age)
    return float(max(ages, default=float("inf")))


def belief_is_uncertain(
    belief: Iterable[ReceiverContextHypothesis],
) -> bool:
    return len(frozenset(belief)) > 1
