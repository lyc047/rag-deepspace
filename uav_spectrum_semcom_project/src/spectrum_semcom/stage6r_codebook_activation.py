"""Versioned 64-bit activation for preinstalled Stage-6R codebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spectrum_semcom.stage6_task_codebook import GreedyTaskCodebook
from spectrum_semcom.stage6_task_codec import (
    CodebookSession,
    codebook_manifest_sha256,
    install_codebook,
)


_MAGIC = 0xEC
_VERSION = 1
_FRAME_BITS = 64


@dataclass(frozen=True)
class PreinstalledCodebookEntry:
    bank_id: int
    codebook: GreedyTaskCodebook
    manifest_sha256: str

    def __post_init__(self) -> None:
        expected = codebook_manifest_sha256(self.codebook)
        if (
            not 0 <= int(self.bank_id) <= 255
            or self.manifest_sha256 != expected
        ):
            raise ValueError("invalid preinstalled codebook entry")

    @property
    def packet_tag(self) -> int:
        return int(self.manifest_sha256[:4], 16)


@dataclass(frozen=True)
class PreinstalledCodebookCatalog:
    catalog_epoch: int
    entries: tuple[PreinstalledCodebookEntry, ...]

    def __post_init__(self) -> None:
        bank_ids = tuple(int(entry.bank_id) for entry in self.entries)
        if (
            not 0 <= int(self.catalog_epoch) <= 255
            or not self.entries
            or len(set(bank_ids)) != len(bank_ids)
        ):
            raise ValueError("invalid preinstalled codebook catalog")

    def entry(self, bank_id: int) -> PreinstalledCodebookEntry:
        matches = [
            entry
            for entry in self.entries
            if int(entry.bank_id) == int(bank_id)
        ]
        if len(matches) != 1:
            raise ValueError("bank id is not present in the catalog")
        return matches[0]


@dataclass(frozen=True)
class CodebookActivation:
    node_id: int
    catalog_epoch: int
    bank_id: int
    codebook_epoch: int
    codebook_tag: int
    activation_epoch: int


@dataclass(frozen=True)
class ResolvedCodebookActivation:
    message: CodebookActivation
    session: CodebookSession


@dataclass(frozen=True)
class ActivationHandshakeState:
    """Sender/receiver handshake state before full reliability integration."""

    phase: str
    activation_attempt_count: int
    sender_context_confirmed: bool
    receiver_context_active: bool

    def __post_init__(self) -> None:
        if (
            self.phase not in {"activation", "full_install", "ready"}
            or int(self.activation_attempt_count) < 0
            or (
                self.phase == "ready"
                and not self.sender_context_confirmed
            )
            or (
                self.sender_context_confirmed
                and self.phase != "ready"
            )
        ):
            raise ValueError("invalid activation handshake state")


@dataclass(frozen=True)
class ActivationHandshakeTransition:
    state: ActivationHandshakeState
    ack_emitted: bool
    catalog_rejected: bool
    full_install_fallback_triggered: bool


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit activation field")
    return np.asarray(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


def build_preinstalled_catalog(
    codebooks: Iterable[tuple[int, GreedyTaskCodebook]],
    *,
    catalog_epoch: int,
) -> PreinstalledCodebookCatalog:
    entries = tuple(
        PreinstalledCodebookEntry(
            bank_id=int(bank_id),
            codebook=codebook,
            manifest_sha256=codebook_manifest_sha256(codebook),
        )
        for bank_id, codebook in codebooks
    )
    return PreinstalledCodebookCatalog(
        catalog_epoch=int(catalog_epoch),
        entries=entries,
    )


def encode_codebook_activation(
    catalog: PreinstalledCodebookCatalog,
    *,
    node_id: int,
    bank_id: int,
    codebook_epoch: int,
    activation_epoch: int,
) -> np.ndarray:
    if (
        not 0 <= int(node_id) <= 63
        or not 0 <= int(codebook_epoch) <= 255
        or not 0 <= int(activation_epoch) <= 255
    ):
        raise ValueError("activation field is outside the fixed schema")
    entry = catalog.entry(bank_id)
    frame = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 2),
            _uint_to_bits(node_id, 6),
            _uint_to_bits(catalog.catalog_epoch, 8),
            _uint_to_bits(entry.bank_id, 8),
            _uint_to_bits(codebook_epoch, 8),
            _uint_to_bits(entry.packet_tag, 16),
            _uint_to_bits(activation_epoch, 8),
        ]
    )
    if int(frame.size) != _FRAME_BITS:
        raise ValueError("activation frame width invariant failed")
    return frame


def decode_codebook_activation(
    bits: np.ndarray,
    catalog: PreinstalledCodebookCatalog,
) -> ResolvedCodebookActivation:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if (
        data.size != _FRAME_BITS
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError("invalid codebook activation bitstream")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if take(8) != _MAGIC or take(2) != _VERSION:
        raise ValueError("invalid codebook activation identity")
    message = CodebookActivation(
        node_id=take(6),
        catalog_epoch=take(8),
        bank_id=take(8),
        codebook_epoch=take(8),
        codebook_tag=take(16),
        activation_epoch=take(8),
    )
    if message.catalog_epoch != int(catalog.catalog_epoch):
        raise ValueError("preinstalled catalog epoch mismatch")
    entry = catalog.entry(message.bank_id)
    if message.codebook_tag != entry.packet_tag:
        raise ValueError("preinstalled codebook tag mismatch")
    return ResolvedCodebookActivation(
        message=message,
        session=install_codebook(
            entry.codebook,
            epoch=message.codebook_epoch,
        ),
    )


def codebook_activation_frame_bits() -> int:
    return _FRAME_BITS


def initial_activation_handshake(
    *,
    activation_eligible: bool,
) -> ActivationHandshakeState:
    return ActivationHandshakeState(
        phase="activation" if activation_eligible else "full_install",
        activation_attempt_count=0,
        sender_context_confirmed=False,
        receiver_context_active=False,
    )


def activation_handshake_attempt(
    state: ActivationHandshakeState,
    *,
    delivered: bool,
    catalog_resolved: bool,
    ack_delivered: bool,
    maximum_activation_attempts: int,
) -> ActivationHandshakeTransition:
    """Apply one activation attempt with fail-closed ACK semantics."""

    if (
        state.phase != "activation"
        or int(maximum_activation_attempts) < 1
        or (ack_delivered and not (delivered and catalog_resolved))
    ):
        raise ValueError("invalid activation handshake attempt")
    accepted = bool(delivered and catalog_resolved)
    attempts = int(state.activation_attempt_count) + 1
    if accepted and ack_delivered:
        next_state = ActivationHandshakeState(
            phase="ready",
            activation_attempt_count=attempts,
            sender_context_confirmed=True,
            receiver_context_active=True,
        )
        fallback = False
    else:
        fallback = attempts >= int(maximum_activation_attempts)
        next_state = ActivationHandshakeState(
            phase="full_install" if fallback else "activation",
            activation_attempt_count=attempts,
            sender_context_confirmed=False,
            receiver_context_active=(
                state.receiver_context_active or accepted
            ),
        )
    return ActivationHandshakeTransition(
        state=next_state,
        ack_emitted=accepted,
        catalog_rejected=bool(delivered and not catalog_resolved),
        full_install_fallback_triggered=fallback,
    )


def full_install_handshake_attempt(
    state: ActivationHandshakeState,
    *,
    delivered: bool,
    ack_delivered: bool,
) -> ActivationHandshakeTransition:
    if (
        state.phase != "full_install"
        or (ack_delivered and not delivered)
    ):
        raise ValueError("invalid full-install handshake attempt")
    if delivered and ack_delivered:
        next_state = ActivationHandshakeState(
            phase="ready",
            activation_attempt_count=state.activation_attempt_count,
            sender_context_confirmed=True,
            receiver_context_active=True,
        )
    else:
        next_state = ActivationHandshakeState(
            phase="full_install",
            activation_attempt_count=state.activation_attempt_count,
            sender_context_confirmed=False,
            receiver_context_active=(
                state.receiver_context_active or bool(delivered)
            ),
        )
    return ActivationHandshakeTransition(
        state=next_state,
        ack_emitted=bool(delivered),
        catalog_rejected=False,
        full_install_fallback_triggered=False,
    )


def request_context_recovery(
    state: ActivationHandshakeState,
    *,
    activation_eligible: bool,
    receiver_context_active: bool,
) -> ActivationHandshakeState:
    """Enter recovery after belief/heartbeat detects uncertain context."""

    return ActivationHandshakeState(
        phase="activation" if activation_eligible else "full_install",
        activation_attempt_count=0,
        sender_context_confirmed=False,
        receiver_context_active=bool(receiver_context_active),
    )
