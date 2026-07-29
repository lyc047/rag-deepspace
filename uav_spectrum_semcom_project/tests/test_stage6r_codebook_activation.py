from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6r_codebook_activation import (
    PreinstalledCodebookCatalog,
    activation_handshake_attempt,
    build_preinstalled_catalog,
    codebook_activation_frame_bits,
    decode_codebook_activation,
    encode_codebook_activation,
    full_install_handshake_attempt,
    initial_activation_handshake,
    request_context_recovery,
)


def _codebook(offset: float = 0.0):
    queries = (SpectrumTaskQuery(1), SpectrumTaskQuery(2))
    rows = [
        np.asarray([-90.0 + offset, -89.0, -80.0, -79.0]),
        np.asarray([-91.0 + offset, -90.0, -80.0, -79.0]),
    ]
    states = [
        build_task_state(row, queries, epsilon_db=0.2) for row in rows
    ]
    return fit_greedy_task_codebook(states)


def test_activation_round_trip_is_exactly_64_bits() -> None:
    catalog = build_preinstalled_catalog(
        [(7, _codebook())],
        catalog_epoch=4,
    )
    frame = encode_codebook_activation(
        catalog,
        node_id=3,
        bank_id=7,
        codebook_epoch=9,
        activation_epoch=11,
    )
    decoded = decode_codebook_activation(frame, catalog)
    assert frame.size == codebook_activation_frame_bits() == 64
    assert decoded.message.node_id == 3
    assert decoded.message.catalog_epoch == 4
    assert decoded.message.bank_id == 7
    assert decoded.message.codebook_epoch == 9
    assert decoded.message.activation_epoch == 11
    assert decoded.session.codebook == catalog.entry(7).codebook


def test_wrong_catalog_epoch_is_rejected() -> None:
    sender = build_preinstalled_catalog([(7, _codebook())], catalog_epoch=4)
    receiver = PreinstalledCodebookCatalog(
        catalog_epoch=5,
        entries=sender.entries,
    )
    frame = encode_codebook_activation(
        sender,
        node_id=3,
        bank_id=7,
        codebook_epoch=9,
        activation_epoch=11,
    )
    with pytest.raises(ValueError, match="catalog epoch mismatch"):
        decode_codebook_activation(frame, receiver)


def test_unknown_bank_id_is_rejected() -> None:
    sender = build_preinstalled_catalog([(7, _codebook())], catalog_epoch=4)
    receiver = build_preinstalled_catalog([(8, _codebook())], catalog_epoch=4)
    frame = encode_codebook_activation(
        sender,
        node_id=3,
        bank_id=7,
        codebook_epoch=9,
        activation_epoch=11,
    )
    with pytest.raises(ValueError, match="bank id"):
        decode_codebook_activation(frame, receiver)


def test_wrong_codebook_tag_is_rejected() -> None:
    sender = build_preinstalled_catalog([(7, _codebook())], catalog_epoch=4)
    receiver = build_preinstalled_catalog(
        [(7, _codebook(offset=20.0))],
        catalog_epoch=4,
    )
    frame = encode_codebook_activation(
        sender,
        node_id=3,
        bank_id=7,
        codebook_epoch=9,
        activation_epoch=11,
    )
    with pytest.raises(ValueError, match="codebook tag mismatch"):
        decode_codebook_activation(frame, receiver)


def test_duplicate_bank_id_is_rejected() -> None:
    codebook = _codebook()
    with pytest.raises(ValueError, match="invalid preinstalled"):
        build_preinstalled_catalog(
            [(7, codebook), (7, codebook)],
            catalog_epoch=4,
        )


def test_missing_activation_ack_retries_then_falls_back() -> None:
    state = initial_activation_handshake(activation_eligible=True)
    first = activation_handshake_attempt(
        state,
        delivered=True,
        catalog_resolved=True,
        ack_delivered=False,
        maximum_activation_attempts=2,
    )
    assert first.ack_emitted
    assert first.state.phase == "activation"
    assert first.state.receiver_context_active
    second = activation_handshake_attempt(
        first.state,
        delivered=True,
        catalog_resolved=True,
        ack_delivered=False,
        maximum_activation_attempts=2,
    )
    assert second.full_install_fallback_triggered
    assert second.state.phase == "full_install"
    installed = full_install_handshake_attempt(
        second.state,
        delivered=True,
        ack_delivered=True,
    )
    assert installed.state.phase == "ready"
    assert installed.state.sender_context_confirmed


def test_catalog_mismatch_never_emits_ack() -> None:
    state = initial_activation_handshake(activation_eligible=True)
    outcome = activation_handshake_attempt(
        state,
        delivered=True,
        catalog_resolved=False,
        ack_delivered=False,
        maximum_activation_attempts=2,
    )
    assert outcome.catalog_rejected
    assert not outcome.ack_emitted
    assert not outcome.state.receiver_context_active


def test_recovery_resets_attempt_budget_but_can_keep_receiver_context() -> None:
    state = initial_activation_handshake(activation_eligible=True)
    ready = activation_handshake_attempt(
        state,
        delivered=True,
        catalog_resolved=True,
        ack_delivered=True,
        maximum_activation_attempts=2,
    ).state
    recovery = request_context_recovery(
        ready,
        activation_eligible=True,
        receiver_context_active=True,
    )
    assert recovery.phase == "activation"
    assert recovery.activation_attempt_count == 0
    assert recovery.receiver_context_active
    assert not recovery.sender_context_confirmed
