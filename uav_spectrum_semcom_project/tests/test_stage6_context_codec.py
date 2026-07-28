import numpy as np
import pytest

from spectrum_semcom.stage6_context_codec import (
    ContextEventState,
    choose_context_event_update,
    decode_compact_update,
    decode_context_install,
    encode_compact_update,
    encode_context_install,
    next_context_event_state,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook


def _session_and_states():
    queries = (SpectrumTaskQuery(1),)
    first = build_task_state(
        np.array([-100.0, -99.9, -95.0]), queries, epsilon_db=0.2
    )
    second = build_task_state(
        np.array([-99.9, -100.0, -95.0]), queries, epsilon_db=0.2
    )
    far = build_task_state(
        np.array([-95.0, -95.0, -100.0]), queries, epsilon_db=0.2
    )
    codebook = fit_greedy_task_codebook((first, second))
    return install_codebook(codebook, epoch=4), first, second, far


def test_binary_context_install_reconstructs_and_hash_verifies_codebook() -> None:
    session, _, _, _ = _session_and_states()
    bits = encode_context_install(session, node_id=9)
    decoded = decode_context_install(bits)
    assert decoded.node_id == 9
    assert decoded.session.epoch == session.epoch
    assert decoded.session.manifest_sha256 == session.manifest_sha256
    assert decoded.session.codebook.codewords == tuple(
        type(codeword)(
            codeword_id=codeword.codeword_id,
            decoder_actions=codeword.decoder_actions,
            training_coverage_count=0,
        )
        for codeword in session.codebook.codewords
    )


def test_context_install_hash_corruption_fails_closed() -> None:
    session, _, _, _ = _session_and_states()
    bits = encode_context_install(session, node_id=9)
    corrupt = bits.copy()
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_context_install(corrupt)


def test_compact_update_and_exact_escape_round_trip() -> None:
    session, first, _, far = _session_and_states()
    regular_bits, regular_decision = encode_compact_update(
        first, session, node_id=2, update_epoch=11
    )
    regular = decode_compact_update(regular_bits, session)
    assert regular_bits.size == 40 + session.codebook.symbol_width_bits
    assert regular.decoder_actions == regular_decision.decoder_actions
    assert regular.update_epoch == 11
    assert not regular.uses_fallback

    fallback_bits, fallback_decision = encode_compact_update(
        far, session, node_id=2, update_epoch=12
    )
    fallback = decode_compact_update(fallback_bits, session)
    assert fallback_decision.uses_fallback
    assert fallback.uses_fallback
    assert fallback.decoder_actions == far.optimal_actions


def test_compact_update_rejects_wrong_epoch() -> None:
    session, first, _, _ = _session_and_states()
    bits, _ = encode_compact_update(first, session, node_id=2)
    wrong = install_codebook(session.codebook, epoch=5)
    with pytest.raises(ValueError, match="epoch"):
        decode_compact_update(bits, wrong)


def test_context_event_policy_reuses_only_epsilon_safe_actions() -> None:
    session, first, second, far = _session_and_states()
    initial = choose_context_event_update(
        first,
        session,
        cached_state=None,
        timestamp_local="2023-01-01T00:00:00",
        max_age_minutes=120.0,
        node_id=1,
    )
    assert initial.transmits
    cached = next_context_event_state(
        initial,
        timestamp_local="2023-01-01T00:00:00",
        codebook_epoch=session.epoch,
    )
    reuse = choose_context_event_update(
        second,
        session,
        cached_state=cached,
        timestamp_local="2023-01-01T00:01:00",
        max_age_minutes=120.0,
        node_id=1,
    )
    assert not reuse.transmits
    assert reuse.reuse_max_regret_db <= 0.2
    refresh = choose_context_event_update(
        far,
        session,
        cached_state=cached,
        timestamp_local="2023-01-01T00:02:00",
        max_age_minutes=120.0,
        node_id=1,
    )
    assert refresh.transmits
    assert refresh.reuse_max_regret_db > 0.2


def test_context_event_policy_expires_old_state() -> None:
    session, first, _, _ = _session_and_states()
    cached = ContextEventState(
        decoder_actions=first.optimal_actions,
        success_timestamp="2023-01-01T00:00:00",
        codebook_epoch=session.epoch,
    )
    decision = choose_context_event_update(
        first,
        session,
        cached_state=cached,
        timestamp_local="2023-01-01T02:00:01",
        max_age_minutes=120.0,
        node_id=1,
    )
    assert decision.transmits
    assert decision.reason == "context_state_age_expired"
