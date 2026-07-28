from dataclasses import replace

import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import (
    decode_task_update,
    encode_task_update,
    install_codebook,
)


def _fitted_session():
    query = (SpectrumTaskQuery(1),)
    states = [
        build_task_state(np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2),
        build_task_state(np.array([-99.9, -100.0, -95.0]), query, epsilon_db=0.2),
    ]
    return query, install_codebook(fit_greedy_task_codebook(states), epoch=7)


def test_codeword_packet_round_trips_with_real_header() -> None:
    query, session = _fitted_session()
    state = build_task_state(
        np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2
    )
    encoded = encode_task_update(state, session, node_id=12)
    decoded = decode_task_update(encoded.bits, session)
    assert encoded.bits.size == 48 + session.codebook.symbol_width_bits
    assert decoded.node_id == 12
    assert decoded.codebook_epoch == 7
    assert decoded.decoder_actions == encoded.decision.decoder_actions
    assert not decoded.uses_fallback


def test_escape_packet_round_trips_exact_actions() -> None:
    query, session = _fitted_session()
    state = build_task_state(
        np.array([-95.0, -95.0, -100.0]), query, epsilon_db=0.2
    )
    encoded = encode_task_update(state, session, node_id=1)
    decoded = decode_task_update(encoded.bits, session)
    assert encoded.decision.uses_fallback
    assert decoded.uses_fallback
    assert decoded.decoder_actions == state.optimal_actions
    assert encoded.decision.max_regret_db == 0.0


def test_wrong_epoch_or_codebook_tag_fails_closed() -> None:
    query, session = _fitted_session()
    state = build_task_state(
        np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2
    )
    encoded = encode_task_update(state, session, node_id=1)
    wrong_epoch = install_codebook(session.codebook, epoch=8)
    with pytest.raises(ValueError, match="epoch"):
        decode_task_update(encoded.bits, wrong_epoch)
    with pytest.raises(ValueError, match="identity"):
        replace(
            session,
            packet_tag=(session.packet_tag + 1) % 65536,
        )


def test_corrupt_or_trailing_packet_is_rejected() -> None:
    query, session = _fitted_session()
    state = build_task_state(
        np.array([-100.0, -99.9, -95.0]), query, epsilon_db=0.2
    )
    encoded = encode_task_update(state, session, node_id=1)
    corrupt = encoded.bits.copy()
    corrupt[0] ^= 1
    with pytest.raises(ValueError, match="identity"):
        decode_task_update(corrupt, session)
    wrong_tag = encoded.bits.copy()
    wrong_tag[32] ^= 1
    with pytest.raises(ValueError, match="tag"):
        decode_task_update(wrong_tag, session)
    with pytest.raises(ValueError, match="trailing"):
        decode_task_update(np.append(encoded.bits, 0), session)
