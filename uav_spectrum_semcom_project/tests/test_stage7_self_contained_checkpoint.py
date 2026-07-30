from __future__ import annotations

import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6r_codebook_activation import (
    build_preinstalled_catalog,
)
from spectrum_semcom.stage7_self_contained_checkpoint import (
    decode_self_contained_checkpoint,
    encode_self_contained_checkpoint,
    minimum_checkpoint_bits,
)


def _fixture():
    queries = (SpectrumTaskQuery(1), SpectrumTaskQuery(2))
    training = [
        build_task_state(
            np.asarray([-91.0, -90.0, -80.0, -79.0]),
            queries,
            epsilon_db=0.2,
        ),
        build_task_state(
            np.asarray([-80.0, -79.0, -91.0, -90.0]),
            queries,
            epsilon_db=0.2,
        ),
    ]
    codebook = fit_greedy_task_codebook(training)
    catalog = build_preinstalled_catalog(
        [(7, codebook)], catalog_epoch=3
    )
    return training, codebook, catalog


def test_checkpoint_roundtrip_without_prior_active_session() -> None:
    states, codebook, catalog = _fixture()
    packet, decision = encode_self_contained_checkpoint(
        states[0],
        catalog,
        node_id=1,
        bank_id=7,
        codebook_epoch=2,
        update_epoch=9,
    )
    decoded = decode_self_contained_checkpoint(packet, catalog)
    assert decoded.decoder_actions == decision.decoder_actions
    assert decoded.session.codebook == codebook
    assert decoded.codebook_epoch == 2
    assert decoded.update_epoch == 9
    assert not decoded.uses_fallback
    assert packet.size == minimum_checkpoint_bits(catalog, 7)
    assert 65 <= packet.size <= 72


def test_checkpoint_escape_roundtrip_appends_exact_actions() -> None:
    _, _, catalog = _fixture()
    queries = catalog.entry(7).codebook.queries
    outside = build_task_state(
        np.asarray([-80.0, -91.0, -79.0, -90.0]),
        queries,
        epsilon_db=0.2,
    )
    packet, decision = encode_self_contained_checkpoint(
        outside,
        catalog,
        node_id=1,
        bank_id=7,
        codebook_epoch=2,
        update_epoch=10,
    )
    decoded = decode_self_contained_checkpoint(packet, catalog)
    assert decision.uses_fallback
    assert decoded.uses_fallback
    assert decoded.decoder_actions == decision.decoder_actions
    assert packet.size > minimum_checkpoint_bits(catalog, 7)


def test_checkpoint_rejects_wrong_catalog_and_corruption() -> None:
    states, codebook, catalog = _fixture()
    packet, _ = encode_self_contained_checkpoint(
        states[0],
        catalog,
        node_id=1,
        bank_id=7,
        codebook_epoch=2,
        update_epoch=9,
    )
    wrong_epoch = build_preinstalled_catalog(
        [(7, codebook)], catalog_epoch=4
    )
    with pytest.raises(ValueError, match="catalog epoch"):
        decode_self_contained_checkpoint(packet, wrong_epoch)
    corrupted = packet.copy()
    corrupted[40] ^= 1
    with pytest.raises(ValueError, match="tag"):
        decode_self_contained_checkpoint(corrupted, catalog)
    with pytest.raises(ValueError, match="bitstream"):
        decode_self_contained_checkpoint(packet[:-2], catalog)


def test_checkpoint_rejects_unknown_bank() -> None:
    states, _, catalog = _fixture()
    with pytest.raises(ValueError, match="bank id"):
        encode_self_contained_checkpoint(
            states[0],
            catalog,
            node_id=1,
            bank_id=8,
            codebook_epoch=2,
            update_epoch=1,
        )
