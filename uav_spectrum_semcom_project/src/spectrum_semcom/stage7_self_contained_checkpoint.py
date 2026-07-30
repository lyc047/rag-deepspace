"""Self-contained bank checkpoint frames for Stage-7 context recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    CodewordDecision,
    SpectrumTaskQuery,
    SpectrumTaskState,
    encode_task_state,
)
from spectrum_semcom.stage6_task_codec import CodebookSession, install_codebook
from spectrum_semcom.stage6r_codebook_activation import (
    PreinstalledCodebookCatalog,
)


_MAGIC = 0xED
_VERSION = 1
_IDENTITY_BITS = 64


@dataclass(frozen=True)
class DecodedSelfContainedCheckpoint:
    node_id: int
    catalog_epoch: int
    bank_id: int
    codebook_epoch: int
    codebook_tag: int
    update_epoch: int
    symbol_id: int
    decoder_actions: tuple[int, ...]
    uses_fallback: bool
    session: CodebookSession


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if width < 1 or not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit checkpoint field")
    return np.asarray(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


def _action_alphabet(
    query: SpectrumTaskQuery,
    n_channels: int,
) -> tuple[int, ...]:
    candidate_count = int(n_channels) - int(query.demand_channels) + 1
    if candidate_count < 1:
        raise ValueError("query exceeds installed channel count")
    if query.allowed_starts is None:
        return tuple(range(candidate_count))
    if any(
        int(start) < 0 or int(start) >= candidate_count
        for start in query.allowed_starts
    ):
        raise ValueError("query action alphabet is outside candidate set")
    return tuple(map(int, query.allowed_starts))


def _action_width(query: SpectrumTaskQuery, n_channels: int) -> int:
    alphabet = _action_alphabet(query, n_channels)
    return max(1, int(math.ceil(math.log2(len(alphabet)))))


def encode_self_contained_checkpoint(
    state: SpectrumTaskState,
    catalog: PreinstalledCodebookCatalog,
    *,
    node_id: int,
    bank_id: int,
    codebook_epoch: int,
    update_epoch: int,
) -> tuple[np.ndarray, CodewordDecision]:
    """Encode catalog identity and the current semantic action in one frame."""

    if (
        not 0 <= int(node_id) <= 63
        or not 0 <= int(codebook_epoch) <= 255
        or not 0 <= int(update_epoch) <= 255
    ):
        raise ValueError("checkpoint field is outside the fixed schema")
    entry = catalog.entry(bank_id)
    session = install_codebook(entry.codebook, epoch=int(codebook_epoch))
    decision = encode_task_state(session.codebook, state)
    payload = [
        _uint_to_bits(
            decision.symbol_id,
            session.codebook.symbol_width_bits,
        )
    ]
    if decision.uses_fallback:
        for query, action in zip(
            session.codebook.queries,
            decision.decoder_actions,
        ):
            alphabet = _action_alphabet(query, session.codebook.n_channels)
            payload.append(
                _uint_to_bits(
                    alphabet.index(int(action)),
                    _action_width(query, session.codebook.n_channels),
                )
            )
    identity = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 2),
            _uint_to_bits(node_id, 6),
            _uint_to_bits(catalog.catalog_epoch, 8),
            _uint_to_bits(entry.bank_id, 8),
            _uint_to_bits(codebook_epoch, 8),
            _uint_to_bits(entry.packet_tag, 16),
            _uint_to_bits(update_epoch, 8),
        ]
    )
    if int(identity.size) != _IDENTITY_BITS:
        raise ValueError("checkpoint identity width invariant failed")
    return np.concatenate([identity, *payload]), decision


def decode_self_contained_checkpoint(
    bits: np.ndarray,
    catalog: PreinstalledCodebookCatalog,
) -> DecodedSelfContainedCheckpoint:
    """Resolve the preinstalled codebook and decode the action fail-closed."""

    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if (
        data.size <= _IDENTITY_BITS
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError("invalid self-contained checkpoint bitstream")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated self-contained checkpoint")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if take(8) != _MAGIC or take(2) != _VERSION:
        raise ValueError("invalid self-contained checkpoint identity")
    node_id = take(6)
    catalog_epoch = take(8)
    bank_id = take(8)
    codebook_epoch = take(8)
    codebook_tag = take(16)
    update_epoch = take(8)
    if catalog_epoch != int(catalog.catalog_epoch):
        raise ValueError("checkpoint catalog epoch mismatch")
    entry = catalog.entry(bank_id)
    if codebook_tag != entry.packet_tag:
        raise ValueError("checkpoint codebook tag mismatch")
    session = install_codebook(entry.codebook, epoch=codebook_epoch)
    symbol = take(session.codebook.symbol_width_bits)
    if symbol < len(session.codebook.codewords):
        actions = session.codebook.codewords[symbol].decoder_actions
        fallback = False
    elif symbol == session.codebook.escape_symbol:
        decoded = []
        for query in session.codebook.queries:
            alphabet = _action_alphabet(query, session.codebook.n_channels)
            rank = take(_action_width(query, session.codebook.n_channels))
            if rank >= len(alphabet):
                raise ValueError("checkpoint fallback rank outside alphabet")
            decoded.append(alphabet[rank])
        actions = tuple(decoded)
        fallback = True
    else:
        raise ValueError("checkpoint symbol is not assigned")
    if offset != data.size:
        raise ValueError("checkpoint has trailing bits")
    return DecodedSelfContainedCheckpoint(
        node_id=node_id,
        catalog_epoch=catalog_epoch,
        bank_id=bank_id,
        codebook_epoch=codebook_epoch,
        codebook_tag=codebook_tag,
        update_epoch=update_epoch,
        symbol_id=symbol,
        decoder_actions=actions,
        uses_fallback=fallback,
        session=session,
    )


def minimum_checkpoint_bits(catalog: PreinstalledCodebookCatalog, bank_id: int) -> int:
    """Return the non-escape checkpoint width for one bank."""

    entry = catalog.entry(bank_id)
    return int(_IDENTITY_BITS + entry.codebook.symbol_width_bits)
