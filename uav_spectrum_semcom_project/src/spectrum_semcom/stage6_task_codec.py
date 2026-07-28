"""Versioned bitstream for the Stage-6 analytic task codebook."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    CodewordDecision,
    GreedyTaskCodebook,
    SpectrumTaskQuery,
    SpectrumTaskState,
    encode_task_state,
)


_MAGIC = 0xE7
_VERSION = 1
_MESSAGE_TYPE = 0
_MINIMUM_HEADER_BITS = 48


@dataclass(frozen=True)
class CodebookSession:
    """Installed codebook identity verified during session establishment."""

    codebook: GreedyTaskCodebook
    epoch: int
    manifest_sha256: str
    packet_tag: int

    def __post_init__(self) -> None:
        expected = codebook_manifest_sha256(self.codebook)
        if (
            not 0 <= int(self.epoch) <= 255
            or self.manifest_sha256 != expected
            or self.packet_tag != int(expected[:4], 16)
        ):
            raise ValueError("invalid codebook session identity")


@dataclass(frozen=True)
class EncodedTaskUpdate:
    bits: np.ndarray
    decision: CodewordDecision


@dataclass(frozen=True)
class DecodedTaskUpdate:
    node_id: int
    codebook_epoch: int
    codebook_tag: int
    symbol_id: int
    decoder_actions: tuple[int, ...]
    uses_fallback: bool


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if width < 0 or not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit requested width")
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
    if any(start >= candidate_count for start in query.allowed_starts):
        raise ValueError("query action alphabet is outside candidate set")
    return query.allowed_starts


def _manifest_payload(codebook: GreedyTaskCodebook) -> dict:
    return {
        "schema_version": 1,
        "n_channels": codebook.n_channels,
        "epsilon_db": codebook.epsilon_db,
        "queries": [
            {
                "demand_channels": query.demand_channels,
                "allowed_starts": (
                    None
                    if query.allowed_starts is None
                    else list(query.allowed_starts)
                ),
                "name": query.name,
            }
            for query in codebook.queries
        ],
        "codewords": [
            {
                "codeword_id": codeword.codeword_id,
                "decoder_actions": list(codeword.decoder_actions),
            }
            for codeword in codebook.codewords
        ],
        "escape_symbol": codebook.escape_symbol,
        "symbol_width_bits": codebook.symbol_width_bits,
    }


def codebook_manifest_sha256(codebook: GreedyTaskCodebook) -> str:
    payload = json.dumps(
        _manifest_payload(codebook),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def install_codebook(
    codebook: GreedyTaskCodebook,
    *,
    epoch: int,
) -> CodebookSession:
    digest = codebook_manifest_sha256(codebook)
    return CodebookSession(
        codebook=codebook,
        epoch=int(epoch),
        manifest_sha256=digest,
        packet_tag=int(digest[:4], 16),
    )


def encode_task_update(
    state: SpectrumTaskState,
    session: CodebookSession,
    *,
    node_id: int,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> EncodedTaskUpdate:
    """Encode a safe codeword or escape plus exact actions."""

    if (
        application_header_bits < _MINIMUM_HEADER_BITS
        or not 0 <= int(node_id) <= 255
        or len(session.codebook.queries) > 7
    ):
        raise ValueError("invalid task update header")
    decision = encode_task_state(session.codebook, state)
    payload = [_uint_to_bits(decision.symbol_id, session.codebook.symbol_width_bits)]
    if decision.uses_fallback:
        for query, action in zip(
            session.codebook.queries, decision.decoder_actions
        ):
            alphabet = _action_alphabet(query, session.codebook.n_channels)
            rank = alphabet.index(int(action))
            width = int(math.ceil(math.log2(len(alphabet))))
            payload.append(_uint_to_bits(rank, width))
    fixed = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_MESSAGE_TYPE, 2),
            _uint_to_bits(int(node_id), 8),
            _uint_to_bits(len(session.codebook.queries), 3),
            _uint_to_bits(int(session.epoch), 8),
            _uint_to_bits(int(session.packet_tag), 16),
        ]
    )
    padding = np.zeros(int(application_header_bits) - fixed.size, dtype=np.uint8)
    return EncodedTaskUpdate(
        bits=np.concatenate([fixed, padding, *payload]),
        decision=decision,
    )


def decode_task_update(
    bits: np.ndarray,
    session: CodebookSession,
    *,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> DecodedTaskUpdate:
    """Decode only when the installed epoch and codebook tag match."""

    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if (
        application_header_bits < _MINIMUM_HEADER_BITS
        or data.size < application_header_bits + session.codebook.symbol_width_bits
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError("invalid task update bitstream")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated task update")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if (
        take(8) != _MAGIC
        or take(3) != _VERSION
        or take(2) != _MESSAGE_TYPE
    ):
        raise ValueError("invalid task update identity")
    node_id = take(8)
    query_count = take(3)
    epoch = take(8)
    tag = take(16)
    if query_count != len(session.codebook.queries):
        raise ValueError("task update query schema mismatch")
    if epoch != session.epoch:
        raise ValueError("installed codebook epoch mismatch")
    if tag != session.packet_tag:
        raise ValueError("installed codebook tag mismatch")
    offset = int(application_header_bits)
    symbol = take(session.codebook.symbol_width_bits)
    if symbol < len(session.codebook.codewords):
        actions = session.codebook.codewords[symbol].decoder_actions
        fallback = False
    elif symbol == session.codebook.escape_symbol:
        decoded = []
        for query in session.codebook.queries:
            alphabet = _action_alphabet(query, session.codebook.n_channels)
            width = int(math.ceil(math.log2(len(alphabet))))
            rank = take(width)
            if rank >= len(alphabet):
                raise ValueError("fallback action rank outside alphabet")
            decoded.append(alphabet[rank])
        actions = tuple(decoded)
        fallback = True
    else:
        raise ValueError("task update symbol is not assigned")
    if offset != data.size:
        raise ValueError("task update has trailing bits")
    return DecodedTaskUpdate(
        node_id=node_id,
        codebook_epoch=epoch,
        codebook_tag=tag,
        symbol_id=symbol,
        decoder_actions=actions,
        uses_fallback=fallback,
    )
