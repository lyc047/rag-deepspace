"""Session-installed codebooks and compact event-driven Stage-6 updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from spectrum_semcom.stage6_task_codebook import (
    CodewordDecision,
    GreedyTaskCodebook,
    SpectrumTaskQuery,
    SpectrumTaskState,
    TaskCodeword,
    encode_task_state,
)
from spectrum_semcom.stage6_task_codec import (
    CodebookSession,
    codebook_manifest_sha256,
    install_codebook,
)


_INSTALL_MAGIC = 0xE8
_UPDATE_MAGIC = 0xE9
_VERSION = 1
_INSTALL_TYPE = 1
_UPDATE_TYPE = 0
_HEADER_BITS = 40
_HASH_BITS = 256


@dataclass(frozen=True)
class DecodedContextInstall:
    node_id: int
    session: CodebookSession


@dataclass(frozen=True)
class DecodedCompactUpdate:
    node_id: int
    codebook_epoch: int
    update_epoch: int
    symbol_id: int
    decoder_actions: tuple[int, ...]
    uses_fallback: bool


@dataclass(frozen=True)
class ContextEventState:
    decoder_actions: tuple[int, ...]
    success_timestamp: str
    codebook_epoch: int
    update_epoch: int = 0

    def __post_init__(self) -> None:
        datetime.fromisoformat(self.success_timestamp)
        if (
            not 0 <= int(self.codebook_epoch) <= 255
            or not 0 <= int(self.update_epoch) <= 255
        ):
            raise ValueError("context epochs must fit eight bits")


@dataclass(frozen=True)
class ContextEventDecision:
    mode: str
    decoder_actions: tuple[int, ...]
    reuse_max_regret_db: float
    age_minutes: float | None
    packet_bits: int
    reason: str
    encoded_bits: np.ndarray | None
    codeword_decision: CodewordDecision | None
    update_epoch: int

    @property
    def transmits(self) -> bool:
        return self.mode == "update"


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


def _action_width(query: SpectrumTaskQuery, n_channels: int) -> int:
    return int(math.ceil(math.log2(len(_action_alphabet(query, n_channels)))))


def _sha256_to_bits(digest: str) -> np.ndarray:
    if len(digest) != 64:
        raise ValueError("manifest digest must be SHA-256")
    raw = bytes.fromhex(digest)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def _bits_to_sha256(bits: np.ndarray) -> str:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size != _HASH_BITS:
        raise ValueError("manifest digest must contain 256 bits")
    return np.packbits(values).tobytes().hex()


def encode_context_install(
    session: CodebookSession,
    *,
    node_id: int,
) -> np.ndarray:
    """Encode a compact binary codebook manifest plus its full SHA-256."""

    codebook = session.codebook
    if (
        not 0 <= int(node_id) <= 255
        or not 1 <= codebook.n_channels <= 127
        or not 1 <= len(codebook.queries) <= 7
        or not 1 <= len(codebook.codewords) <= 255
        or any(query.name for query in codebook.queries)
    ):
        raise ValueError("codebook is outside the compact install schema")
    epsilon_millidb = int(round(codebook.epsilon_db * 1000.0))
    if (
        not 0 <= epsilon_millidb <= 65535
        or abs(epsilon_millidb / 1000.0 - codebook.epsilon_db) > 1e-12
    ):
        raise ValueError("epsilon is not exactly representable in milli-dB")
    fixed = np.concatenate(
        [
            _uint_to_bits(_INSTALL_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_INSTALL_TYPE, 2),
            _uint_to_bits(int(node_id), 8),
            _uint_to_bits(int(session.epoch), 8),
            _uint_to_bits(len(codebook.queries), 3),
            _uint_to_bits(len(codebook.codewords), 8),
        ]
    )
    payload = [
        _uint_to_bits(codebook.n_channels, 7),
        _uint_to_bits(epsilon_millidb, 16),
    ]
    for query in codebook.queries:
        allowed = query.allowed_starts
        payload.extend(
            [
                _uint_to_bits(query.demand_channels, 7),
                _uint_to_bits(0 if allowed is None else len(allowed), 7),
            ]
        )
        if allowed is not None:
            payload.extend(_uint_to_bits(start, 7) for start in allowed)
    for codeword in codebook.codewords:
        for query, action in zip(
            codebook.queries, codeword.decoder_actions
        ):
            alphabet = _action_alphabet(query, codebook.n_channels)
            payload.append(
                _uint_to_bits(
                    alphabet.index(int(action)),
                    _action_width(query, codebook.n_channels),
                )
            )
    payload.append(_sha256_to_bits(session.manifest_sha256))
    return np.concatenate([fixed, *payload])


def decode_context_install(bits: np.ndarray) -> DecodedContextInstall:
    """Reconstruct and hash-verify an installed task codebook."""

    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < _HEADER_BITS + 7 + 16 + _HASH_BITS:
        raise ValueError("truncated context install")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("invalid context install bits")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated context install")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if (
        take(8) != _INSTALL_MAGIC
        or take(3) != _VERSION
        or take(2) != _INSTALL_TYPE
    ):
        raise ValueError("invalid context install identity")
    node_id = take(8)
    epoch = take(8)
    query_count = take(3)
    codeword_count = take(8)
    n_channels = take(7)
    epsilon_db = take(16) / 1000.0
    if query_count < 1 or codeword_count < 1 or n_channels < 1:
        raise ValueError("empty context install schema")
    queries = []
    for _ in range(query_count):
        demand = take(7)
        allowed_count = take(7)
        allowed = (
            None
            if allowed_count == 0
            else tuple(take(7) for _ in range(allowed_count))
        )
        queries.append(SpectrumTaskQuery(demand, allowed_starts=allowed))
    codewords = []
    for codeword_id in range(codeword_count):
        actions = []
        for query in queries:
            alphabet = _action_alphabet(query, n_channels)
            rank = take(_action_width(query, n_channels))
            if rank >= len(alphabet):
                raise ValueError("installed codeword action rank is invalid")
            actions.append(alphabet[rank])
        codewords.append(
            TaskCodeword(
                codeword_id=codeword_id,
                decoder_actions=tuple(actions),
                training_coverage_count=0,
            )
        )
    if offset + _HASH_BITS != data.size:
        raise ValueError("context install length mismatch")
    declared_hash = _bits_to_sha256(data[offset:])
    reconstructed = GreedyTaskCodebook(
        n_channels=n_channels,
        queries=tuple(queries),
        epsilon_db=epsilon_db,
        codewords=tuple(codewords),
        training_scene_count=0,
        exact_action_tuple_count=0,
        training_covered_count=0,
    )
    actual_hash = codebook_manifest_sha256(reconstructed)
    if declared_hash != actual_hash:
        raise ValueError("context install manifest hash mismatch")
    return DecodedContextInstall(
        node_id=node_id,
        session=install_codebook(reconstructed, epoch=epoch),
    )


def encode_compact_update(
    state: SpectrumTaskState,
    session: CodebookSession,
    *,
    node_id: int,
    update_epoch: int = 0,
) -> tuple[np.ndarray, CodewordDecision]:
    """Encode an update using a previously hash-verified context."""

    if (
        not 0 <= int(node_id) <= 255
        or not 0 <= int(update_epoch) <= 255
        or len(session.codebook.queries) > 7
    ):
        raise ValueError("invalid compact update header")
    decision = encode_task_state(session.codebook, state)
    payload = [_uint_to_bits(decision.symbol_id, session.codebook.symbol_width_bits)]
    if decision.uses_fallback:
        for query, action in zip(
            session.codebook.queries, decision.decoder_actions
        ):
            alphabet = _action_alphabet(query, session.codebook.n_channels)
            payload.append(
                _uint_to_bits(
                    alphabet.index(int(action)),
                    _action_width(query, session.codebook.n_channels),
                )
            )
    fixed = np.concatenate(
        [
            _uint_to_bits(_UPDATE_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_UPDATE_TYPE, 2),
            _uint_to_bits(int(node_id), 8),
            _uint_to_bits(len(session.codebook.queries), 3),
            _uint_to_bits(int(session.epoch), 8),
            _uint_to_bits(int(update_epoch), 8),
        ]
    )
    return np.concatenate([fixed, *payload]), decision


def maximum_compact_update_bits(session: CodebookSession) -> int:
    """Return the schema-known worst-case update length before observation."""

    fallback_bits = sum(
        _action_width(query, session.codebook.n_channels)
        for query in session.codebook.queries
    )
    return int(
        _HEADER_BITS
        + session.codebook.symbol_width_bits
        + fallback_bits
    )


def decode_compact_update(
    bits: np.ndarray,
    session: CodebookSession,
) -> DecodedCompactUpdate:
    """Decode a compact update only under the installed full-hash context."""

    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < _HEADER_BITS + session.codebook.symbol_width_bits:
        raise ValueError("truncated compact update")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("invalid compact update bits")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated compact update")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if (
        take(8) != _UPDATE_MAGIC
        or take(3) != _VERSION
        or take(2) != _UPDATE_TYPE
    ):
        raise ValueError("invalid compact update identity")
    node_id = take(8)
    query_count = take(3)
    epoch = take(8)
    update_epoch = take(8)
    if query_count != len(session.codebook.queries):
        raise ValueError("compact update schema mismatch")
    if epoch != session.epoch:
        raise ValueError("compact update codebook epoch mismatch")
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
                raise ValueError("fallback action rank outside alphabet")
            decoded.append(alphabet[rank])
        actions = tuple(decoded)
        fallback = True
    else:
        raise ValueError("compact update symbol is not assigned")
    if offset != data.size:
        raise ValueError("compact update has trailing bits")
    return DecodedCompactUpdate(
        node_id=node_id,
        codebook_epoch=epoch,
        update_epoch=update_epoch,
        symbol_id=symbol,
        decoder_actions=actions,
        uses_fallback=fallback,
    )


def _state_age_minutes(state: ContextEventState, timestamp: str) -> float:
    current = datetime.fromisoformat(timestamp)
    previous = datetime.fromisoformat(state.success_timestamp)
    age = (current - previous).total_seconds() / 60.0
    if age < 0:
        raise ValueError("current timestamp precedes context state")
    return float(age)


def choose_context_event_update(
    task_state: SpectrumTaskState,
    session: CodebookSession,
    *,
    cached_state: ContextEventState | None,
    timestamp_local: str,
    max_age_minutes: float,
    node_id: int,
    update_epoch: int = 0,
) -> ContextEventDecision:
    """Choose silence or a compact update using exact multi-query regret."""

    if max_age_minutes <= 0:
        raise ValueError("max_age_minutes must be positive")
    if (
        cached_state is None
        or cached_state.codebook_epoch != session.epoch
        or len(cached_state.decoder_actions) != len(task_state.profiles)
    ):
        bits, decision = encode_compact_update(
            task_state,
            session,
            node_id=node_id,
            update_epoch=update_epoch,
        )
        return ContextEventDecision(
            mode="update",
            decoder_actions=decision.decoder_actions,
            reuse_max_regret_db=float("inf"),
            age_minutes=None,
            packet_bits=int(bits.size),
            reason="missing_or_incompatible_context_state",
            encoded_bits=bits,
            codeword_decision=decision,
            update_epoch=int(update_epoch),
        )
    age = _state_age_minutes(cached_state, timestamp_local)
    reuse = task_state.max_regret_db(cached_state.decoder_actions)
    if age <= max_age_minutes and reuse <= session.codebook.epsilon_db + 1e-12:
        return ContextEventDecision(
            mode="silence",
            decoder_actions=cached_state.decoder_actions,
            reuse_max_regret_db=reuse,
            age_minutes=age,
            packet_bits=0,
            reason="cached_semantic_action_within_risk",
            encoded_bits=None,
            codeword_decision=None,
            update_epoch=cached_state.update_epoch,
        )
    bits, decision = encode_compact_update(
        task_state,
        session,
        node_id=node_id,
        update_epoch=update_epoch,
    )
    return ContextEventDecision(
        mode="update",
        decoder_actions=decision.decoder_actions,
        reuse_max_regret_db=reuse,
        age_minutes=age,
        packet_bits=int(bits.size),
        reason=(
            "context_state_age_expired"
            if age > max_age_minutes
            else "cached_semantic_action_exceeds_risk"
        ),
        encoded_bits=bits,
        codeword_decision=decision,
        update_epoch=int(update_epoch),
    )


def next_context_event_state(
    decision: ContextEventDecision,
    *,
    timestamp_local: str,
    codebook_epoch: int,
) -> ContextEventState:
    if not decision.transmits:
        raise ValueError("silence cannot advance confirmed context state")
    return ContextEventState(
        decoder_actions=decision.decoder_actions,
        success_timestamp=timestamp_local,
        codebook_epoch=int(codebook_epoch),
        update_epoch=int(decision.update_epoch),
    )
