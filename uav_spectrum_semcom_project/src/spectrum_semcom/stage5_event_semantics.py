"""Query-aware event-triggered stateful semantics for Stage-5 development.

The module contains only deterministic codec and policy primitives.  It never
loads Stage-4 Final artifacts and does not select thresholds from evaluation
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np


ReportMode = Literal["silence", "absolute", "delta"]

_MAGIC = 0xE5
_VERSION = 1
_MODE_TO_ID = {"absolute": 0, "delta": 1}
_ID_TO_MODE = {value: key for key, value in _MODE_TO_ID.items()}
_MINIMUM_HEADER_BITS = 40


@dataclass(frozen=True)
class QueryState:
    """ACK-synchronised receiver state for one demand width."""

    selected_start: int
    success_timestamp: str
    epoch: int

    def __post_init__(self) -> None:
        if self.selected_start < 0:
            raise ValueError("selected_start must be non-negative")
        datetime.fromisoformat(self.success_timestamp)
        if not 0 <= int(self.epoch) <= 255:
            raise ValueError("epoch must fit eight bits")


@dataclass(frozen=True)
class EventDecision:
    mode: ReportMode
    target_start: int
    reuse_regret_db: float
    age_minutes: float | None
    application_bits: int
    payload_bits: int
    reason: str

    @property
    def transmits(self) -> bool:
        return self.mode != "silence"


@dataclass(frozen=True)
class DecodedUpdate:
    mode: Literal["absolute", "delta"]
    node_id: int
    demand_channels: int
    epoch: int
    selected_start: int


def contiguous_block_costs(
    channel_power_dbm: np.ndarray,
    demand_channels: int,
) -> np.ndarray:
    """Return the mean received power for every contiguous demand block."""

    values = np.asarray(channel_power_dbm, dtype=np.float64).reshape(-1)
    if (
        values.size < 1
        or not np.all(np.isfinite(values))
        or not 1 <= int(demand_channels) <= values.size
    ):
        raise ValueError("invalid channel powers or contiguous demand")
    kernel = np.ones(int(demand_channels), dtype=np.float64) / int(demand_channels)
    return np.convolve(values, kernel, mode="valid")


def block_regret(costs_dbm: np.ndarray, selected_start: int) -> float:
    costs = np.asarray(costs_dbm, dtype=np.float64).reshape(-1)
    if (
        costs.size < 1
        or not np.all(np.isfinite(costs))
        or not 0 <= int(selected_start) < costs.size
    ):
        raise ValueError("invalid block costs or selected start")
    return float(max(0.0, costs[int(selected_start)] - np.min(costs)))


def state_age_minutes(state: QueryState, timestamp_local: str) -> float:
    current = datetime.fromisoformat(timestamp_local)
    previous = datetime.fromisoformat(state.success_timestamp)
    age = (current - previous).total_seconds() / 60.0
    if age < 0:
        raise ValueError("current timestamp precedes state timestamp")
    return float(age)


def absolute_index_width(candidate_blocks: int) -> int:
    if candidate_blocks < 1:
        raise ValueError("candidate_blocks must be positive")
    return int(math.ceil(math.log2(candidate_blocks)))


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


def encode_signed_gamma(delta: int) -> np.ndarray:
    """Encode a non-zero signed integer using sign + Elias-gamma magnitude."""

    if int(delta) == 0:
        raise ValueError("delta update must be non-zero")
    magnitude = abs(int(delta))
    binary = bin(magnitude)[2:]
    gamma = ("0" * (len(binary) - 1)) + binary
    sign = "0" if int(delta) > 0 else "1"
    return np.fromiter((int(bit) for bit in sign + gamma), dtype=np.uint8)


def decode_signed_gamma(bits: np.ndarray) -> int:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < 2 or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid signed gamma code")
    sign = int(data[0])
    gamma = data[1:]
    zero_count = 0
    while zero_count < gamma.size and gamma[zero_count] == 0:
        zero_count += 1
    width = zero_count + 1
    if zero_count >= gamma.size or gamma.size != zero_count + width:
        raise ValueError("truncated or trailing signed gamma code")
    magnitude = _bits_to_uint(gamma[zero_count:])
    if magnitude < 1:
        raise ValueError("gamma magnitude must be positive")
    return -magnitude if sign else magnitude


def signed_gamma_width(delta: int) -> int:
    return int(encode_signed_gamma(delta).size)


def encode_update(
    *,
    mode: Literal["absolute", "delta"],
    target_start: int,
    cached_start: int | None,
    candidate_blocks: int,
    node_id: int,
    demand_channels: int,
    epoch: int,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> np.ndarray:
    """Encode a versioned absolute or state-differential block update."""

    if mode not in _MODE_TO_ID:
        raise ValueError("unsupported update mode")
    if (
        application_header_bits < _MINIMUM_HEADER_BITS
        or not 0 <= int(node_id) <= 255
        or not 1 <= int(demand_channels) <= 63
        or not 0 <= int(epoch) <= 255
        or not 0 <= int(target_start) < int(candidate_blocks)
    ):
        raise ValueError("invalid update header or target")
    if mode == "absolute":
        payload = _uint_to_bits(
            int(target_start),
            absolute_index_width(int(candidate_blocks)),
        )
    else:
        if cached_start is None or not 0 <= int(cached_start) < int(candidate_blocks):
            raise ValueError("delta mode requires valid cached state")
        payload = encode_signed_gamma(int(target_start) - int(cached_start))
    fixed = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_MODE_TO_ID[mode], 2),
            _uint_to_bits(int(node_id), 8),
            _uint_to_bits(int(demand_channels), 6),
            _uint_to_bits(int(epoch), 8),
            np.zeros(5, dtype=np.uint8),
        ]
    )
    padding = np.zeros(int(application_header_bits) - fixed.size, dtype=np.uint8)
    return np.concatenate([fixed, padding, payload])


def decode_update(
    bits: np.ndarray,
    *,
    candidate_blocks: int,
    cached_start: int | None,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> DecodedUpdate:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if (
        application_header_bits < _MINIMUM_HEADER_BITS
        or data.size < application_header_bits
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError("invalid update bitstream")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated update bitstream")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if take(8) != _MAGIC or take(3) != _VERSION:
        raise ValueError("invalid update magic or version")
    mode_id = take(2)
    node_id = take(8)
    demand = take(6)
    epoch = take(8)
    reserved = take(5)
    if mode_id not in _ID_TO_MODE or reserved != 0:
        raise ValueError("invalid update mode or reserved bits")
    offset = int(application_header_bits)
    payload = data[offset:]
    mode = _ID_TO_MODE[mode_id]
    if mode == "absolute":
        expected = absolute_index_width(int(candidate_blocks))
        if payload.size != expected:
            raise ValueError("absolute payload length mismatch")
        target = _bits_to_uint(payload)
    else:
        if cached_start is None:
            raise ValueError("delta decode requires cached state")
        target = int(cached_start) + decode_signed_gamma(payload)
    if not 0 <= int(target) < int(candidate_blocks):
        raise ValueError("decoded target is outside candidate set")
    return DecodedUpdate(mode, node_id, demand, epoch, int(target))


def choose_event_update(
    costs_dbm: np.ndarray,
    *,
    state: QueryState | None,
    timestamp_local: str,
    regret_threshold_db: float,
    max_age_minutes: float,
    demand_channels: int,
    application_header_bits: int = _MINIMUM_HEADER_BITS,
) -> EventDecision:
    """Choose silence, absolute index, or delta index without learned values."""

    costs = np.asarray(costs_dbm, dtype=np.float64).reshape(-1)
    if (
        costs.size < 1
        or not np.all(np.isfinite(costs))
        or regret_threshold_db < 0
        or max_age_minutes <= 0
    ):
        raise ValueError("invalid event-policy input")
    target = int(np.argmin(costs))
    absolute_payload = absolute_index_width(costs.size)
    absolute_bits = int(application_header_bits) + absolute_payload
    if state is None or state.selected_start >= costs.size:
        return EventDecision(
            "absolute",
            target,
            float("inf"),
            None,
            absolute_bits,
            absolute_payload,
            "missing_or_incompatible_state",
        )
    age = state_age_minutes(state, timestamp_local)
    reuse = block_regret(costs, state.selected_start)
    if age > float(max_age_minutes):
        return EventDecision(
            "absolute",
            target,
            reuse,
            age,
            absolute_bits,
            absolute_payload,
            "state_age_expired",
        )
    if reuse <= float(regret_threshold_db):
        return EventDecision(
            "silence",
            target,
            reuse,
            age,
            0,
            0,
            "reuse_regret_within_threshold",
        )
    delta = target - state.selected_start
    delta_payload = signed_gamma_width(delta)
    if delta_payload < absolute_payload:
        return EventDecision(
            "delta",
            target,
            reuse,
            age,
            int(application_header_bits) + delta_payload,
            delta_payload,
            "task_regret_trigger_delta_shorter",
        )
    return EventDecision(
        "absolute",
        target,
        reuse,
        age,
        absolute_bits,
        absolute_payload,
        "task_regret_trigger_absolute_shorter_or_equal",
    )


def next_success_state(
    decision: EventDecision,
    *,
    previous: QueryState | None,
    timestamp_local: str,
) -> QueryState:
    """Advance shared state only after the caller confirms frame success."""

    if not decision.transmits:
        raise ValueError("silence cannot create a successful update state")
    epoch = 0 if previous is None else (int(previous.epoch) + 1) % 256
    return QueryState(decision.target_start, timestamp_local, epoch)

