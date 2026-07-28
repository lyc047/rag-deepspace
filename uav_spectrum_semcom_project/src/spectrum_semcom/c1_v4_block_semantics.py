"""Decision-aligned spectrum-block semantics for post-confirmation C1-v4 development."""

from __future__ import annotations

import json
import zipfile
from zlib import crc32
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from .aerpaw_spectrum import AerpawZipPair
from .digital_link import DigitalLinkConfig, transmit_payload_analytic


RepresentationKind = Literal["channel", "block"]

_RESOURCE_MAGIC = 0xB4
_RESOURCE_VERSION = 1
_REPRESENTATION_TO_ID = {
    "occupancy_fixed4": 0,
    "channel_power_fixed4": 1,
    "block_score_fixed4": 2,
    "block_score_fixed6": 3,
    "best_block_indicator1": 4,
}
_ID_TO_REPRESENTATION = {value: key for key, value in _REPRESENTATION_TO_ID.items()}
_BITS_TO_ID = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
_ID_TO_BITS = {value: key for key, value in _BITS_TO_ID.items()}


@dataclass(frozen=True)
class V4SpectrumArrays:
    site: str
    timestamp_local: str
    frequencies_mhz: np.ndarray
    powers_dbm: np.ndarray

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequencies_mhz, dtype=np.float64)
        power = np.asarray(self.powers_dbm, dtype=np.float32)
        if frequency.ndim != 1 or power.shape != frequency.shape or frequency.size < 2:
            raise ValueError("frequency and power must be equal non-trivial vectors")
        if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(power)) or np.any(np.diff(frequency) <= 0):
            raise ValueError("frequency/power vectors must be finite with increasing frequency")
        if not self.site or not self.timestamp_local:
            raise ValueError("site and timestamp are required")
        object.__setattr__(self, "frequencies_mhz", frequency)
        object.__setattr__(self, "powers_dbm", power)


@dataclass(frozen=True)
class BlockSemanticScene:
    features: np.ndarray
    occupancy: np.ndarray
    channel_power_dbm: np.ndarray
    normalized_channel_power: np.ndarray
    block_cost_dbm: np.ndarray
    normalized_block_cost: np.ndarray
    best_block_indicator: np.ndarray
    robust_threshold_dbm: float

    def __post_init__(self) -> None:
        arrays = (
            self.occupancy,
            self.channel_power_dbm,
            self.normalized_channel_power,
            self.block_cost_dbm,
            self.normalized_block_cost,
            self.best_block_indicator,
        )
        if self.features.ndim != 2 or any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("features must be 2-D and semantic arrays one-dimensional")
        if not all(np.all(np.isfinite(value)) for value in (self.features,) + arrays):
            raise ValueError("semantic scene arrays must be finite")


@dataclass(frozen=True)
class ResourceSemanticPayload:
    name: str
    values: np.ndarray
    value_bits: int
    representation_kind: RepresentationKind
    application_overhead_bits: int = 152

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if not self.name or values.size < 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("payload requires named finite unit-interval values")
        if not 1 <= int(self.value_bits) <= 8 or self.representation_kind not in {"channel", "block"}:
            raise ValueError("value_bits or representation_kind is invalid")
        if self.application_overhead_bits < 0:
            raise ValueError("application overhead must be non-negative")
        object.__setattr__(self, "values", values.copy())

    @property
    def application_bits(self) -> int:
        return int(self.application_overhead_bits + self.values.size * self.value_bits)

    @property
    def quantized_values(self) -> np.ndarray:
        return quantize_unit_interval(self.values, self.value_bits)


@dataclass(frozen=True)
class DecodedResourceSemanticPayload:
    name: str
    node_id: int
    scene_tag: int
    values: np.ndarray
    value_bits: int


@dataclass(frozen=True)
class ResourcePayloadOutcome:
    selected_start: int
    regret_db: float
    transmitted_bits: int
    frame_success: bool


def guarded_fallback_start(
    last_success_start: int | None,
    last_success_timestamp_local: str | None,
    current_timestamp_local: str,
    *,
    max_age_minutes: float,
    default_start: int = 0,
) -> int:
    """Return recent receiver state, otherwise a deterministic safe default.

    The timestamp guard prevents a decision decoded before a long reporting
    outage from being reused indefinitely.  It changes receiver state only and
    therefore adds no application or transmitted bits.
    """

    if max_age_minutes <= 0 or default_start < 0:
        raise ValueError("max_age_minutes must be positive and default_start non-negative")
    current = datetime.fromisoformat(current_timestamp_local)
    if last_success_start is None or last_success_timestamp_local is None:
        return int(default_start)
    previous = datetime.fromisoformat(last_success_timestamp_local)
    age_minutes = (current - previous).total_seconds() / 60.0
    if age_minutes < 0:
        raise ValueError("current timestamp precedes receiver state timestamp")
    return int(last_success_start) if age_minutes <= float(max_age_minutes) else int(default_start)


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if width < 1 or not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit requested bit width")
    return np.asarray([(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_to_uint(values: np.ndarray) -> int:
    output = 0
    for value in np.asarray(values, dtype=np.uint8).reshape(-1):
        output = (output << 1) | int(value)
    return output


def encode_resource_semantic_payload(payload: ResourceSemanticPayload, *, node_id: int, scene_id: str) -> np.ndarray:
    """Encode a versioned resource payload with equal 152-bit application overhead."""

    if payload.name not in _REPRESENTATION_TO_ID or not 0 <= int(node_id) <= 4095 or not scene_id:
        raise ValueError("unregistered representation, invalid node, or empty scene")
    if payload.value_bits not in _BITS_TO_ID or payload.application_overhead_bits != 152:
        raise ValueError("codec requires a registered bit width and 152-bit overhead")
    header = np.concatenate(
        [
            _uint_to_bits(_RESOURCE_MAGIC, 8),
            _uint_to_bits(_RESOURCE_VERSION, 3),
            _uint_to_bits(_REPRESENTATION_TO_ID[payload.name], 3),
            _uint_to_bits(node_id, 12),
            _uint_to_bits(crc32(scene_id.encode("utf-8")) & 0xFFFFFFFF, 32),
            _uint_to_bits(payload.values.size, 8),
            _uint_to_bits(_BITS_TO_ID[payload.value_bits], 3),
        ]
    )
    # Keep the same 83-bit quality footprint as G2. V4 development does not
    # exploit these fields, so zeros prevent representation-specific side information.
    quality_compatible = np.zeros(83, dtype=np.uint8)
    levels = 2**payload.value_bits - 1
    codes = np.rint(payload.values * levels).astype(int)
    value_bits = np.concatenate([_uint_to_bits(int(code), payload.value_bits) for code in codes])
    output = np.concatenate([header, quality_compatible, value_bits])
    if output.size != payload.application_bits:
        raise RuntimeError("resource payload codec length differs from declared application bits")
    return output


def decode_resource_semantic_payload(bits: np.ndarray) -> DecodedResourceSemanticPayload:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < 153 or np.any((data != 0) & (data != 1)):
        raise ValueError("resource bitstream is too short or non-binary")
    offset = 0
    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated resource bitstream")
        value = _bits_to_uint(data[offset : offset + width]); offset += width
        return value
    if take(8) != _RESOURCE_MAGIC or take(3) != _RESOURCE_VERSION:
        raise ValueError("invalid resource message magic or version")
    representation_id = take(3); node_id = take(12); tag = take(32); count = take(8); bit_id = take(3)
    if representation_id not in _ID_TO_REPRESENTATION or bit_id not in _ID_TO_BITS or count < 1:
        raise ValueError("invalid resource representation header")
    value_bits = _ID_TO_BITS[bit_id]
    offset += 83
    if offset + count * value_bits != data.size:
        raise ValueError("truncated resource values or trailing bits")
    levels = 2**value_bits - 1
    values = np.asarray([take(value_bits) / levels for _ in range(count)], dtype=np.float64)
    return DecodedResourceSemanticPayload(_ID_TO_REPRESENTATION[representation_id], node_id, tag, values, value_bits)


def load_aerpaw_zip_power_sweep(pair: AerpawZipPair, handle: zipfile.ZipFile | None = None) -> V4SpectrumArrays:
    """Read one selected ZIP pair without extracting or treating power as complex I/Q."""

    own = handle is None
    archive = zipfile.ZipFile(pair.archive_path) if own else handle
    try:
        metadata = json.loads(archive.read(pair.meta_member).decode("utf-8"))
        global_meta = metadata.get("global", {})
        if str(global_meta.get("core:datatype", "")).lower() != "rf32_le":
            raise ValueError("C1-v4 requires rf32_le received-power sweeps")
        frequency = np.asarray(global_meta.get("dataset:frequency_axis_MHz", []), dtype=np.float64)
        declared = int(global_meta.get("dataset:num_bins", frequency.size))
        raw = archive.read(pair.data_member)
        if declared < 2 or frequency.size != declared or len(raw) != declared * 4:
            raise ValueError("AERPAW ZIP sweep metadata/data size mismatch")
        power = np.frombuffer(raw, dtype="<f4").copy()
        return V4SpectrumArrays(str(global_meta.get("dataset:site", "")), pair.timestamp_local, frequency, power)
    finally:
        if own:
            archive.close()


def _channel_segments(frequency: np.ndarray, n_channels: int) -> list[slice]:
    if n_channels < 1 or n_channels > len(frequency):
        raise ValueError("invalid channel count")
    edges = np.linspace(float(frequency[0]), float(frequency[-1]), n_channels + 1)
    indices = np.searchsorted(frequency, edges, side="left")
    indices[0], indices[-1] = 0, len(frequency)
    output = [slice(int(indices[index]), int(indices[index + 1])) for index in range(n_channels)]
    if any(item.start == item.stop for item in output):
        raise ValueError("frequency partition contains an empty channel")
    return output


def normalize_cost(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 1 or not np.all(np.isfinite(data)):
        raise ValueError("cost values must be non-empty and finite")
    span = float(np.max(data) - np.min(data))
    if span <= 1e-12:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - np.min(data)) / span).astype(np.float32)


def build_block_semantic_scene(
    sweep: V4SpectrumArrays,
    *,
    frequency_low_mhz: float,
    frequency_high_mhz: float,
    n_channels: int,
    demand_channels: int,
    sigma_multiplier: float = 3.0,
    gaussian_mad_scale: float = 1.4826,
    sigma_floor_db: float = 0.25,
) -> BlockSemanticScene:
    if frequency_low_mhz >= frequency_high_mhz or not 1 <= demand_channels <= n_channels:
        raise ValueError("invalid frequency band or resource demand")
    mask = (sweep.frequencies_mhz >= frequency_low_mhz) & (sweep.frequencies_mhz < frequency_high_mhz)
    frequency = sweep.frequencies_mhz[mask]
    power = sweep.powers_dbm[mask]
    if frequency.size < n_channels:
        raise ValueError("selected frequency band is too small")
    median = float(np.median(power))
    mad = float(np.median(np.abs(power.astype(np.float64) - median)))
    robust_scale = max(float(gaussian_mad_scale * mad), float(sigma_floor_db))
    threshold = median + float(sigma_multiplier) * robust_scale
    segments = _channel_segments(frequency, n_channels)
    occupancy = np.asarray([np.mean(power[item] > threshold) for item in segments], dtype=np.float32)
    means = np.asarray([np.mean(power[item], dtype=np.float64) for item in segments], dtype=np.float32)
    maxima = np.asarray([np.max(power[item]) for item in segments], dtype=np.float32)
    deviations = np.asarray([np.std(power[item], dtype=np.float64) for item in segments], dtype=np.float32)
    mean_z = np.clip((means - median) / robust_scale, -8.0, 8.0) / 8.0
    max_z = np.clip((maxima - median) / robust_scale, -8.0, 8.0) / 8.0
    std_scaled = np.clip(deviations / robust_scale, 0.0, 8.0) / 8.0
    features = np.stack([occupancy, mean_z, max_z, std_scaled], axis=1).astype(np.float32)
    block_cost = np.convolve(means.astype(np.float64), np.ones(demand_channels) / demand_channels, mode="valid").astype(np.float32)
    normalized_block = normalize_cost(block_cost)
    indicator = np.ones_like(normalized_block, dtype=np.float32)
    indicator[int(np.argmin(block_cost))] = 0.0
    return BlockSemanticScene(
        features,
        occupancy,
        means,
        normalize_cost(means),
        block_cost,
        normalized_block,
        indicator,
        threshold,
    )


def quantize_unit_interval(values: np.ndarray, bits: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not 1 <= int(bits) <= 8 or not np.all(np.isfinite(data)) or np.any((data < 0) | (data > 1)):
        raise ValueError("unit-interval quantization input or bit count is invalid")
    levels = 2**int(bits) - 1
    return (np.rint(data * levels) / levels).astype(np.float64)


def selected_block_start(values: np.ndarray, kind: RepresentationKind, demand_channels: int) -> int:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 1 or not np.all(np.isfinite(data)):
        raise ValueError("semantic values must be non-empty and finite")
    if kind == "block":
        return int(np.argmin(data))
    if kind != "channel" or not 1 <= demand_channels <= data.size:
        raise ValueError("invalid representation kind or resource demand")
    costs = np.convolve(data, np.ones(demand_channels) / demand_channels, mode="valid")
    return int(np.argmin(costs))


def block_regret_db(selected_start: int, block_cost_dbm: np.ndarray) -> float:
    costs = np.asarray(block_cost_dbm, dtype=np.float64).reshape(-1)
    if costs.size < 1 or not np.all(np.isfinite(costs)) or not 0 <= int(selected_start) < costs.size:
        raise ValueError("selected block or true block costs are invalid")
    return float(max(0.0, costs[int(selected_start)] - np.min(costs)))


def canonical_v4_payloads(scene: BlockSemanticScene) -> tuple[ResourceSemanticPayload, ...]:
    """Classical representations frozen before any C1-v4 development result."""

    return (
        ResourceSemanticPayload("occupancy_fixed4", scene.occupancy, 4, "channel"),
        ResourceSemanticPayload("channel_power_fixed4", scene.normalized_channel_power, 4, "channel"),
        ResourceSemanticPayload("block_score_fixed4", scene.normalized_block_cost, 4, "block"),
        ResourceSemanticPayload("block_score_fixed6", scene.normalized_block_cost, 6, "block"),
        ResourceSemanticPayload("best_block_indicator1", scene.best_block_indicator, 1, "block"),
    )


def transmit_resource_payload(
    payload: ResourceSemanticPayload,
    *,
    scene_id: str,
    block_cost_dbm: np.ndarray,
    demand_channels: int,
    link: DigitalLinkConfig,
    seed: int,
) -> ResourcePayloadOutcome:
    """Transmit an exact CRC-protected resource payload and evaluate its decision."""

    encoded = encode_resource_semantic_payload(payload, node_id=0, scene_id=scene_id)
    result = transmit_payload_analytic(int(encoded.size), link, seed)
    decoded = payload.quantized_values if result.frame_success else np.full(payload.values.size, 0.5)
    start = selected_block_start(decoded, payload.representation_kind, demand_channels)
    return ResourcePayloadOutcome(start, block_regret_db(start, block_cost_dbm), int(result.transmitted_bits), bool(result.frame_success))
