"""Auditable G0--G3 digital spectrum-semantic messages for stage 4.

The codec represents application bits explicitly.  Link header, CRC, FEC,
modulation padding, and retransmissions are then counted by the frozen stage-2
digital-link model instead of being estimated with an ideal compression ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from zlib import crc32

import numpy as np

from .digital_link import DigitalLinkConfig, TransmissionResult, transmit_payload_analytic


SemanticGranularity = Literal["G0", "G1", "G2", "G3"]

_MAGIC = 0xA7
_VERSION = 1
_GRANULARITY_TO_ID = {"G1": 0, "G2": 1, "G3": 2}
_ID_TO_GRANULARITY = {value: key for key, value in _GRANULARITY_TO_ID.items()}
_PROBABILITY_BITS_TO_ID = {1: 0, 2: 1, 4: 2, 8: 3}
_ID_TO_PROBABILITY_BITS = {value: key for key, value in _PROBABILITY_BITS_TO_ID.items()}


@dataclass(frozen=True)
class SemanticQuality:
    sensing_snr_db: float
    prediction_confidence: float
    normalized_entropy: float
    clipping_ratio: float
    out_of_band_leakage_ratio: float
    noise_floor_stability_db: float
    peak_to_background_db: float
    age_s: float
    report_success_probability: float
    calibration_error: float

    def __post_init__(self) -> None:
        unit_fields = (
            "prediction_confidence",
            "normalized_entropy",
            "clipping_ratio",
            "out_of_band_leakage_ratio",
            "report_success_probability",
            "calibration_error",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        finite_fields = ("sensing_snr_db", "noise_floor_stability_db", "peak_to_background_db", "age_s")
        if any(not np.isfinite(float(getattr(self, name))) for name in finite_fields):
            raise ValueError("quality metadata must be finite")
        if self.noise_floor_stability_db < 0 or self.age_s < 0:
            raise ValueError("noise_floor_stability_db and age_s must be non-negative")


@dataclass(frozen=True)
class SemanticEvidence:
    start_channel: int
    stop_channel: int
    class_id: int
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        if not 0 <= int(self.start_channel) < int(self.stop_channel) <= 255:
            raise ValueError("evidence channel range must satisfy 0 <= start < stop <= 255")
        if not 0 <= int(self.class_id) <= 255:
            raise ValueError("class_id must fit in 8 bits")
        for name in ("confidence", "uncertainty"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"evidence {name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class SemanticMessage:
    granularity: SemanticGranularity
    node_id: int
    scene_id: str
    occupancy: np.ndarray
    probability_bits: int = 1
    quality: SemanticQuality | None = None
    evidence: tuple[SemanticEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.granularity not in {"G0", "G1", "G2", "G3"}:
            raise ValueError(f"unsupported semantic granularity: {self.granularity}")
        if not 0 <= int(self.node_id) <= 4095:
            raise ValueError("node_id must fit in 12 bits")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        occupancy = np.asarray(self.occupancy, dtype=np.float64).reshape(-1)
        if self.granularity == "G0":
            if occupancy.size or self.quality is not None or self.evidence:
                raise ValueError("G0 is silence and cannot carry occupancy, quality, or evidence")
            if self.probability_bits != 0:
                raise ValueError("G0 probability_bits must be zero")
        else:
            if not 1 <= occupancy.size <= 255:
                raise ValueError("G1-G3 occupancy must contain 1 to 255 channels")
            if not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
                raise ValueError("occupancy probabilities must be finite and in [0, 1]")
            allowed = {"G1": {1, 2}, "G2": {1, 2, 4, 8}, "G3": {2, 4, 8}}[self.granularity]
            if self.probability_bits not in allowed:
                raise ValueError(f"{self.granularity} probability_bits must be one of {sorted(allowed)}")
            if self.granularity == "G1" and (self.quality is not None or self.evidence):
                raise ValueError("G1 carries occupancy preview only")
            if self.granularity in {"G2", "G3"} and self.quality is None:
                raise ValueError(f"{self.granularity} requires quality metadata")
            if self.granularity == "G2" and self.evidence:
                raise ValueError("G2 cannot carry local evidence")
            if len(self.evidence) > 255:
                raise ValueError("G3 supports at most 255 evidence entries")
            if any(item.stop_channel > occupancy.size for item in self.evidence):
                raise ValueError("evidence range exceeds occupancy width")
        object.__setattr__(self, "occupancy", occupancy.copy())


@dataclass(frozen=True)
class EncodedSemanticMessage:
    bits: np.ndarray
    granularity: SemanticGranularity
    node_id: int
    scene_tag: int
    n_channels: int
    probability_bits: int

    @property
    def application_bits(self) -> int:
        return int(self.bits.size)


@dataclass(frozen=True)
class DecodedSemanticMessage:
    granularity: SemanticGranularity
    node_id: int
    scene_tag: int
    occupancy: np.ndarray
    probability_bits: int
    quality: SemanticQuality | None
    evidence: tuple[SemanticEvidence, ...]


@dataclass(frozen=True)
class SemanticTransmission:
    encoded: EncodedSemanticMessage
    link: TransmissionResult
    decoded: DecodedSemanticMessage | None


def scene_tag(scene_id: str) -> int:
    return int(crc32(scene_id.encode("utf-8")) & 0xFFFFFFFF)


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if width <= 0 or not 0 <= int(value) < 2**width:
        raise ValueError(f"value {value} does not fit in {width} bits")
    return np.asarray([(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _quantize(value: float, low: float, high: float, width: int) -> int:
    clipped = float(np.clip(value, low, high))
    levels = 2**width - 1
    return int(np.rint((clipped - low) / (high - low) * levels))


def _dequantize(code: int, low: float, high: float, width: int) -> float:
    return float(low + (high - low) * int(code) / (2**width - 1))


def quantize_probabilities(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    if bits not in {1, 2, 4, 8}:
        raise ValueError("probability bits must be one of 1, 2, 4, or 8")
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    levels = 2**bits - 1
    codes = np.rint(probabilities * levels).astype(np.uint16)
    reconstructed = codes.astype(np.float64) / levels
    return codes, reconstructed


def _encode_quality(quality: SemanticQuality) -> np.ndarray:
    fields = (
        (quality.sensing_snr_db, -32.0, 31.5, 7),
        (quality.prediction_confidence, 0.0, 1.0, 8),
        (quality.normalized_entropy, 0.0, 1.0, 8),
        (quality.clipping_ratio, 0.0, 1.0, 8),
        (quality.out_of_band_leakage_ratio, 0.0, 1.0, 8),
        (quality.noise_floor_stability_db, 0.0, 16.0, 8),
        (quality.peak_to_background_db, -16.0, 48.0, 8),
        (quality.age_s, 0.0, 4.095, 12),
        (quality.report_success_probability, 0.0, 1.0, 8),
        (quality.calibration_error, 0.0, 1.0, 8),
    )
    return np.concatenate([_uint_to_bits(_quantize(value, low, high, width), width) for value, low, high, width in fields])


def _decode_quality(bits: np.ndarray, offset: int) -> tuple[SemanticQuality, int]:
    specs = (
        (-32.0, 31.5, 7),
        (0.0, 1.0, 8),
        (0.0, 1.0, 8),
        (0.0, 1.0, 8),
        (0.0, 1.0, 8),
        (0.0, 16.0, 8),
        (-16.0, 48.0, 8),
        (0.0, 4.095, 12),
        (0.0, 1.0, 8),
        (0.0, 1.0, 8),
    )
    values: list[float] = []
    for low, high, width in specs:
        values.append(_dequantize(_bits_to_uint(bits[offset : offset + width]), low, high, width))
        offset += width
    return SemanticQuality(*values), offset


def encode_semantic_message(message: SemanticMessage) -> EncodedSemanticMessage:
    tag = scene_tag(message.scene_id)
    if message.granularity == "G0":
        return EncodedSemanticMessage(np.empty(0, dtype=np.uint8), "G0", message.node_id, tag, 0, 0)

    header = np.concatenate(
        [
            _uint_to_bits(_MAGIC, 8),
            _uint_to_bits(_VERSION, 3),
            _uint_to_bits(_GRANULARITY_TO_ID[message.granularity], 2),
            _uint_to_bits(message.node_id, 12),
            _uint_to_bits(tag, 32),
            _uint_to_bits(message.occupancy.size, 8),
            _uint_to_bits(_PROBABILITY_BITS_TO_ID[message.probability_bits], 3),
        ]
    )
    codes, _ = quantize_probabilities(message.occupancy, message.probability_bits)
    occupancy_bits = np.concatenate([_uint_to_bits(int(code), message.probability_bits) for code in codes])
    parts = [header, occupancy_bits]
    if message.quality is not None:
        parts.append(_encode_quality(message.quality))
    if message.granularity == "G3":
        parts.append(_uint_to_bits(len(message.evidence), 8))
        for item in message.evidence:
            parts.append(
                np.concatenate(
                    [
                        _uint_to_bits(item.start_channel, 8),
                        _uint_to_bits(item.stop_channel, 8),
                        _uint_to_bits(item.class_id, 8),
                        _uint_to_bits(_quantize(item.confidence, 0.0, 1.0, 8), 8),
                        _uint_to_bits(_quantize(item.uncertainty, 0.0, 1.0, 8), 8),
                    ]
                )
            )
    bits = np.concatenate(parts).astype(np.uint8, copy=False)
    return EncodedSemanticMessage(bits, message.granularity, message.node_id, tag, message.occupancy.size, message.probability_bits)


def decode_semantic_message(bits: np.ndarray) -> DecodedSemanticMessage:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < 68 or np.any((data != 0) & (data != 1)):
        raise ValueError("semantic bitstream is too short or contains non-binary values")
    offset = 0

    def take(width: int) -> int:
        nonlocal offset
        if offset + width > data.size:
            raise ValueError("truncated semantic bitstream")
        value = _bits_to_uint(data[offset : offset + width])
        offset += width
        return value

    if take(8) != _MAGIC:
        raise ValueError("invalid semantic message magic")
    if take(3) != _VERSION:
        raise ValueError("unsupported semantic message version")
    granularity_id = take(2)
    if granularity_id not in _ID_TO_GRANULARITY:
        raise ValueError("invalid semantic granularity id")
    granularity = _ID_TO_GRANULARITY[granularity_id]
    node_id = take(12)
    tag = take(32)
    n_channels = take(8)
    probability_id = take(3)
    if n_channels == 0 or probability_id not in _ID_TO_PROBABILITY_BITS:
        raise ValueError("invalid semantic occupancy header")
    probability_bits = _ID_TO_PROBABILITY_BITS[probability_id]
    required = n_channels * probability_bits
    if offset + required > data.size:
        raise ValueError("truncated occupancy payload")
    codes = np.asarray(
        [_bits_to_uint(data[offset + i * probability_bits : offset + (i + 1) * probability_bits]) for i in range(n_channels)]
    )
    occupancy = codes.astype(np.float64) / (2**probability_bits - 1)
    offset += required

    quality = None
    if granularity in {"G2", "G3"}:
        if offset + 83 > data.size:
            raise ValueError("truncated quality metadata")
        quality, offset = _decode_quality(data, offset)

    evidence: list[SemanticEvidence] = []
    if granularity == "G3":
        count = take(8)
        if offset + count * 40 > data.size:
            raise ValueError("truncated local evidence")
        for _ in range(count):
            start = take(8)
            stop = take(8)
            class_id = take(8)
            confidence = _dequantize(take(8), 0.0, 1.0, 8)
            uncertainty = _dequantize(take(8), 0.0, 1.0, 8)
            evidence.append(SemanticEvidence(start, stop, class_id, confidence, uncertainty))
    if offset != data.size:
        raise ValueError("semantic bitstream has trailing bits")
    return DecodedSemanticMessage(granularity, node_id, tag, occupancy, probability_bits, quality, tuple(evidence))


def transmit_semantic_message(
    message: SemanticMessage,
    link_config: DigitalLinkConfig,
    seed: int,
) -> SemanticTransmission:
    """Transmit one message using the stage-2 fair packet-cost model.

    Packet-level delivery is all-or-nothing for this first codec version.  The
    application decoder is run only after every fragment has passed CRC/FEC.
    Later selective evidence recovery must be introduced as a new, explicitly
    versioned protocol rather than silently changing this assumption.
    """

    encoded = encode_semantic_message(message)
    link = transmit_payload_analytic(encoded.application_bits, link_config, seed)
    decoded = None
    if message.granularity != "G0" and link.frame_success:
        decoded = decode_semantic_message(encoded.bits)
    return SemanticTransmission(encoded, link, decoded)
