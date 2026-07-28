from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import SemanticPacket, SignalBox


@dataclass(frozen=True)
class BitBudget:
    name: str
    bits_per_frame: int
    frame_duration_s: float

    @property
    def bitrate_bps(self) -> float:
        return self.bits_per_frame / max(self.frame_duration_s, 1e-12)


def iq_payload_bits(n_samples: int, bits_per_i: int = 12, bits_per_q: int = 12) -> int:
    return int(n_samples) * (int(bits_per_i) + int(bits_per_q))


def stft_payload_bits(shape: Iterable[int], bits_per_value: int = 8) -> int:
    return int(np.prod(tuple(shape))) * int(bits_per_value)


def signal_box_bits(
    label_bits: int = 8,
    time_start_bits: int = 16,
    time_end_bits: int = 16,
    freq_low_bits: int = 16,
    freq_high_bits: int = 16,
    confidence_bits: int = 8,
) -> int:
    return label_bits + time_start_bits + time_end_bits + freq_low_bits + freq_high_bits + confidence_bits


def packet_header_bits(metadata_bits: int = 160, uav_id_bits: int = 16, frame_id_bits: int = 32, trigger_bits: int = 1) -> int:
    return int(metadata_bits + uav_id_bits + frame_id_bits + trigger_bits)


def semantic_packet_bits(packet: SemanticPacket, per_box_bits: int | None = None) -> int:
    if per_box_bits is None:
        per_box_bits = signal_box_bits()
    header_bits = packet_header_bits(metadata_bits=packet.metadata_bits)
    return int(header_bits + len(packet.boxes) * per_box_bits + packet.feature_bits)


def boxes_only_bits(boxes: list[SignalBox]) -> int:
    return len(boxes) * signal_box_bits()


def hard_box_packet_bits(
    n_boxes: int,
    metadata_bits: int = 160,
    label_bits: int = 8,
    coord_bits: int = 16,
) -> int:
    """Hard-decision payload: label and quantized time/frequency box only."""

    per_box = label_bits + 4 * coord_bits
    return packet_header_bits(metadata_bits=metadata_bits) + int(n_boxes) * per_box


def soft_box_packet_bits(
    n_boxes: int,
    n_classes: int = 8,
    prob_bits: int = 8,
    metadata_bits: int = 160,
    coord_bits: int = 16,
    confidence_bits: int = 8,
) -> int:
    """Soft payload: box coordinates plus class probability vector."""

    per_box = 4 * coord_bits + confidence_bits + int(n_classes) * prob_bits
    return packet_header_bits(metadata_bits=metadata_bits) + int(n_boxes) * per_box


def feature_token_packet_bits(
    n_tokens: int,
    token_dim: int = 16,
    bits_per_value: int = 6,
    metadata_bits: int = 160,
    position_bits_per_token: int = 32,
    global_context_bits: int = 64,
) -> int:
    """Quantized feature-token payload for task-oriented semantic transmission."""

    per_token = int(token_dim) * int(bits_per_value) + int(position_bits_per_token)
    return packet_header_bits(metadata_bits=metadata_bits) + int(global_context_bits) + int(n_tokens) * per_token
