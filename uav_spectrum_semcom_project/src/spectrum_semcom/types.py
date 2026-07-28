from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SignalBox:
    """A time-frequency annotation for one signal instance.

    Time is measured in seconds and frequency offset is measured in Hz relative
    to the frame center frequency.
    """

    label: str
    t_start_s: float
    t_end_s: float
    f_low_hz: float
    f_high_hz: float
    confidence: float = 1.0

    def duration_s(self) -> float:
        return max(0.0, self.t_end_s - self.t_start_s)

    def bandwidth_hz(self) -> float:
        return max(0.0, self.f_high_hz - self.f_low_hz)


@dataclass
class IQFrame:
    """A unified sample object for raw complex baseband I/Q data."""

    iq: np.ndarray
    sample_rate_hz: float
    center_freq_hz: float = 0.0
    boxes: List[SignalBox] = field(default_factory=list)
    frame_id: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.iq = np.asarray(self.iq, dtype=np.complex64)
        if self.iq.ndim != 1:
            raise ValueError(f"IQFrame.iq must be 1-D complex array, got shape {self.iq.shape}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def n_samples(self) -> int:
        return int(self.iq.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz


@dataclass(frozen=True)
class SemanticPacket:
    """A compact task-oriented payload sent by one UAV."""

    uav_id: int
    frame_id: str
    boxes: List[SignalBox]
    feature_bits: int = 0
    raw_evidence_requested: bool = False
    metadata_bits: int = 0
    extra: Optional[Dict[str, object]] = None

