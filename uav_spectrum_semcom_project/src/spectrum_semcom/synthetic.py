from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import IQFrame, SignalBox


def _complex_awgn(rng: np.random.Generator, n: int, noise_power: float) -> np.ndarray:
    sigma = np.sqrt(noise_power / 2.0)
    return (rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)).astype(np.complex64)


def generate_synthetic_iq_frame(
    n_samples: int = 65_536,
    sample_rate_hz: float = 10e6,
    n_signals: int = 3,
    snr_db: float = 5.0,
    seed: Optional[int] = 0,
    frame_id: str = "synthetic_0000",
) -> IQFrame:
    """Generate a small synthetic wideband I/Q frame for pipeline debugging.

    This is not intended to replace real RF data. Its purpose is to verify data
    interfaces, STFT preprocessing, bit accounting, and baseline code before
    large datasets are introduced.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    iq = _complex_awgn(rng, n_samples, noise_power=1.0)
    boxes: List[SignalBox] = []
    labels = ["tone", "chirp", "burst"]

    signal_power = 10.0 ** (snr_db / 10.0)
    frame_duration_s = n_samples / sample_rate_hz
    nyquist = sample_rate_hz / 2.0

    for k in range(n_signals):
        duration = rng.uniform(0.08, 0.28) * frame_duration_s
        start = rng.uniform(0.02 * frame_duration_s, max(0.03 * frame_duration_s, frame_duration_s - duration))
        end = min(frame_duration_s, start + duration)
        i0 = max(0, int(start * sample_rate_hz))
        i1 = min(n_samples, int(end * sample_rate_hz))
        if i1 <= i0 + 8:
            continue

        label = labels[k % len(labels)]
        center = rng.uniform(-0.38 * nyquist, 0.38 * nyquist)
        bw = rng.uniform(0.02 * sample_rate_hz, 0.08 * sample_rate_hz)
        local_t = t[i0:i1] - t[i0]

        if label == "chirp":
            f0 = center - bw / 2.0
            f1 = center + bw / 2.0
            slope = (f1 - f0) / max(local_t[-1], 1e-12)
            phase = 2 * np.pi * (f0 * local_t + 0.5 * slope * local_t**2)
        elif label == "burst":
            f0 = center
            phase = 2 * np.pi * f0 * local_t
        else:
            f0 = center
            phase = 2 * np.pi * f0 * local_t

        window = np.hanning(i1 - i0)
        amp = np.sqrt(signal_power)
        iq[i0:i1] += (amp * window * np.exp(1j * phase)).astype(np.complex64)

        boxes.append(
            SignalBox(
                label=label,
                t_start_s=i0 / sample_rate_hz,
                t_end_s=i1 / sample_rate_hz,
                f_low_hz=center - bw / 2.0,
                f_high_hz=center + bw / 2.0,
                confidence=1.0,
            )
        )

    return IQFrame(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        center_freq_hz=0.0,
        boxes=boxes,
        frame_id=frame_id,
        metadata={"source": "synthetic", "snr_db": snr_db, "seed": seed},
    )

