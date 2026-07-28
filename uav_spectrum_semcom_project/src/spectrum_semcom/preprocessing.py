from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class STFTResult:
    power_db: np.ndarray
    freqs_hz: np.ndarray
    times_s: np.ndarray
    n_fft: int
    hop_length: int


def stft_power(
    iq: np.ndarray,
    sample_rate_hz: float,
    n_fft: int = 512,
    hop_length: int = 128,
    eps: float = 1e-12,
) -> STFTResult:
    """Compute centered STFT power in dB for complex I/Q data."""

    iq = np.asarray(iq, dtype=np.complex64)
    if iq.ndim != 1:
        raise ValueError(f"iq must be 1-D, got {iq.shape}")
    if not 0 < hop_length <= n_fft:
        raise ValueError("hop_length must be in (0, n_fft]")

    noverlap = n_fft - hop_length
    freqs, times, zxx = signal.stft(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    freqs = np.fft.fftshift(freqs)
    zxx = np.fft.fftshift(zxx, axes=0)
    power = np.abs(zxx) ** 2
    power_db = 10.0 * np.log10(power + eps)
    return STFTResult(power_db=power_db.astype(np.float32), freqs_hz=freqs, times_s=times, n_fft=n_fft, hop_length=hop_length)

