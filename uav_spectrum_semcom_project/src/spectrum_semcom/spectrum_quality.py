"""Front-end and prediction-quality diagnostics for spectrum reports.

Each feature has an explicit measurable definition.  The module intentionally
does not collapse the vector to a hard SNR-monotonic reliability score: a
high-SNR observation may still be clipped, spectrally leaky, stale, or poorly
calibrated.
"""

from __future__ import annotations

import numpy as np

from .multigranular_semantics import SemanticQuality


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite array")
    return array


def clipping_ratio(samples: np.ndarray, full_scale: float = 1.0, margin: float = 0.98) -> float:
    """Fraction of real ADC components at or above a declared full-scale margin."""

    values = _finite_array(samples, "samples")
    if full_scale <= 0 or not 0.0 < margin <= 1.0:
        raise ValueError("full_scale must be positive and margin must be in (0, 1]")
    if np.iscomplexobj(values):
        components = np.concatenate((values.real.reshape(-1), values.imag.reshape(-1)))
    else:
        components = values.astype(np.float64, copy=False).reshape(-1)
    return float(np.mean(np.abs(components) >= float(full_scale) * float(margin)))


def out_of_band_leakage_ratio(power: np.ndarray, inband_mask: np.ndarray) -> float:
    """Power outside the declared intended band divided by total measured power."""

    values = _finite_array(power, "power").astype(np.float64, copy=False)
    mask = np.asarray(inband_mask, dtype=bool)
    if mask.shape != values.shape or not np.any(mask):
        raise ValueError("inband_mask must match power and contain at least one in-band element")
    if np.any(values < 0):
        raise ValueError("power must be non-negative")
    total = float(np.sum(values))
    if total <= 1e-20:
        return 0.0
    return float(np.clip(np.sum(values[~mask]) / total, 0.0, 1.0))


def noise_floor_stability_db(power_frames: np.ndarray, noise_quantile: float = 0.2) -> float:
    """Sample standard deviation of per-frame lower-quantile noise floors in dB."""

    values = _finite_array(power_frames, "power_frames").astype(np.float64, copy=False)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("power_frames must contain at least two time frames")
    if np.any(values < 0) or not 0.0 < noise_quantile <= 0.5:
        raise ValueError("power must be non-negative and noise_quantile must be in (0, 0.5]")
    flattened = values.reshape(values.shape[0], -1)
    floors = np.quantile(flattened, noise_quantile, axis=1)
    floors_db = 10.0 * np.log10(np.maximum(floors, 1e-20))
    return float(np.std(floors_db, ddof=1))


def peak_to_background_db(power: np.ndarray, background_quantile: float = 0.5) -> float:
    """Peak power relative to a declared background quantile, in dB."""

    values = _finite_array(power, "power").astype(np.float64, copy=False)
    if np.any(values < 0) or not 0.0 < background_quantile < 1.0:
        raise ValueError("power must be non-negative and background_quantile must be in (0, 1)")
    peak = float(np.max(values))
    background = float(np.quantile(values, background_quantile))
    if peak <= 1e-20:
        return 0.0
    return float(10.0 * np.log10(peak / max(background, 1e-20)))


def normalized_binary_entropy(probabilities: np.ndarray) -> float:
    """Mean Bernoulli entropy in [0, 1] for channel-occupancy predictions."""

    values = _finite_array(probabilities, "probabilities").astype(np.float64, copy=False)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    return float(np.clip(np.mean(entropy), 0.0, 1.0))


def build_semantic_quality(
    *,
    sensing_snr_db: float,
    occupancy_probabilities: np.ndarray,
    prediction_confidence: float,
    samples: np.ndarray,
    full_scale: float,
    spectrum_power: np.ndarray,
    intended_inband_mask: np.ndarray,
    noise_power_frames: np.ndarray,
    age_s: float,
    report_success_probability: float,
    calibration_error: float,
    clipping_margin: float = 0.98,
    noise_quantile: float = 0.2,
    background_quantile: float = 0.5,
) -> SemanticQuality:
    """Construct the complete G2/G3 quality vector from auditable inputs."""

    return SemanticQuality(
        sensing_snr_db=float(sensing_snr_db),
        prediction_confidence=float(prediction_confidence),
        normalized_entropy=normalized_binary_entropy(occupancy_probabilities),
        clipping_ratio=clipping_ratio(samples, full_scale, clipping_margin),
        out_of_band_leakage_ratio=out_of_band_leakage_ratio(spectrum_power, intended_inband_mask),
        noise_floor_stability_db=noise_floor_stability_db(noise_power_frames, noise_quantile),
        peak_to_background_db=peak_to_background_db(spectrum_power, background_quantile),
        age_s=float(age_s),
        report_success_probability=float(report_success_probability),
        calibration_error=float(calibration_error),
    )
