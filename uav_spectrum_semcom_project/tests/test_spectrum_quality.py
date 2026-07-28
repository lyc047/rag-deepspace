import numpy as np
import pytest

from spectrum_semcom.spectrum_quality import (
    build_semantic_quality,
    clipping_ratio,
    noise_floor_stability_db,
    normalized_binary_entropy,
    out_of_band_leakage_ratio,
    peak_to_background_db,
)


def test_clipping_ratio_uses_real_adc_components() -> None:
    samples = np.asarray([0.2 + 0.1j, 1.0 + 0.2j, -1.0 - 1.0j])
    assert clipping_ratio(samples, full_scale=1.0, margin=0.98) == pytest.approx(3 / 6)
    assert clipping_ratio(samples * 0.1, full_scale=1.0, margin=0.98) == 0.0


def test_out_of_band_leakage_has_known_power_ratio() -> None:
    power = np.asarray([1.0, 4.0, 4.0, 1.0])
    mask = np.asarray([False, True, True, False])
    assert out_of_band_leakage_ratio(power, mask) == pytest.approx(0.2)
    assert out_of_band_leakage_ratio(np.zeros(4), mask) == 0.0


def test_noise_floor_stability_detects_time_variation() -> None:
    stable = np.ones((4, 8))
    varying = np.stack([np.ones(8), np.ones(8) * 2, np.ones(8) * 4, np.ones(8) * 8])
    assert noise_floor_stability_db(stable) == pytest.approx(0.0)
    assert noise_floor_stability_db(varying) > 3.0


def test_prediction_entropy_distinguishes_uncertain_and_deterministic_reports() -> None:
    assert normalized_binary_entropy(np.full(8, 0.5)) == pytest.approx(1.0)
    assert normalized_binary_entropy(np.asarray([0.0, 1.0, 0.0, 1.0])) < 1e-8


def test_peak_to_background_is_finite_for_empty_energy_and_positive_for_peak() -> None:
    assert peak_to_background_db(np.zeros(8)) == 0.0
    assert peak_to_background_db(np.asarray([1.0, 1.0, 1.0, 10.0])) > 9.0


def test_quality_builder_keeps_snr_and_front_end_failures_as_separate_features() -> None:
    power = np.asarray([1.0, 8.0, 8.0, 1.0])
    mask = np.asarray([False, True, True, False])
    clean = build_semantic_quality(
        sensing_snr_db=18.0,
        occupancy_probabilities=np.asarray([0.1, 0.9, 0.8, 0.2]),
        prediction_confidence=0.9,
        samples=np.asarray([0.1, 0.2, -0.3, 0.4]),
        full_scale=1.0,
        spectrum_power=power,
        intended_inband_mask=mask,
        noise_power_frames=np.ones((3, 4)),
        age_s=0.1,
        report_success_probability=0.95,
        calibration_error=0.05,
    )
    clipped = build_semantic_quality(
        sensing_snr_db=18.0,
        occupancy_probabilities=np.asarray([0.1, 0.9, 0.8, 0.2]),
        prediction_confidence=0.9,
        samples=np.asarray([1.0, -1.0, 1.0, -1.0]),
        full_scale=1.0,
        spectrum_power=power,
        intended_inband_mask=mask,
        noise_power_frames=np.ones((3, 4)),
        age_s=0.1,
        report_success_probability=0.95,
        calibration_error=0.05,
    )
    assert clean.sensing_snr_db == clipped.sensing_snr_db == 18.0
    assert clean.clipping_ratio == 0.0
    assert clipped.clipping_ratio == 1.0
