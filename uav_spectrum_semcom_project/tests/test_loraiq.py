import numpy as np
import pytest

from spectrum_semcom.loraiq import (
    channel_band_occupancy,
    excess_db_to_probability,
    quantize_probability,
    welch_channel_excess_db,
)


def test_channel_band_occupancy_maps_center_half_band():
    occupancy = channel_band_occupancy(-125.0, 125.0, 500.0, 4)
    assert np.allclose(occupancy, [0.0, 1.0, 1.0, 0.0])


def test_welch_excess_localizes_in_frame_tone():
    rng = np.random.default_rng(7)
    iq = (rng.normal(size=8192) + 1j * rng.normal(size=8192)).astype(np.complex64) * 0.05
    start, count = 2048, 4096
    time = np.arange(count) / 1000.0
    iq[start : start + count] += np.exp(2j * np.pi * 60.0 * time).astype(np.complex64)
    excess = welch_channel_excess_db(iq, start, count, 1000.0, 4, nperseg=256, guard_samples=64)
    assert int(np.argmax(excess)) == 2
    assert excess[2] > max(excess[0], excess[1], excess[3]) + 10.0


def test_probability_mapping_and_quantization_are_bounded():
    probabilities = excess_db_to_probability(np.asarray([-10.0, 1.5, 10.0]), 1.5, 1.0)
    assert probabilities[1] == pytest.approx(0.5)
    quantized = quantize_probability(probabilities, 8)
    assert np.all((quantized >= 0.0) & (quantized <= 1.0))
    assert np.allclose(quantized * 255.0, np.rint(quantized * 255.0), atol=1e-5)
