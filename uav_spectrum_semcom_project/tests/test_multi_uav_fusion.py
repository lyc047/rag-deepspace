import numpy as np
import pytest

from spectrum_semcom.correlated_scenes import compose_max_hold_scene, grouped_source_indices, sliding_source_indices
from spectrum_semcom.iq_replay import ReplayChannel, frequency_channel_occupancy, reference_rf_energy_j, replay_iq_view
from spectrum_semcom.multi_uav_fusion import (
    fit_monotonic_confidence_calibrator,
    NodeOccupancyReport,
    fuse_occupancy,
    perturb_correlated_spectrogram,
    quality_weight,
    maximum_pair_disagreement,
)
from spectrum_semcom.types import SignalBox


def report(values, snr=0.0, confidence=1.0, reliability=1.0, age=0.0):
    return NodeOccupancyReport(np.asarray(values, dtype=np.float32), snr, confidence, reliability, age)


def test_classical_fusion_rules_are_auditable():
    reports = [report([0.0, 0.8]), report([0.2, 0.0]), report([0.0, 0.9])]
    assert np.allclose(fuse_occupancy(reports, "or"), [0.2, 0.9])
    assert np.allclose(fuse_occupancy(reports, "majority", occupancy_threshold=0.1), [0.0, 1.0])
    assert np.allclose(fuse_occupancy(reports, "mean"), [0.2 / 3.0, 1.7 / 3.0])
    assert np.allclose(fuse_occupancy(reports, "median"), [0.0, 0.8])


def test_quality_weight_rewards_reliable_fresh_high_snr_report():
    good = report([1.0], snr=10.0, confidence=0.9, reliability=0.99, age=0.01)
    poor = report([0.0], snr=-10.0, confidence=0.4, reliability=0.5, age=1.0)
    assert quality_weight(good) > quality_weight(poor)
    assert fuse_occupancy([good, poor], "quality_weighted")[0] > 0.8


def test_robust_quality_fusion_suppresses_disagreeing_high_confidence_node():
    consistent = [
        report([0.0, 0.8, 0.0], snr=0.0, confidence=0.7),
        report([0.0, 0.9, 0.0], snr=2.0, confidence=0.8),
        report([0.0, 0.7, 0.0], snr=-2.0, confidence=0.7),
    ]
    anomalous = report([0.9, 0.0, 0.9], snr=15.0, confidence=0.99)
    ordinary = fuse_occupancy(consistent + [anomalous], "quality_weighted")
    robust = fuse_occupancy(
        consistent + [anomalous],
        "robust_quality_weighted",
        disagreement_scale=0.05,
        max_weight_ratio=2.0,
    )
    assert robust[1] > ordinary[1]
    assert robust[0] < ordinary[0]


def test_pair_disagreement_is_symmetric_and_auditable():
    reports = [report([0.0, 0.2]), report([0.0, 0.8]), report([0.1, 0.5])]
    assert maximum_pair_disagreement(reports) == pytest.approx(0.3)
    assert maximum_pair_disagreement([reports[0]]) == 0.0


def test_invalid_fusion_inputs_fail_loudly():
    with pytest.raises(ValueError):
        fuse_occupancy([], "mean")
    with pytest.raises(ValueError):
        fuse_occupancy([report([0.0]), report([0.0, 1.0])], "mean")
    with pytest.raises(ValueError):
        fuse_occupancy([report([0.0])], "not-a-method")


def test_correlated_view_is_deterministic_and_shape_preserving():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    first = perturb_correlated_spectrogram(image, 6.0, np.random.default_rng(7), gain_db=-1.0, frequency_shift_bins=1)
    second = perturb_correlated_spectrogram(image, 6.0, np.random.default_rng(7), gain_db=-1.0, frequency_shift_bins=1)
    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_composite_scene_is_traceable_nonoverlapping_partition():
    groups = grouped_source_indices(10, 4)
    assert [group.tolist() for group in groups] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert np.array_equal(compose_max_hold_scene([np.array([[1, 9]], dtype=np.uint8), np.array([[4, 3]], dtype=np.uint8)]), np.array([[4, 9]], dtype=np.uint8))
    with pytest.raises(ValueError):
        grouped_source_indices(2, 0)


def test_sliding_scene_indices_preserve_temporal_overlap():
    groups = sliding_source_indices(6, 3)
    assert [group.tolist() for group in groups] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    with pytest.raises(ValueError):
        sliding_source_indices(6, 3, stride=0)


def test_monotonic_confidence_calibration_is_bounded_and_ordered():
    scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    qualities = np.asarray([0.2, 0.1, 0.4, 0.6, 0.55, 0.9])
    calibrator = fit_monotonic_confidence_calibrator(scores, qualities, n_bins=6)
    predicted = calibrator.predict(np.linspace(0.0, 1.0, 25))
    assert np.all(np.diff(predicted) >= -1e-12)
    assert np.all((predicted >= 0.0) & (predicted <= 1.0))


def test_same_source_iq_replay_is_deterministic_and_channel_specific():
    source = np.exp(2j * np.pi * 0.05 * np.arange(256)).astype(np.complex64)
    channel = ReplayChannel(10.0, gain_db=-1.0, frequency_offset_hz=100.0, multipath_taps=(1.0 + 0j, 0.2j))
    first = replay_iq_view(source, 10_000.0, channel, np.random.default_rng(4))
    second = replay_iq_view(source, 10_000.0, channel, np.random.default_rng(4))
    assert np.array_equal(first, second)
    assert first.shape == source.shape
    assert not np.array_equal(first, source)


def test_signal_boxes_map_to_frequency_channel_occupancy_and_reference_energy():
    boxes = [SignalBox("x", 0.0, 0.5, -50.0, 0.0)]
    occupancy = frequency_channel_occupancy(boxes, 200.0, 1.0, 4)
    assert np.allclose(occupancy, [0.0, 0.5, 0.0, 0.0])
    assert reference_rf_energy_j(100, 1, 0.002, 0.001, 2.0) == pytest.approx(0.002)
