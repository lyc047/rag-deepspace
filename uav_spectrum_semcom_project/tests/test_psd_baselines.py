import numpy as np

from spectrum_semcom.psd_baselines import (
    cfar_occupancy_summary,
    high_resolution_channel_psd,
    reduce_high_resolution_psd,
)


def test_high_resolution_psd_preserves_subchannel_structure() -> None:
    image = np.zeros((8, 4), dtype=np.uint8)
    image[2:4, :] = 255
    values = high_resolution_channel_psd(image, n_channels=2, bins_per_channel=2, axis="y")
    assert np.allclose(values, [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(reduce_high_resolution_psd(values, 2, 2, "max"), [1.0, 0.0])


def test_cfar_summary_marks_local_energy_outlier() -> None:
    values = np.asarray([0.10, 0.11, 0.10, 0.62, 0.09, 0.10, 0.11, 0.10], dtype=np.float32)
    summary = cfar_occupancy_summary(values, n_channels=2, bins_per_channel=4, threshold_sigma=3.0)
    assert summary[0] > summary[1]
