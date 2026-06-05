# tests/test_channel.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from channel import AWGNChannel, FreeSpacePathLoss, DeepSpaceChannel, DopplerShift, SolarScintillation


def test_awgn_preserves_shape():
    ch = AWGNChannel(snr_db=-150)
    x = np.ones(1000)
    y = ch.forward(x)
    assert y.shape == x.shape


def test_awgn_adds_noise():
    ch = AWGNChannel(snr_db=-140)
    x = np.ones(10000)
    y = ch.forward(x)
    assert not np.allclose(x, y)
    assert y.dtype == np.float32


def test_awgn_low_snr_more_noise():
    ch_high = AWGNChannel(snr_db=-130)
    ch_low = AWGNChannel(snr_db=-160)
    x = np.ones(5000)
    std_high = np.std(ch_high.forward(x) - x)
    std_low = np.std(ch_low.forward(x) - x)
    assert std_low > std_high


def test_free_space_loss_attenuation():
    x = np.ones(1000)
    y_near = FreeSpacePathLoss.attenuate(x, distance_au=1.0, freq_ghz=8.4)
    y_far = FreeSpacePathLoss.attenuate(x, distance_au=3.0, freq_ghz=8.4)
    assert np.mean(np.abs(y_far)) < np.mean(np.abs(y_near))


def test_deepspace_channel_mvp():
    ch = DeepSpaceChannel(distance_au=2.0, snr_db=-150, freq_ghz=8.4)
    x = np.random.randn(1000).astype(np.float32)
    y = ch.forward(x, level='A')
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.var(y) < np.var(x)


def test_doppler_shift():
    doppler = DopplerShift(doppler_hz=5000, fs_hz=100.0)
    # 用宽带信号（多频率成分），多普勒频移会导致可见变化
    t = np.arange(1000) / 100.0
    x = (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) +
         0.3 * np.sin(2 * np.pi * 25 * t)).astype(np.float32)
    y = doppler.forward(x)
    assert y.shape == x.shape
    # 频移后信号功率应大致不变
    assert abs(np.var(y) - np.var(x)) / np.var(x) < 0.3


def test_doppler_zero():
    doppler = DopplerShift(doppler_hz=0, fs_hz=10.0)
    x = np.random.randn(500).astype(np.float32)
    y = doppler.forward(x)
    np.testing.assert_array_almost_equal(x, y, decimal=5)


def test_scintillation():
    scint = SolarScintillation(scintillation_index=0.3, fs_hz=10.0, seed=42)
    x = np.ones(1000).astype(np.float32)
    y = scint.forward(x)
    assert y.shape == x.shape
    assert not np.allclose(x, y)  # 有闪烁
    assert abs(np.mean(y) - 1.0) < 0.1  # 平均功率不变


def test_scintillation_zero():
    scint = SolarScintillation(scintillation_index=0, fs_hz=10.0)
    x = np.random.randn(500).astype(np.float32)
    y = scint.forward(x)
    np.testing.assert_array_almost_equal(x, y, decimal=5)


def test_b_level_channel():
    ch = DeepSpaceChannel(distance_au=2.0, snr_db=-150, freq_ghz=8.4,
                          doppler_hz=10000, scintillation_index=0.2, fs_hz=100.0)
    x = np.random.randn(2000).astype(np.float32)
    y = ch.forward(x, level='B')
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.var(y) < np.var(x)
