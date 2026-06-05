# tests/test_channel.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from channel import AWGNChannel, FreeSpacePathLoss, DeepSpaceChannel


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
    y = ch.forward(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.var(y) < np.var(x)
