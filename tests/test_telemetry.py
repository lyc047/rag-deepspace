# tests/test_telemetry.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from telemetry import (
    generate_slow_varying, generate_periodic, generate_transient,
    generate_science_data, generate_physical_metadata,
    generate_telemetry_segment, TelemetrySample,
)


def test_slow_varying_shape():
    y = generate_slow_varying(duration_sec=60, fs_hz=10, base_temp=25.0)
    assert y.shape == (600,)
    assert np.all(np.isfinite(y))


def test_slow_varying_trend():
    y = generate_slow_varying(duration_sec=60, fs_hz=10, base_temp=25.0, trend=0.001)
    assert np.all(np.abs(y - np.mean(y)) < 3.0)


def test_periodic_shape():
    y = generate_periodic(duration_sec=60, fs_hz=10, base_freq=0.5, amplitude=1.0)
    assert y.shape == (600,)
    assert np.all(np.isfinite(y))


def test_periodic_has_dominant_freq():
    y = generate_periodic(duration_sec=60, fs_hz=10, base_freq=0.5, amplitude=1.0)
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1/10)
    peak_freq = freqs[np.argmax(fft[1:]) + 1]
    assert 0.3 < peak_freq < 0.7


def test_transient_has_spike():
    y = generate_transient(duration_sec=10, fs_hz=100, event_time=5.0)
    baseline_std = np.std(y[:300])
    event_peak = np.max(np.abs(y[400:600]))
    assert event_peak > 3 * baseline_std


def test_science_data_shape():
    y = generate_science_data(duration_sec=60, fs_hz=10, base_value=100.0)
    assert y.shape == (600,)
    assert np.all(np.isfinite(y))


def test_physical_metadata_ranges():
    m = generate_physical_metadata()
    assert 0.5 <= m['distance_au'] <= 3.0
    assert 0 <= m['sun_earth_probe_angle'] <= 90
    assert -170 <= m['snr_db'] <= -130
    assert -50000 <= m['doppler_shift_hz'] <= 50000
    assert 0 <= m['scintillation_index'] <= 0.5


def test_telemetry_sample_structure():
    y = generate_slow_varying()
    m = generate_physical_metadata()
    sample = TelemetrySample(signal=y, physics=m, signal_type='slow_varying', channel_name='temp_bus')
    assert sample.signal is not None
    assert sample.physics is not None
    assert sample.signal_type == 'slow_varying'


def test_generate_telemetry_segment():
    seg = generate_telemetry_segment(signal_type='periodic', seed=42)
    assert isinstance(seg, TelemetrySample)
    assert seg.signal.shape == (600,)
    assert seg.signal_type == 'periodic'
    assert seg.sample_id is not None
