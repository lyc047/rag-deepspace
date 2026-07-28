from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.baselines import connected_component_energy_detector
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.synthetic import generate_synthetic_iq_frame


def test_connected_component_energy_detector_smoke():
    frame = generate_synthetic_iq_frame(n_samples=16_384, sample_rate_hz=2e6, n_signals=2, snr_db=12, seed=123)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=256, hop_length=64)
    pred = connected_component_energy_detector(stft, threshold_sigma=3.0, min_cells=4, enhancement="freq_median")
    metrics = evaluate_detections(pred, frame.boxes, iou_threshold=0.01)
    assert len(pred) > 0
    assert metrics.best_iou >= 0.0
    assert metrics.n_truth == len(frame.boxes)


def test_connected_component_enhancement_modes():
    frame = generate_synthetic_iq_frame(n_samples=8192, sample_rate_hz=2e6, n_signals=1, snr_db=10, seed=321)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=128, hop_length=32)
    for mode in ["raw", "freq_median", "time_median", "local_median", "local_zscore"]:
        pred = connected_component_energy_detector(stft, threshold_sigma=3.0, min_cells=3, enhancement=mode)
        assert isinstance(pred, list)
