from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.baselines import binary_mask_to_boxes
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.synthetic import generate_synthetic_iq_frame


def test_binary_mask_to_boxes_smoke():
    frame = generate_synthetic_iq_frame(n_samples=8192, sample_rate_hz=1e6, n_signals=1, snr_db=8, seed=5)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=128, hop_length=32)
    active = np.zeros_like(stft.power_db, dtype=bool)
    active[20:40, 10:20] = True
    boxes = binary_mask_to_boxes(active, stft, min_cells=4)
    assert len(boxes) == 1
    assert boxes[0].duration_s() > 0
    assert boxes[0].bandwidth_hz() > 0

