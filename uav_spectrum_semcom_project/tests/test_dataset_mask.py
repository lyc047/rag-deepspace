from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.datasets import boxes_to_occupancy_mask, sample_stft_patches
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.synthetic import generate_synthetic_iq_frame


def test_boxes_to_mask_and_patch_sampling():
    frame = generate_synthetic_iq_frame(n_samples=16_384, sample_rate_hz=2e6, n_signals=2, snr_db=10, seed=11)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=128, hop_length=32)
    mask = boxes_to_occupancy_mask(stft, frame.boxes)
    assert mask.shape == stft.power_db.shape
    assert mask.max() == 1.0
    patches = sample_stft_patches(stft.power_db, mask, patch_shape=(64, 64), max_patches=8, seed=0)
    assert patches.x.shape == (8, 1, 64, 64)
    assert patches.y.shape == (8, 1, 64, 64)

