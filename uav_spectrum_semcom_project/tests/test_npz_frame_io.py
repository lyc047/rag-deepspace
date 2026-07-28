from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.data_io import load_npz_frame, save_npz_frame
from spectrum_semcom.synthetic import generate_synthetic_iq_frame


def test_npz_frame_roundtrip(tmp_path):
    frame = generate_synthetic_iq_frame(n_samples=4096, sample_rate_hz=1e6, n_signals=2, seed=77)
    path = tmp_path / "frame.npz"
    save_npz_frame(frame, path)
    loaded = load_npz_frame(path)
    assert loaded.n_samples == frame.n_samples
    assert loaded.sample_rate_hz == frame.sample_rate_hz
    assert loaded.frame_id == frame.frame_id
    assert len(loaded.boxes) == len(frame.boxes)

