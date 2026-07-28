from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.data_io import load_sigmf_frame


def test_load_sigmf_short_slice():
    meta = PROJECT_DIR / "data" / "raw" / "sigmf_5g_short" / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
    if not meta.exists():
        return
    frame = load_sigmf_frame(meta, start_sample=0, max_samples=100_000)
    assert frame.n_samples == 100_000
    assert frame.sample_rate_hz > 0
    assert frame.iq.dtype.name == "complex64"
    assert len(frame.boxes) > 0

