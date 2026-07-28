from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.frame_manifest import load_frame_manifest, load_iq_frame_from_manifest_row


def test_create_and_load_frame_manifest():
    subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "create_sigmf_frame_manifest.py")], check=True)
    rows = load_frame_manifest(PROJECT_DIR / "data" / "frame_manifest.csv", project_dir=PROJECT_DIR)
    assert len(rows) >= 3
    assert {row.split for row in rows} >= {"train", "val", "test"}
    frame = load_iq_frame_from_manifest_row(rows[0])
    assert frame.n_samples == rows[0].sample_count
    assert frame.frame_id == rows[0].frame_id
    assert len(frame.boxes) >= 1

