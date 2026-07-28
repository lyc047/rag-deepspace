from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.manifest import load_manifest


def test_load_manifest():
    rows = load_manifest(PROJECT_DIR / "data" / "manifest.csv", project_dir=PROJECT_DIR)
    assert rows
    row = rows[0]
    assert row.dataset_id == "sigmf_5g_short"
    assert row.meta_path.exists()
    assert row.data_path.exists()

