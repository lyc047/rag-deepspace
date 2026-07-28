from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.raddet import parse_raddet_label_file


def test_parse_raddet_label_file(tmp_path: Path):
    label = tmp_path / "sample.txt"
    label.write_text("2 0.5 0.25 0.1 0.2\n9 0.2 0.3 0.4 0.5\n", encoding="utf-8")
    boxes = parse_raddet_label_file(label)
    assert len(boxes) == 2
    assert boxes[0].class_name == "Frank"
    assert boxes[1].class_name == "LFM"
