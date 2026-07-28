from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.visdrone import parse_visdrone_annotation


def test_parse_visdrone_annotation_ignores_category_zero(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "10,20,30,40,1,4,0,0\n"
        "50,60,70,80,1,0,0,0\n"
        "1,2,3,4,1,11,0,0\n",
        encoding="utf-8",
    )
    boxes = parse_visdrone_annotation(path)
    assert len(boxes) == 2
    assert boxes[0].class_name == "car"
    assert boxes[1].class_name == "others"
