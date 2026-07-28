from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.radioml import load_radioml2016a, make_radioml_split, radioml_classes, radioml_snrs


def test_radioml_loader_if_present():
    path = PROJECT_DIR / "data" / "raw" / "radioml2016_10a" / "RML2016.10a_dict_optimized.pkl"
    if not path.exists():
        return
    data = load_radioml2016a(path)
    assert len(radioml_classes(data)) == 11
    assert len(radioml_snrs(data)) == 20
    split = make_radioml_split(data, train_per_group=2, val_per_group=1, test_per_group=1, min_snr=18)
    assert split.x_train.shape[1:] == (2, 128)
    assert len(split.classes) == 11

