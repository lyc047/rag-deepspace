import json
from pathlib import Path

from spectrum_semcom.reproducibility import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_registry() -> dict:
    path = PROJECT_DIR / "configs" / "dataset_registry.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_registered_manifest_hashes_match_workspace() -> None:
    registry = load_registry()
    for entry in registry["manifests"]:
        assert sha256_file(PROJECT_DIR / entry["path"]) == entry["sha256"]


def test_formal_manifests_have_no_cross_split_leakage() -> None:
    registry = load_registry()
    for entry in registry["manifests"]:
        if entry["formal_eligible"]:
            assert entry["cross_split_frame_ids"] == []
            assert entry["cross_split_source_groups"] == []


def test_single_recording_sigmf_is_not_formal_generalization_evidence() -> None:
    registry = load_registry()
    entry = next(item for item in registry["manifests"] if item["path"] == "data/frame_manifest.csv")
    assert entry["unique_source_groups"] == 1
    assert entry["cross_split_source_groups"]
    assert entry["formal_eligible"] is False


def test_raddet_uses_observed_directory_splits() -> None:
    registry = load_registry()
    raddet = registry["raddet"]
    assert raddet["formal_eligible"] is True
    assert raddet["split_policy"] == "local_data_yaml_directory_split"
    assert set(raddet["splits"]) == {"train", "val", "test"}
    assert all(item["frames"] > 0 for item in raddet["splits"].values())
