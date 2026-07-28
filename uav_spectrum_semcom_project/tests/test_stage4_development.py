from spectrum_semcom.reproducibility import sha256_strings
from spectrum_semcom.stage4_development import make_scene_records, validate_development_registry


def registry() -> dict:
    splits = {}
    for index, split in enumerate(("train", "calibration", "validation")):
        records = make_scene_records(
            [f"source-{index}:frame-{index}-{item}" for item in range(4)],
            split_name=split,
            sources_per_scene=2,
            scene_count=2,
        )
        scene_ids = [record["scene_id"] for record in records]
        frames = [frame for record in records for frame in record["source_frame_ids"]]
        splits[split] = {
            "source_split": f"source-{index}",
            "scene_count": 2,
            "scene_ids_sha256": sha256_strings(scene_ids),
            "source_frame_ids_sha256": sha256_strings(frames),
            "scenes": records,
        }
    return {
        "split_id": "stage4_development_registry_v1",
        "parent_split_id": "stage4_scene_registry_v1",
        "selected_task": {"sources_per_scene": 2},
        "splits": splits,
        "final_holdout_included": False,
        "final_holdout_accessed": False,
    }


def test_development_registry_is_disjoint_and_excludes_final_holdout() -> None:
    assert validate_development_registry(registry()) == []


def test_development_registry_detects_source_frame_leakage() -> None:
    broken = registry()
    shared = broken["splits"]["train"]["scenes"][0]["source_frame_ids"][0]
    broken["splits"]["validation"]["scenes"][0]["source_frame_ids"][0] = shared
    errors = validate_development_registry(broken)
    assert any("source frame leakage" in error for error in errors)
