import copy

import pytest

from spectrum_semcom.final_holdout import build_final_registry, consume_single_access, validate_final_catalog, verify_catalog_files


def scene(index: int) -> dict:
    digest = f"{index + 1:064x}"[-64:]
    return {
        "scene_id": f"final-{index}",
        "independence_group_id": f"group-{index}",
        "collection_event_id": f"event-{index}",
        "start_time_utc": f"2026-01-01T00:{index % 60:02d}:00Z",
        "receiver_ids": ["rx-1", "rx-2"],
        "provenance_ids": [f"new:{index}:rx1", f"new:{index}:rx2"],
        "source_files": [{"path": f"scene-{index}.bin", "sha256": digest}],
        "label_ref": f"scene-{index}.labels.json",
        "label_sha256": f"{index + 1000:064x}"[-64:],
        "integrity_status": "passed_without_model_output_access",
    }


def development() -> dict:
    return {"splits": {"train": {"scenes": [{"scene_id": "dev-1", "source_frame_ids": ["dev:frame:1"]}]}}}


def test_catalog_rejects_small_or_leaking_catalog() -> None:
    catalog = {"catalog_id": "new", "status": "candidate_final_catalog", "scenes": [scene(0)]}
    catalog["scenes"][0]["provenance_ids"][0] = "dev:frame:1"
    errors = validate_final_catalog(catalog, development())
    assert any("at least 200" in x for x in errors)
    assert any("overlaps development" in x for x in errors)


def test_materialization_and_access_are_single_use() -> None:
    catalog = {"catalog_id": "new", "status": "candidate_final_catalog", "scenes": [scene(i) for i in range(200)]}
    registry = build_final_registry(catalog, development())
    registry["file_integrity_audit"] = {"all_declared_files_exist": True, "all_sha256_match": True}
    assert registry["independent_scene_count"] == 200
    state = {"status": "not_accessed", "access_count": 0}
    updated = consume_single_access(state, registry, actor="test", code_snapshot_sha256="a" * 64, timestamp_utc="2026-01-01T00:00:00Z")
    assert updated["access_count"] == 1
    with pytest.raises(ValueError, match="already been consumed"):
        consume_single_access(updated, registry, actor="test", code_snapshot_sha256="a" * 64)


def test_duplicate_independence_group_is_rejected() -> None:
    catalog = {"catalog_id": "new", "status": "candidate_final_catalog", "scenes": [scene(i) for i in range(200)]}
    catalog = copy.deepcopy(catalog)
    catalog["scenes"][1]["independence_group_id"] = catalog["scenes"][0]["independence_group_id"]
    assert any("duplicate independence_group_id" in x for x in validate_final_catalog(catalog, development()))


def test_file_integrity_verifies_source_and_label(tmp_path) -> None:
    import hashlib

    source = tmp_path / "source.bin"
    label = tmp_path / "label.json"
    source.write_bytes(b"source")
    label.write_bytes(b"label")
    item = scene(0)
    item["source_files"] = [{"path": source.name, "sha256": hashlib.sha256(b"source").hexdigest()}]
    item["label_ref"] = label.name
    item["label_sha256"] = hashlib.sha256(b"label").hexdigest()
    catalog = {"scenes": [item]}
    assert verify_catalog_files(catalog, tmp_path) == []
    item["label_sha256"] = "0" * 64
    assert any("mismatch" in x for x in verify_catalog_files(catalog, tmp_path))
