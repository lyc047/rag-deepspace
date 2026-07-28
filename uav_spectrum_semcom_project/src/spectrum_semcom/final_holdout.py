"""Strict registration and single-use access controls for stage-4 final data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def consumed_development_ids(registry: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    scene_ids: set[str] = set()
    provenance_ids: set[str] = set()
    for split in registry.get("splits", {}).values():
        for scene in split.get("scenes", []):
            scene_id = scene.get("scene_id")
            if _nonempty_string(scene_id):
                scene_ids.add(scene_id)
            provenance_ids.update(x for x in scene.get("source_frame_ids", []) if _nonempty_string(x))
    return scene_ids, provenance_ids


def validate_final_catalog(catalog: Mapping[str, Any], development_registry: Mapping[str, Any], minimum_scenes: int = 200) -> list[str]:
    errors: list[str] = []
    if catalog.get("status") != "candidate_final_catalog":
        errors.append("catalog status must be candidate_final_catalog")
    if not _nonempty_string(catalog.get("catalog_id")):
        errors.append("catalog_id must be a non-empty string")
    scenes = catalog.get("scenes")
    if not isinstance(scenes, list):
        return errors + ["scenes must be a list"]
    if len(scenes) < minimum_scenes:
        errors.append(f"catalog requires at least {minimum_scenes} independent scenes; found {len(scenes)}")

    development_scenes, development_provenance = consumed_development_ids(development_registry)
    seen_scene: set[str] = set()
    seen_group: set[str] = set()
    seen_event: set[str] = set()
    seen_source_hash: set[str] = set()
    seen_provenance: set[str] = set()
    for index, scene in enumerate(scenes):
        prefix = f"scene[{index}]"
        if not isinstance(scene, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        scene_id = scene.get("scene_id")
        group_id = scene.get("independence_group_id")
        event_id = scene.get("collection_event_id")
        for field, value in (("scene_id", scene_id), ("independence_group_id", group_id), ("collection_event_id", event_id), ("start_time_utc", scene.get("start_time_utc"))):
            if not _nonempty_string(value):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if scene_id in seen_scene:
            errors.append(f"duplicate scene_id: {scene_id}")
        if group_id in seen_group:
            errors.append(f"duplicate independence_group_id: {group_id}")
        if event_id in seen_event:
            errors.append(f"duplicate collection_event_id: {event_id}")
        if scene_id in development_scenes:
            errors.append(f"scene overlaps development registry: {scene_id}")
        if _nonempty_string(scene_id): seen_scene.add(scene_id)
        if _nonempty_string(group_id): seen_group.add(group_id)
        if _nonempty_string(event_id): seen_event.add(event_id)
        receivers = scene.get("receiver_ids")
        if not isinstance(receivers, list) or not receivers or len(receivers) != len(set(receivers)) or any(not _nonempty_string(x) for x in receivers):
            errors.append(f"{prefix}.receiver_ids must contain unique non-empty receiver IDs")
        provenance = scene.get("provenance_ids")
        if not isinstance(provenance, list) or not provenance or any(not _nonempty_string(x) for x in provenance):
            errors.append(f"{prefix}.provenance_ids must be a non-empty string list")
        else:
            for value in provenance:
                if value in development_provenance:
                    errors.append(f"provenance overlaps development data: {value}")
                if value in seen_provenance:
                    errors.append(f"duplicate provenance_id: {value}")
                seen_provenance.add(value)
        sources = scene.get("source_files")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{prefix}.source_files must be non-empty")
        else:
            for source_index, source in enumerate(sources):
                if not isinstance(source, Mapping) or not _nonempty_string(source.get("path")):
                    errors.append(f"{prefix}.source_files[{source_index}] requires path")
                    continue
                digest = source.get("sha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
                    errors.append(f"{prefix}.source_files[{source_index}] requires a 64-character SHA-256")
                elif digest.lower() in seen_source_hash:
                    errors.append(f"duplicate source SHA-256: {digest.lower()}")
                else:
                    seen_source_hash.add(digest.lower())
        label_sha = scene.get("label_sha256")
        if not _nonempty_string(scene.get("label_ref")):
            errors.append(f"{prefix}.label_ref must be a non-empty path")
        if not isinstance(label_sha, str) or len(label_sha) != 64 or any(ch not in "0123456789abcdef" for ch in label_sha.lower()):
            errors.append(f"{prefix}.label_sha256 requires a 64-character SHA-256")
        if scene.get("integrity_status") != "passed_without_model_output_access":
            errors.append(f"{prefix}.integrity_status must be passed_without_model_output_access")
    return errors


def verify_catalog_files(catalog: Mapping[str, Any], base_directory: str | Path) -> list[str]:
    """Verify declared source and label bytes without evaluating model outputs."""

    base = Path(base_directory).resolve()
    errors: list[str] = []
    for index, scene in enumerate(catalog.get("scenes", [])):
        if not isinstance(scene, Mapping):
            continue
        declared: list[tuple[str, object]] = []
        for source in scene.get("source_files", []):
            if isinstance(source, Mapping):
                declared.append((str(source.get("path", "")), source.get("sha256")))
        declared.append((str(scene.get("label_ref", "")), scene.get("label_sha256")))
        for relative, expected in declared:
            path = Path(relative)
            path = path if path.is_absolute() else base / path
            if not path.is_file():
                errors.append(f"scene[{index}] declared file does not exist: {relative}")
                continue
            actual = sha256_path(path)
            if not isinstance(expected, str) or actual != expected.lower():
                errors.append(f"scene[{index}] SHA-256 mismatch: {relative}")
    return errors


def build_final_registry(catalog: Mapping[str, Any], development_registry: Mapping[str, Any], minimum_scenes: int = 200) -> dict[str, Any]:
    errors = validate_final_catalog(catalog, development_registry, minimum_scenes)
    if errors:
        raise ValueError("invalid final catalog: " + "; ".join(errors))
    scenes = catalog["scenes"]
    real_multi_receiver = sum(len(scene["receiver_ids"]) >= 2 for scene in scenes)
    scene_ids = [scene["scene_id"] for scene in scenes]
    return {
        "version": "1.0",
        "registry_id": "stage4_final_registry_v1",
        "status": "materialized_not_accessed",
        "catalog_id": catalog["catalog_id"],
        "catalog_sha256": canonical_json_sha256(catalog),
        "development_registry_sha256": canonical_json_sha256(development_registry),
        "scene_count": len(scenes),
        "scene_ids": scene_ids,
        "scene_ids_sha256": canonical_json_sha256(scene_ids),
        "independent_scene_count": len({scene["independence_group_id"] for scene in scenes}),
        "real_multi_receiver_scene_count": real_multi_receiver,
        "scenes": scenes,
        "leakage_audit": {
            "unique_scene_ids": True,
            "unique_independence_groups": True,
            "unique_collection_events": True,
            "unique_source_hashes": True,
            "unique_provenance_ids": True,
            "development_overlap_count": 0,
        },
        "access_count": 0,
        "labels_or_model_outputs_accessed": False,
        "claim_boundary": "Registration checks provenance and integrity metadata only; no model outputs are produced.",
    }


def consume_single_access(state: Mapping[str, Any], registry: Mapping[str, Any], *, actor: str, code_snapshot_sha256: str, timestamp_utc: str | None = None) -> dict[str, Any]:
    if state.get("access_count") != 0 or state.get("status") != "not_accessed":
        raise ValueError("final holdout single-use access has already been consumed")
    if registry.get("status") != "materialized_not_accessed" or registry.get("access_count") != 0:
        raise ValueError("final registry is not materialized and unaccessed")
    if registry.get("scene_count", 0) < 200 or registry.get("independent_scene_count") != registry.get("scene_count"):
        raise ValueError("final registry does not meet the 200 independent-scene gate")
    integrity = registry.get("file_integrity_audit", {})
    if integrity.get("all_declared_files_exist") is not True or integrity.get("all_sha256_match") is not True:
        raise ValueError("final registry has not passed declared-file integrity verification")
    if not _nonempty_string(actor):
        raise ValueError("actor is required")
    if not isinstance(code_snapshot_sha256, str) or len(code_snapshot_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in code_snapshot_sha256.lower()):
        raise ValueError("code_snapshot_sha256 must be a SHA-256 string")
    timestamp = timestamp_utc or datetime.now(timezone.utc).isoformat()
    receipt = {
        "access_number": 1,
        "timestamp_utc": timestamp,
        "actor": actor,
        "registry_sha256": canonical_json_sha256(registry),
        "code_snapshot_sha256": code_snapshot_sha256.lower(),
    }
    return {"version": "1.0", "status": "access_consumed", "access_count": 1, "receipt": receipt}


def atomic_write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
