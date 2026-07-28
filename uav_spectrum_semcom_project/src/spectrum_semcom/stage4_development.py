"""Build and validate development-only scene registries for Gate A."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .reproducibility import sha256_strings


DEVELOPMENT_SPLITS = ("train", "calibration", "validation")


def make_scene_records(
    frame_ids: Sequence[str],
    *,
    split_name: str,
    sources_per_scene: int,
    scene_count: int,
) -> list[dict[str, Any]]:
    if split_name not in DEVELOPMENT_SPLITS:
        raise ValueError(f"unsupported development split: {split_name}")
    sources = int(sources_per_scene)
    count = int(scene_count)
    if sources < 1 or count < 1 or len(frame_ids) < sources * count:
        raise ValueError("insufficient frame ids for the requested development scenes")
    selected = [str(value) for value in frame_ids[: sources * count]]
    if any(not value for value in selected) or len(selected) != len(set(selected)):
        raise ValueError("source frame ids must be non-empty and unique")
    records: list[dict[str, Any]] = []
    for index in range(count):
        group = selected[index * sources : (index + 1) * sources]
        records.append(
            {
                "scene_id": f"stage4-dev-{split_name}-{index:05d}-{sha256_strings(group)[:12]}",
                "source_frame_ids": group,
            }
        )
    return records


def validate_development_registry(registry: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("split_id") != "stage4_development_registry_v1":
        errors.append("unexpected development split_id")
    if registry.get("parent_split_id") != "stage4_scene_registry_v1":
        errors.append("development registry must reference the stage4 parent split")
    if registry.get("final_holdout_included") is not False or registry.get("final_holdout_accessed") is not False:
        errors.append("development registry must exclude and not access final_holdout")
    task = registry.get("selected_task")
    if not isinstance(task, Mapping) or int(task.get("sources_per_scene", 0)) < 1:
        errors.append("selected_task must define a positive sources_per_scene")
        return errors
    sources_per_scene = int(task["sources_per_scene"])
    splits = registry.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(DEVELOPMENT_SPLITS):
        errors.append("development registry must define train, calibration, and validation")
        return errors

    scene_owner: dict[str, str] = {}
    frame_owner: dict[str, str] = {}
    for split_name in DEVELOPMENT_SPLITS:
        entry = splits[split_name]
        scenes = entry.get("scenes") if isinstance(entry, Mapping) else None
        if not isinstance(scenes, list) or not scenes:
            errors.append(f"{split_name} must contain scene records")
            continue
        scene_ids: list[str] = []
        frame_ids: list[str] = []
        for scene in scenes:
            if not isinstance(scene, Mapping):
                errors.append(f"{split_name} has an invalid scene record")
                continue
            scene_id = scene.get("scene_id")
            sources = scene.get("source_frame_ids")
            if not isinstance(scene_id, str) or not scene_id:
                errors.append(f"{split_name} has an empty scene_id")
                continue
            if not isinstance(sources, list) or len(sources) != sources_per_scene or len(sources) != len(set(sources)):
                errors.append(f"{scene_id} must contain exactly {sources_per_scene} unique source frames")
                continue
            source_split = entry.get("source_split")
            if source_split is not None and any(
                not str(frame_id).startswith(f"{source_split}:") for frame_id in sources
            ):
                errors.append(f"{scene_id} contains a source id outside its declared source_split namespace")
            previous_scene = scene_owner.setdefault(scene_id, split_name)
            if previous_scene != split_name:
                errors.append(f"scene leakage between {previous_scene} and {split_name}: {scene_id}")
            scene_ids.append(scene_id)
            for frame_id in sources:
                previous_frame = frame_owner.setdefault(str(frame_id), split_name)
                if previous_frame != split_name:
                    errors.append(f"source frame leakage between {previous_frame} and {split_name}: {frame_id}")
                frame_ids.append(str(frame_id))
        if entry.get("scene_count") != len(scenes):
            errors.append(f"{split_name} scene_count does not match")
        if entry.get("scene_ids_sha256") != sha256_strings(scene_ids):
            errors.append(f"{split_name} scene hash does not match")
        if entry.get("source_frame_ids_sha256") != sha256_strings(frame_ids):
            errors.append(f"{split_name} source frame hash does not match")
    return errors
