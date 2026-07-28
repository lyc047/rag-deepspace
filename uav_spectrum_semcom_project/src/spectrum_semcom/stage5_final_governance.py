"""Governance primitives for the Stage-5 unified external Final."""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from .final_holdout import canonical_json_sha256, sha256_path


_TIMESTAMP = re.compile(r"(\d{8})[_-](\d{6})")


def _safe_member(name: str) -> PurePosixPath:
    value = PurePosixPath(name.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    return value


def discover_timestamped_sigmf_pairs(
    archive_path: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """List paired sweep members without reading metadata or signal bytes."""

    archive_path = Path(archive_path)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as handle:
        infos = {}
        for info in handle.infolist():
            try:
                member = _safe_member(info.filename)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if info.is_dir() or "__MACOSX" in member.parts or member.name.startswith("._"):
                continue
            infos[member.as_posix()] = info
        for name in sorted(value for value in infos if value.endswith(".sigmf-meta")):
            member = PurePosixPath(name)
            match = _TIMESTAMP.search(member.name)
            if match is None:
                continue
            data_name = member.with_name(
                member.name.removesuffix(".sigmf-meta") + ".sigmf-data"
            ).as_posix()
            if data_name not in infos:
                errors.append(f"missing data pair: {name}")
                continue
            meta_info = infos[name]
            data_info = infos[data_name]
            rows.append(
                {
                    "stem": member.name.removesuffix(".sigmf-meta"),
                    "timestamp_local": datetime.strptime(
                        "".join(match.groups()), "%Y%m%d%H%M%S"
                    ).isoformat(),
                    "meta_member": name,
                    "data_member": data_name,
                    "meta_size_bytes": int(meta_info.file_size),
                    "data_size_bytes": int(data_info.file_size),
                    "meta_crc32": f"{meta_info.CRC:08x}",
                    "data_crc32": f"{data_info.CRC:08x}",
                }
            )
    return rows, errors


def select_spaced_evenly(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    minimum_spacing_s: float,
) -> list[dict[str, Any]]:
    """Greedy time thinning followed by deterministic even coverage."""

    if count < 1 or minimum_spacing_s < 0:
        raise ValueError("invalid selection count or spacing")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (row["timestamp_local"], row["stem"]),
    )
    thinned = []
    last: datetime | None = None
    for row in ordered:
        current = datetime.fromisoformat(row["timestamp_local"])
        if last is None or (current - last).total_seconds() >= minimum_spacing_s:
            thinned.append(row)
            last = current
    if len(thinned) < count:
        raise ValueError(
            f"spacing leaves {len(thinned)} rows, fewer than requested {count}"
        )
    indices = np.linspace(0, len(thinned) - 1, count, dtype=int)
    if len(set(indices.tolist())) != count:
        raise AssertionError("even-spread selection produced duplicate indices")
    return [thinned[int(index)] for index in indices]


def expected_cluster_id(
    campaign_id: str, timestamp_local: str, cluster_minutes: int
) -> str:
    value = datetime.fromisoformat(timestamp_local)
    if cluster_minutes < 1 or 60 % cluster_minutes != 0:
        raise ValueError("cluster_minutes must divide one hour")
    minute = (value.minute // cluster_minutes) * cluster_minutes
    return f"{campaign_id}:{value:%Y%m%dT%H}{minute:02d}"


def validate_external_final_catalog(
    catalog: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    sampling = protocol["sampling"]
    scenes = catalog.get("scenes")
    if catalog.get("status") != "candidate_external_final_catalog":
        errors.append("catalog status must be candidate_external_final_catalog")
    if not isinstance(scenes, list):
        return errors + ["catalog scenes must be a list"]
    if len(scenes) != int(sampling["target_scene_count"]):
        errors.append("catalog scene count differs from frozen target")
    ids = [row.get("scene_id") for row in scenes if isinstance(row, Mapping)]
    if len(ids) != len(set(ids)):
        errors.append("scene IDs must be unique")
    eligible = {row["dataset_id"] for row in registry["final_candidates"]}
    counts = Counter()
    per_campaign: dict[str, list[datetime]] = {}
    provenance = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            errors.append(f"scene[{index}] must be an object")
            continue
        campaign = scene.get("campaign_id")
        if campaign not in eligible:
            errors.append(f"scene[{index}] uses ineligible campaign")
            continue
        counts[campaign] += 1
        try:
            timestamp = datetime.fromisoformat(str(scene["timestamp_local"]))
        except (KeyError, ValueError):
            errors.append(f"scene[{index}] has invalid timestamp")
            continue
        per_campaign.setdefault(campaign, []).append(timestamp)
        expected_cluster = expected_cluster_id(
            campaign,
            timestamp.isoformat(),
            int(sampling["cluster_minutes"]),
        )
        if scene.get("cluster_id") != expected_cluster:
            errors.append(f"scene[{index}] has invalid cluster ID")
        key = (
            scene.get("archive_sha256"),
            scene.get("meta_member"),
            scene.get("data_member"),
        )
        if key in provenance:
            errors.append(f"scene[{index}] duplicates archive provenance")
        provenance.add(key)
        if scene.get("selection_used_signal_values") is not False:
            errors.append(f"scene[{index}] selection signal-value flag is invalid")
        for field in (
            "archive_path",
            "archive_sha256",
            "meta_member",
            "data_member",
            "meta_crc32",
            "data_crc32",
        ):
            if not isinstance(scene.get(field), str) or not scene[field]:
                errors.append(f"scene[{index}].{field} is required")
    if dict(counts) != {
        key: int(value)
        for key, value in sampling["campaign_allocations"].items()
    }:
        errors.append("campaign allocation differs from frozen protocol")
    minimum = float(sampling["minimum_spacing_minutes"]) * 60.0
    for campaign, timestamps in per_campaign.items():
        ordered = sorted(timestamps)
        if any(
            (right - left).total_seconds() < minimum
            for left, right in zip(ordered, ordered[1:])
        ):
            errors.append(f"{campaign} violates minimum spacing")
    if catalog.get("signal_values_loaded_for_selection") is not False:
        errors.append("catalog must declare signal-free selection")
    if catalog.get("method_outputs_loaded_for_selection") is not False:
        errors.append("catalog must declare method-output-free selection")
    return errors


def build_code_snapshot(
    project_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = protocol["freeze"]
    paths = set()
    for pattern in freeze["include_globs"]:
        paths.update(
            path for path in project_dir.glob(pattern) if path.is_file()
        )
    paths.update(project_dir / value for value in freeze["include_files"])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot inputs missing: {missing}")
    files = [
        {
            "path": path.relative_to(project_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return {
        "version": "1.0",
        "status": "stage5_external_final_code_frozen",
        "protocol_id": protocol["protocol_id"],
        "files": files,
        "executable_snapshot_sha256": canonical_json_sha256(files),
    }


def verify_code_snapshot(
    project_dir: Path, snapshot: Mapping[str, Any]
) -> list[str]:
    errors = []
    files = snapshot.get("files", [])
    if (
        snapshot.get("status") != "stage5_external_final_code_frozen"
        or canonical_json_sha256(files)
        != snapshot.get("executable_snapshot_sha256")
    ):
        errors.append("snapshot identity or canonical hash is invalid")
    for row in files:
        path = project_dir / row["path"]
        if not path.is_file():
            errors.append(f"missing frozen file: {row['path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"frozen file size changed: {row['path']}")
        elif sha256_path(path) != row["sha256"]:
            errors.append(f"frozen file SHA-256 changed: {row['path']}")
    return errors
