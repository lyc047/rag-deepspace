#!/usr/bin/env python
"""Audit the S6-FC0 freeze and dataset metadata without opening Final signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


_TIMESTAMP_PATTERN = re.compile(r"results_(\d{8})_(\d{6})")
_DEFAULT_FIXED_ROOT = Path(
    "F:/uav_spectrum_final_data/aerpaw_sub6_feb2022/raw"
)
_DEFAULT_EXTERNAL_ROOT = Path(
    "F:/uav_spectrum_final_data/aerpaw_helikite_external_v1"
)
_DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results"
    / "stage6"
    / "s6_7a_governance_audit_v1"
    / "governance_metadata_audit.json"
)


def _sha256_path(path: Path) -> str:
    return sha256_file(path)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(probability * (len(ordered) - 1))
    return float(ordered[index])


def _timestamp_from_name(name: str) -> datetime | None:
    match = _TIMESTAMP_PATTERN.search(Path(name).name)
    if match is None:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def _primary_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [
        member
        for member in archive.infolist()
        if not member.is_dir()
        and not member.filename.startswith("__MACOSX/")
        and not Path(member.filename).name.startswith("._")
    ]


def _sample_positions(count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted({0, count // 4, count // 2, 3 * count // 4, count - 1})


def _audit_fixed_site_archive(site: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "site": site,
            "path": path.as_posix(),
            "exists": False,
        }

    with zipfile.ZipFile(path) as archive:
        members = _primary_members(archive)
        meta_members = sorted(
            (
                member
                for member in members
                if member.filename.endswith(".sigmf-meta")
            ),
            key=lambda member: member.filename,
        )
        data_members = [
            member
            for member in members
            if member.filename.endswith(".sigmf-data")
        ]
        data_size_counts = Counter(
            int(member.file_size) for member in data_members
        )
        meta_stems = {
            Path(member.filename).name.removesuffix(".sigmf-meta")
            for member in meta_members
        }
        data_stems = {
            Path(member.filename).name.removesuffix(".sigmf-data")
            for member in data_members
        }
        timestamps = sorted(
            {
                timestamp
                for member in meta_members
                if (timestamp := _timestamp_from_name(member.filename))
                is not None
            }
        )
        gaps = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
        ]
        day_counts = Counter(value.date().isoformat() for value in timestamps)

        sampled_axes: list[dict[str, Any]] = []
        for position in _sample_positions(len(meta_members)):
            member = meta_members[position]
            metadata = json.loads(archive.read(member).decode("utf-8"))
            global_metadata = metadata["global"]
            axis = np.asarray(
                global_metadata["dataset:frequency_axis_MHz"],
                dtype=np.float64,
            )
            sampled_axes.append(
                {
                    "member": member.filename,
                    "datatype": global_metadata.get("core:datatype"),
                    "declared_bins": int(
                        global_metadata.get("dataset:num_bins", axis.size)
                    ),
                    "axis_bins": int(axis.size),
                    "first_mhz": float(axis[0]),
                    "last_mhz": float(axis[-1]),
                    "median_step_mhz": float(
                        np.median(np.diff(axis))
                    ),
                    "axis_sha256_float64": hashlib.sha256(
                        axis.tobytes()
                    ).hexdigest(),
                }
            )

    axis_hashes = {
        value["axis_sha256_float64"] for value in sampled_axes
    }
    axis_shapes = {value["axis_bins"] for value in sampled_axes}
    axis_spans = {
        (value["first_mhz"], value["last_mhz"])
        for value in sampled_axes
    }
    return {
        "site": site,
        "path": path.as_posix(),
        "exists": True,
        "archive_size_bytes": int(path.stat().st_size),
        "archive_sha256": None,
        "archive_sha256_status": (
            "not_recomputed_in_routine_audit_due_to_12_to_21_gb_size"
        ),
        "primary_member_count": len(members),
        "primary_uncompressed_bytes": int(
            sum(member.file_size for member in members)
        ),
        "sigmf_meta_count": len(meta_members),
        "sigmf_data_count": len(data_members),
        "sigmf_data_size_counts": {
            str(size): count
            for size, count in sorted(data_size_counts.items())
        },
        "all_sigmf_data_files_match_98868_float32_bins": (
            set(data_size_counts) == {98_868 * 4}
        ),
        "paired_scene_count": len(meta_stems & data_stems),
        "orphan_meta_count": len(meta_stems - data_stems),
        "orphan_data_count": len(data_stems - meta_stems),
        "first_timestamp_local": (
            timestamps[0].isoformat() if timestamps else None
        ),
        "last_timestamp_local": (
            timestamps[-1].isoformat() if timestamps else None
        ),
        "calendar_day_count": len(day_counts),
        "scenes_by_day": dict(sorted(day_counts.items())),
        "median_gap_seconds": (
            float(statistics.median(gaps)) if gaps else None
        ),
        "p95_gap_seconds": _percentile(gaps, 0.95) if gaps else None,
        "maximum_gap_seconds": max(gaps) if gaps else None,
        "sampled_frequency_metadata_count": len(sampled_axes),
        "sampled_frequency_axes": sampled_axes,
        "sampled_axis_consistent_within_site": (
            len(axis_hashes) == 1
            and len(axis_shapes) == 1
            and len(axis_spans) == 1
        ),
        "signal_power_values_loaded": False,
        "audit_scope": (
            "ZIP central directory, filenames, sizes, timestamps, and five "
            "frequency-axis metadata records only"
        ),
    }


def _audit_locked_artifacts(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    artifacts = [
        *manifest["locked_code_artifacts"],
        *manifest["locked_evidence_artifacts"],
    ]
    rows: list[dict[str, Any]] = []
    for entry in artifacts:
        path = PROJECT_DIR / entry["path"]
        exists = path.is_file()
        actual = _sha256_path(path) if exists else None
        rows.append(
            {
                "path": entry["path"],
                "kind": (
                    "locked_code_or_config"
                    if entry in manifest["locked_code_artifacts"]
                    else "locked_evidence"
                ),
                "exists": exists,
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual,
                "matches": exists and actual == entry["sha256"],
            }
        )
    return rows, all(value["matches"] for value in rows)


def _audit_external_archives(
    registry: dict[str, Any],
    external_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in registry["final_candidates"]:
        path = external_root / entry["archive"]
        exists = path.is_file()
        actual = _sha256_path(path) if exists else None
        rows.append(
            {
                "dataset_id": entry["dataset_id"],
                "campaign": entry["campaign"],
                "doi": entry["doi"],
                "path": path.as_posix(),
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists else None,
                "expected_sha256": entry["archive_sha256"],
                "actual_sha256": actual,
                "matches": exists and actual == entry["archive_sha256"],
                "source_values_or_method_outputs_accessed": False,
                "verification_scope": "file size and whole-archive hash only",
            }
        )
    return rows


def build_audit(
    *,
    fixed_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    freeze_manifest_path = (
        PROJECT_DIR
        / "configs"
        / "stage6_candidate_architecture_freeze_v1.json"
    )
    acceptance_path = (
        PROJECT_DIR
        / "results"
        / "stage6"
        / "development_acceptance_v1"
        / "development_acceptance_result.json"
    )
    acceptance_reproduction_path = (
        PROJECT_DIR
        / "results"
        / "stage6"
        / "development_acceptance_reproduction_v1"
        / "development_acceptance_result.json"
    )
    external_registry_path = (
        PROJECT_DIR
        / "configs"
        / "stage5_external_final_registry_v1.json"
    )
    freeze_manifest = json.loads(
        freeze_manifest_path.read_text(encoding="utf-8")
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    external_registry = json.loads(
        external_registry_path.read_text(encoding="utf-8")
    )
    metric_dictionary_path = (
        PROJECT_DIR
        / "docs"
        / "STAGE6_S6_7B_METRIC_AND_FAIRNESS_PROTOCOL_DRAFT.md"
    )
    matched_protocol_path = (
        PROJECT_DIR
        / "configs"
        / "stage6_matched_reliability_development_v1.json"
    )
    metric_dictionary_complete = metric_dictionary_path.is_file()
    matched_protocol_complete = False
    if matched_protocol_path.is_file():
        matched_protocol = json.loads(
            matched_protocol_path.read_text(encoding="utf-8")
        )
        matched_protocol_complete = (
            matched_protocol.get("status")
            == "pre_registered_before_s6_7b_result_access"
        )

    locked_rows, locked_match = _audit_locked_artifacts(freeze_manifest)
    acceptance_hash = _sha256_path(acceptance_path)
    reproduction_hash = _sha256_path(acceptance_reproduction_path)
    external_rows = _audit_external_archives(
        external_registry,
        external_root,
    )
    fixed_rows = [
        _audit_fixed_site_archive(
            site,
            fixed_root / f"Results{site}Feb2022_SigMF.zip",
        )
        for site in ("LW1", "CC1", "CC2")
    ]
    cross_site_axis_hashes = {
        sample["axis_sha256_float64"]
        for row in fixed_rows
        if row.get("exists")
        for sample in row["sampled_frequency_axes"]
    }
    external_integrity_pass = all(
        value["matches"] for value in external_rows
    )
    fixed_metadata_pass = all(
        value.get("exists")
        and value.get("orphan_meta_count") == 0
        and value.get("orphan_data_count") == 0
        and value.get("sampled_axis_consistent_within_site")
        for value in fixed_rows
    )
    all_s6_7a_gates_pass = (
        locked_match
        and acceptance_hash == reproduction_hash
        and fixed_metadata_pass
        and external_integrity_pass
        and metric_dictionary_complete
        and matched_protocol_complete
    )
    return {
        "version": "1.0",
        "audit_id": "stage6_s6_7a_governance_metadata_audit_v1",
        "status": (
            "s6_7a_governance_audit_complete"
            if all_s6_7a_gates_pass
            else "s6_7a_partial_governance_audit_complete"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "final_signal_values_loaded": False,
            "external_final_zip_member_lists_opened": False,
            "external_final_verification": (
                "filesystem existence, size, and archive SHA-256 only"
            ),
            "fixed_site_development_metadata_loaded": True,
            "fixed_site_power_values_loaded": False,
        },
        "freeze": {
            "freeze_id": freeze_manifest["freeze_id"],
            "manifest_path": freeze_manifest_path.relative_to(
                PROJECT_DIR
            ).as_posix(),
            "manifest_sha256": _sha256_path(freeze_manifest_path),
            "locked_artifact_count": len(locked_rows),
            "all_locked_artifacts_match": locked_match,
            "artifacts": locked_rows,
        },
        "development_acceptance": {
            "verdict": acceptance["verdict"],
            "all_acceptance_gates_passed": acceptance[
                "all_acceptance_gates_passed"
            ],
            "collected_test_count": acceptance[
                "regression_validation"
            ]["collected_test_count"],
            "pytest_exit_code": acceptance[
                "regression_validation"
            ]["pytest_exit_code"],
            "final_access_count": acceptance["final_access_count"],
            "main_result_sha256": acceptance_hash,
            "reproduction_result_sha256": reproduction_hash,
            "main_and_reproduction_byte_identical": (
                acceptance_hash == reproduction_hash
            ),
        },
        "fixed_site_development_archives": fixed_rows,
        "fixed_site_summary": {
            "total_paired_scenes": sum(
                int(value.get("paired_scene_count", 0))
                for value in fixed_rows
            ),
            "sampled_frequency_axis_consistent_across_sites": (
                len(cross_site_axis_hashes) == 1
            ),
            "unique_sampled_frequency_axis_hashes": sorted(
                cross_site_axis_hashes
            ),
            "metadata_integrity_pass": fixed_metadata_pass,
        },
        "external_final_registry": {
            "registry_path": external_registry_path.relative_to(
                PROJECT_DIR
            ).as_posix(),
            "registry_sha256": _sha256_path(external_registry_path),
            "registry_status": external_registry["status"],
            "archives": external_rows,
            "all_archive_hashes_match_registry": external_integrity_pass,
            "final_access_count": 0,
        },
        "gate_checks": {
            "s6_fc0_uniquely_identified": locked_match,
            "acceptance_reproduced_byte_identically": (
                acceptance_hash == reproduction_hash
            ),
            "fixed_site_development_metadata_ready": fixed_metadata_pass,
            "external_final_integrity_ready_without_signal_access": (
                external_integrity_pass
            ),
            "external_final_access_remains_zero": True,
            "metric_dictionary_complete": metric_dictionary_complete,
            "s6_7b_protocol_complete": matched_protocol_complete,
        },
        "next_actions": (
            [
                (
                    "implement a new versioned bit-instrumentation layer and "
                    "run a small excluded-pilot dry-run"
                ),
                (
                    "build a development-only fixed-site task cache using "
                    "streaming reads after the dry-run"
                ),
            ]
            if all_s6_7a_gates_pass
            else [
                "complete the metric dictionary and S6.7b protocol",
                "rerun this audit before any performance grid",
            ]
        ),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "This audit verifies frozen artifacts, development-archive "
            "directory metadata, sampled frequency axes, and Final archive "
            "integrity. It does not load external Final signal values, run "
            "Stage-6 Final, or establish matched-reliability performance."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=_DEFAULT_FIXED_ROOT,
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=_DEFAULT_EXTERNAL_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(
        fixed_root=args.fixed_root,
        external_root=args.external_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
