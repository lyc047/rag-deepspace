#!/usr/bin/env python
"""Preflight the S6.7b grid without loading spectrum signal values."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


_DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "configs"
    / "stage6_matched_reliability_development_v1.json"
)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify_hash(path: Path, expected: str) -> dict[str, Any]:
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    return {
        "path": path.as_posix(),
        "exists": exists,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "matches": exists and actual == expected,
    }


def _semantic_grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = protocol["semantic_working_point_grid"]
    sessions = protocol["session_grid"]
    rows = []
    for values in itertools.product(
        grid["heartbeat_silence_scenes"],
        grid["maximum_state_age_minutes"],
        grid["update_reservation_fraction"],
        grid["task_update_open_loop_attempts"],
        sessions["deployment_modes"],
    ):
        heartbeat, age, reservation, attempts, deployment = values
        rows.append(
            {
                "heartbeat_silence_scenes": heartbeat,
                "maximum_state_age_minutes": age,
                "update_reservation_fraction": reservation,
                "task_update_open_loop_attempts": attempts,
                "deployment_mode": deployment,
                "ack_policy": grid["ack_policy"][0],
            }
        )
    return rows


def _exact_grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = protocol["exact_reference_grid"]
    return [
        {
            "task_packet_open_loop_attempts": attempts,
            "feedback_policy": grid["feedback_policy"],
        }
        for attempts in grid["task_packet_open_loop_attempts"]
    ]


def build_preflight(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    freeze = protocol["upstream_freeze"]
    audit = protocol["governance_audit"]
    freeze_check = _verify_hash(
        _resolve_project_path(freeze["manifest"]),
        freeze["manifest_sha256"],
    )
    audit_check = _verify_hash(
        _resolve_project_path(audit["result"]),
        audit["result_sha256"],
    )
    cache = protocol["development_sources"]["coarse_grid_cache"]
    cache_check = _verify_hash(
        _resolve_project_path(cache["path"]),
        cache["sha256"],
    )
    ranker = protocol["frozen_risk_ranker"]
    ranker_result_check = _verify_hash(
        _resolve_project_path(ranker["selection_result"]),
        ranker["selection_result_sha256"],
    )
    ranker_feature_check = _verify_hash(
        _resolve_project_path(ranker["feature_module"]),
        ranker["feature_module_sha256"],
    )
    grouped_archives = []
    for entry in protocol["development_sources"][
        "grouped_validation_archives"
    ]:
        path = _resolve_project_path(entry["path"])
        grouped_archives.append(
            {
                "site": entry["site"],
                "path": path.as_posix(),
                "exists": path.is_file(),
                "size_bytes": (
                    int(path.stat().st_size) if path.is_file() else None
                ),
                "paired_scene_count_from_audit": entry[
                    "paired_scene_count"
                ],
                "role": entry["role"],
            }
        )

    semantic = _semantic_grid(protocol)
    exact = _exact_grid(protocol)
    n_count = len(protocol["task_grid"]["n_channels"])
    session_count = len(protocol["session_grid"]["scene_counts"])
    trajectories = int(protocol["monte_carlo"]["trajectories"])
    coarse_scene_count = min(
        int(max(protocol["session_grid"]["scene_counts"])),
        364,
    )
    estimated_semantic_scene_evaluations = (
        len(semantic)
        * n_count
        * session_count
        * trajectories
        * coarse_scene_count
    )
    estimated_exact_scene_evaluations = (
        len(exact)
        * n_count
        * session_count
        * trajectories
        * coarse_scene_count
    )
    external_forbidden = (
        not protocol["development_sources"][
            "external_final_archives_may_be_opened"
        ]
        and not protocol["development_sources"][
            "external_final_signal_values_may_be_loaded"
        ]
        and protocol["safety_boundary"][
            "external_final_access_count_must_remain_zero"
        ]
    )
    required = set(protocol["bit_accounting"]["required_fields"])
    identity_fields = {
        "actual_forward_bits",
        "actual_feedback_bits",
        "actual_total_application_bits",
        "reserved_capacity_bits",
        "used_reserved_capacity_bits",
        "unused_reserved_capacity_bits",
        "resource_equivalent_bits",
    }
    checks = {
        "protocol_status_is_pre_registered": (
            protocol["status"]
            == "pre_registered_before_s6_7b_result_access"
        ),
        "freeze_hash_matches": freeze_check["matches"],
        "governance_audit_hash_matches": audit_check["matches"],
        "coarse_grid_cache_hash_matches": cache_check["matches"],
        "frozen_risk_ranker_hashes_match": (
            ranker_result_check["matches"]
            and ranker_feature_check["matches"]
        ),
        "all_grouped_archives_exist": all(
            entry["exists"] for entry in grouped_archives
        ),
        "external_final_is_forbidden": external_forbidden,
        "bit_identity_fields_are_required": identity_fields <= required,
        "common_random_numbers_are_enabled": protocol["monte_carlo"][
            "paired_common_random_numbers"
        ],
        "paired_random_stream_layout_is_complete": (
            protocol["monte_carlo"]["random_stream_count"] == 11
            and protocol["monte_carlo"]["random_stream_layout"][
                "nested_task_packet_attempts"
            ]
            == [3, 4, 5]
        ),
        "final_rematching_is_forbidden": (
            not protocol["matched_clean"][
                "final_may_rematch_on_final_clean"
            ]
        ),
    }
    return {
        "version": "1.0",
        "preflight_id": "stage6_matched_reliability_preflight_v1",
        "status": (
            "ready_for_instrumentation_dry_run"
            if all(checks.values())
            else "preflight_failed"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": protocol_path.relative_to(
            PROJECT_DIR
        ).as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "checks": checks,
        "upstream": {
            "freeze": freeze_check,
            "governance_audit": audit_check,
            "coarse_grid_cache": cache_check,
            "frozen_risk_ranker_result": ranker_result_check,
            "frozen_risk_ranker_feature_module": ranker_feature_check,
        },
        "grouped_validation_archives": grouped_archives,
        "grid": {
            "semantic_working_points_per_n": len(semantic),
            "exact_working_points_per_n": len(exact),
            "n_count": n_count,
            "session_length_count": session_count,
            "semantic_session_working_points_per_n": (
                len(semantic) * session_count
            ),
            "exact_session_working_points_per_n": (
                len(exact) * session_count
            ),
            "semantic_working_points_all_n": (
                len(semantic) * n_count * session_count
            ),
            "exact_working_points_all_n": (
                len(exact) * n_count * session_count
            ),
            "estimated_coarse_semantic_scene_evaluations": (
                estimated_semantic_scene_evaluations
            ),
            "estimated_coarse_exact_scene_evaluations": (
                estimated_exact_scene_evaluations
            ),
            "semantic_examples": semantic[:3] + semantic[-3:],
            "exact_points": exact,
        },
        "result_access": {
            "s6_7b_results_loaded": False,
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
        },
        "next_action": (
            "implement a new versioned instrumentation layer and run a "
            "small synthetic or excluded-pilot dry-run before the full grid"
        ),
        "environment": environment_snapshot([]),
        "claim_boundary": (
            "This preflight enumerates configuration metadata and verifies "
            "hashes only. It does not load spectrum values or produce "
            "performance evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=_DEFAULT_PROTOCOL,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = (
        args.output.resolve()
        if args.output is not None
        else _resolve_project_path(protocol["outputs"]["preflight"])
    )
    result = build_preflight(protocol_path, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
