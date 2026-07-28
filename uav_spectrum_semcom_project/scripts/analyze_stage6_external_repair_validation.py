#!/usr/bin/env python
"""Analyze the non-confirmatory Stage-6 external repair validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from analyze_stage6_external_final import analyze_n  # noqa: E402
from run_stage6_external_repair_validation import (  # noqa: E402
    verify_repair_snapshot,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6_external_repair_validation_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol_path = (
        PROJECT_DIR / config["parent_final"]["protocol"]["path"]
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    snapshot_path = args.snapshot or PROJECT_DIR / config["execution"][
        "freeze_snapshot"
    ]
    raw_path = args.raw or PROJECT_DIR / config["execution"]["raw_output"]
    output = args.output or PROJECT_DIR / config["execution"]["analysis_output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite repair-validation analysis")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    errors = verify_repair_snapshot(snapshot)
    if errors:
        raise ValueError("repair snapshot changed: " + "; ".join(errors))
    if (
        raw.get("status")
        != "stage6_external_repair_validation_execution_complete"
        or raw.get("classification") != config["classification"]
        or raw.get("repair_code_snapshot_sha256")
        != snapshot.get("executable_snapshot_sha256")
        or raw.get("scene_count") != 200
    ):
        raise ValueError("repair-validation raw result is incomplete or unbound")

    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    analysis_by_n = {
        str(n_channels): analyze_n(
            n_channels=n_channels,
            n_position=n_position,
            raw_n=raw["n_results"][str(n_channels)],
            protocol=protocol,
        )
        for n_position, n_channels in enumerate(n_values)
    }
    target = float(
        protocol["system_evaluation"]["primary_matched_clean_target"]
    )
    primary_by_n = {}
    for n_channels in n_values:
        comparisons = analysis_by_n[str(n_channels)][
            "matched_clean_comparisons"
        ]
        matches = [
            value
            for value in comparisons
            if abs(float(value["target_clean_rate"]) - target) < 1e-12
        ]
        primary_by_n[str(n_channels)] = (
            matches[0]
            if matches
            else {
                "status": "missing",
                "passes_primary_acceptance": False,
                "passes_strong_effect": False,
            }
        )
    passing_n = [
        n_channels
        for n_channels in n_values
        if primary_by_n[str(n_channels)].get(
            "passes_primary_acceptance", False
        )
    ]
    primary_n = int(
        protocol["decision_rules"]["primary_configuration"]["n_channels"]
    )
    all_campaigns = set(raw["campaign_ids"]) == set(
        protocol["sampling"]["campaign_allocations"]
    )
    wrong_zero = bool(
        raw["checks"]["wrong_codebook_actions_must_be_zero"]
    )
    scale_support = len(passing_n) >= 3
    primary_pass = primary_by_n[str(primary_n)].get(
        "passes_primary_acceptance", False
    )
    descriptive_support = bool(
        primary_pass and scale_support and all_campaigns and wrong_zero
    )
    result = {
        "version": "1.0",
        "status": "stage6_external_repair_validation_analysis_complete",
        "verification_status": "ANALYZED",
        "classification": config["classification"],
        "validation_config_sha256": sha256_file(args.config),
        "parent_protocol_sha256": sha256_file(protocol_path),
        "raw_result_sha256": sha256_file(raw_path),
        "repair_code_snapshot_sha256": snapshot[
            "executable_snapshot_sha256"
        ],
        "bootstrap": {
            "replicates": int(
                protocol["system_evaluation"][
                    "paired_bootstrap_replicates"
                ]
            ),
            "confidence_level": float(
                protocol["system_evaluation"]["confidence_level"]
            ),
            "outer_unit": "campaign",
            "inner_unit": "paired_link_trajectory",
        },
        "analysis_by_n": analysis_by_n,
        "decision_summary": {
            "primary_n_channels": primary_n,
            "primary_target_clean_rate": target,
            "original_primary_acceptance_pattern_met": bool(primary_pass),
            "n_values_meeting_original_acceptance_pattern": passing_n,
            "n_values_meeting_count": len(passing_n),
            "original_scale_support_pattern_met": scale_support,
            "all_three_campaigns_present": all_campaigns,
            "wrong_codebook_actions_zero": wrong_zero,
            "stage6_candidate_supported_by_repair_validation": (
                descriptive_support
            ),
            "external_confirmatory_final_passed": None,
            "new_single_access_final_still_required": True,
        },
        "checks": {
            "complete_registered_200_scene_catalog_used": raw["checks"][
                "complete_registered_200_scene_catalog_used"
            ],
            "all_registered_n_retained": raw["checks"][
                "all_registered_n_retained"
            ],
            "all_methods_used_identical_scene_catalog": raw["checks"][
                "all_methods_used_identical_scene_catalog"
            ],
            "repair_snapshot_verified": True,
            "raw_result_hash_recorded": True,
            "adapter_only_repair": raw["checks"]["adapter_only_repair"],
        },
        "interpretation": config["interpretation"],
        "claim_boundary": raw["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "stage6_candidate_supported_by_repair_validation": (
                    descriptive_support
                ),
                "original_primary_acceptance_pattern_met": bool(primary_pass),
                "n_values_meeting_original_acceptance_pattern": passing_n,
                "external_confirmatory_final_passed": None,
                "new_single_access_final_still_required": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
