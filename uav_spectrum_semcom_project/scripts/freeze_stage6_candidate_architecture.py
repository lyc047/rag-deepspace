#!/usr/bin/env python
"""Verify and freeze the Stage-6 candidate architecture manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_candidate_architecture_freeze_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/candidate_architecture_freeze_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "candidate_architecture_freeze_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite architecture freeze")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    retained = set(manifest["retained_components"])
    excluded = set(manifest["excluded_components"])
    code_checks = {
        row["path"]: sha256_file(PROJECT_DIR / row["path"])
        == row["sha256"]
        for row in manifest["locked_code_artifacts"]
    }
    evidence_checks = {
        row["path"]: sha256_file(PROJECT_DIR / row["path"])
        == row["sha256"]
        for row in manifest["locked_evidence_artifacts"]
    }
    governance = manifest["freeze_governance"]
    access_path = PROJECT_DIR / governance["final_access_state_path"]
    access_state = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "retained_and_excluded_components_disjoint": not bool(
            retained & excluded
        ),
        "all_locked_code_hashes_match": all(code_checks.values()),
        "all_locked_evidence_hashes_match": all(
            evidence_checks.values()
        ),
        "final_access_state_hash_matches": (
            sha256_file(access_path)
            == governance["final_access_state_sha256"]
        ),
        "final_access_count_remains_zero": (
            access_state["access_count"]
            == int(governance["required_final_access_count"])
            and not access_state["final_signal_values_accessed"]
            and not access_state["final_method_outputs_accessed"]
        ),
        "post_freeze_algorithm_addition_forbidden": not governance[
            "algorithm_modules_may_be_added_after_freeze"
        ],
        "post_freeze_parameter_tuning_forbidden": not governance[
            "existing_parameters_may_be_tuned_after_freeze"
        ],
        "defect_requires_new_freeze_version": governance[
            "defect_correction_requires_new_freeze_version"
        ],
        "external_final_forbidden_during_s6_6": not governance[
            "external_final_may_be_accessed_during_s6_6"
        ],
    }
    if not all(checks.values()):
        raise AssertionError(f"candidate freeze failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_candidate_architecture_frozen",
        "freeze_id": manifest["freeze_id"],
        "manifest_sha256": sha256_file(args.manifest),
        "retained_components": manifest["retained_components"],
        "excluded_components": manifest["excluded_components"],
        "frozen_parameters": manifest["frozen_parameters"],
        "deployment_boundary": manifest["deployment_boundary"],
        "locked_code_hash_checks": code_checks,
        "locked_evidence_hash_checks": evidence_checks,
        "validation_matrix": manifest["s6_6_validation_matrix"],
        "checks": checks,
        "final_access_count": access_state["access_count"],
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": manifest["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "freeze_id": result["freeze_id"],
                "manifest_sha256": result["manifest_sha256"],
                "retained_component_count": len(retained),
                "excluded_component_count": len(excluded),
                "checks": checks,
                "final_access_count": access_state["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
