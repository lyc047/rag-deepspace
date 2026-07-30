#!/usr/bin/env python
"""Metadata-only dry-run for the frozen Stage-6R confirmation lockbox."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import read_json, sha256_file  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.stage6r_confirmation_governance import (  # noqa: E402
    validate_execution_preconditions,
)
from spectrum_semcom.stage6r_external_final import code_snapshot  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_confirmation_lockbox_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/confirmation_lockbox_dry_run_v1/result.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite lockbox dry-run")

    config = read_json(args.config)
    paths = {
        name: Path(value) if name == "archive" else PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    registry = read_json(paths["registry"])
    access = read_json(paths["access_state"])
    freeze = read_json(paths["freeze_result"])
    snapshot = code_snapshot(PROJECT_DIR, config["code_snapshot_paths"])
    role, sites = validate_execution_preconditions(
        config=config,
        registry=registry,
        access_state=access,
        freeze=freeze,
        protocol_sha256=sha256_file(args.config),
        code_snapshot=snapshot
    )
    if role != "confirmation_lockbox":
        raise ValueError("dry-run is restricted to confirmation_lockbox")
    if paths["archive"].stat().st_size != registry["archive"]["size_bytes"]:
        raise ValueError("archive size differs from registry")

    metadata_checks = []
    for site in sites:
        row = registry["selected_site_members"][site]
        passed = bool(
            row["technology"] == config["data_adapter"]["technology"]
            and row["shape"][0] >= 200
            and row["shape"][1] >= 64
            and row["member"].endswith(".npy")
        )
        metadata_checks.append(
            {
                "site": site,
                "member": row["member"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "metadata_gate_passed": passed
            }
        )
    heartbeat = {
        n_key: int(
            artifact["workpoints"]["semantic"][
                "heartbeat_silence_scenes"
            ]
        )
        for n_key, artifact in freeze["n_artifacts"].items()
    }
    expected_heartbeat = {
        str(key): int(value)
        for key, value in config["reliability"][
            "heartbeat_silence_scenes_by_n"
        ].items()
    }
    checks = {
        "all_six_registered_sites_present": len(sites) == 6,
        "all_metadata_gates_passed": all(
            row["metadata_gate_passed"] for row in metadata_checks
        ),
        "heartbeat_map_matches_protocol": heartbeat == expected_heartbeat,
        "confirmation_lockbox_access_count_zero": (
            access["roles"]["confirmation_lockbox"]["access_count"] == 0
        ),
        "reserve_access_count_zero": (
            access["roles"]["reserve"]["access_count"] == 0
        ),
        "archive_signal_values_loaded": False,
        "output_result_absent": not (
            PROJECT_DIR
            / "results/stage6r/confirmation_lockbox_v1/result.json"
        ).exists()
    }
    passed = all(
        value for key, value in checks.items()
        if key != "archive_signal_values_loaded"
    ) and checks["archive_signal_values_loaded"] is False
    result = {
        "version": "1.0",
        "status": "stage6r_confirmation_lockbox_dry_run_complete",
        "verification_status": "PASSED" if passed else "FAILED",
        "experiment_id": config["experiment_id"],
        "protocol_sha256": freeze["protocol_sha256"],
        "freeze_result_sha256": sha256_file(paths["freeze_result"]),
        "code_snapshot_sha256": snapshot["combined_sha256"],
        "execution_role": role,
        "sites": sites,
        "metadata_checks": metadata_checks,
        "heartbeat_silence_scenes_by_n": heartbeat,
        "checks": checks,
        "signal_values_accessed": False,
        "next_action": (
            "eligible_for_explicit_single_consume"
            if passed
            else "repair_protocol_without_signal_access"
        ),
        "claim_boundary": config["claim_boundary"]
    }
    atomic_write_json(args.output, result)
    print(json.dumps(
        {
            "output": str(args.output),
            "verification_status": result["verification_status"],
            "signal_values_accessed": False,
            "next_action": result["next_action"]
        },
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
