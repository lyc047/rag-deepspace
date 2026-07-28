#!/usr/bin/env python
"""Evaluate semantic belief-set recovery without accessing Final data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage5_ack_state_recovery_development import (  # noqa: E402
    load_json,
    load_npz,
    run_fault_condition,
    validate_inputs,
)
from run_stage5_dynamic_query_development import build_query_schedule  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_spectrum_stress import apply_spectrum_regime  # noqa: E402


def validate_predecessor(protocol: dict) -> None:
    predecessor = protocol["ack_recovery_predecessor"]
    result_path = PROJECT_DIR / predecessor["result"]
    if sha256_file(result_path) != predecessor["result_sha256"]:
        raise ValueError("ACK-recovery predecessor result hash mismatch")
    result = load_json(result_path)
    if result["governance"]["confirmatory_final"]:
        raise ValueError("belief-risk development cannot consume Final evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_belief_risk_recovery_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/belief_risk_recovery_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite belief-risk output")
    protocol = load_json(args.protocol)
    validate_predecessor(protocol)
    cache, stage2, schedule_protocol = validate_inputs(protocol)
    transformed = dict(cache)
    transformed["channel_power_dbm"] = apply_spectrum_regime(
        cache["channel_power_dbm"],
        cache["site_ids"],
        regime=protocol["stress_predecessor"]["regime"],
        parameters=protocol["stress_predecessor"]["parameters"],
    )
    query = build_query_schedule(
        transformed, schedule_protocol, protocol["schedule"]["name"]
    )
    conditions = {}
    condition_index = 0
    for ack_loss in protocol["ack_fault_grid"]["ack_loss_probability"]:
        for duplicate in protocol["ack_fault_grid"][
            "delayed_duplicate_probability"
        ]:
            key = f"ack_loss_{ack_loss:g}_duplicate_{duplicate:g}"
            conditions[key] = run_fault_condition(
                transformed,
                stage2,
                protocol,
                query,
                ack_loss_probability=float(ack_loss),
                duplicate_probability=float(duplicate),
                condition_index=condition_index,
            )
            condition_index += 1
    result = {
        "version": "1.0",
        "status": "stage5_belief_risk_recovery_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "conditions": conditions,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "ack_faults_are_real_measurements": False,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "belief_risk_recovery_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "conditions": len(conditions),
                "final_loaded": False,
                "fixed_policy_retuned": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
