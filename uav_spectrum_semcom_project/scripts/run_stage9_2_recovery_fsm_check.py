#!/usr/bin/env python
"""Run preregistered Stage-9.2 bounded recovery-FSM checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectrum_semcom.stage9_2_recovery_fsm import (  # noqa: E402
    ProtocolDesign,
    bounded_model_check,
    protocol_identity_costs,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "--config", default="configs/stage9_2_recovery_fsm_protocol_v1.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freeze", default="results/stage9_2/stage9_2_candidate_freeze_v1.json"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {
        "config": config_path,
        "runtime": ROOT / "src/spectrum_semcom/stage9_2_recovery_fsm.py",
        "runner": Path(__file__).resolve(),
        "tests": ROOT / "tests/test_stage9_2_recovery_fsm.py",
    }
    hashes = {name: _hash(path) for name, path in paths.items()}
    if args.mode == "confirmation":
        freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
        if freeze["status"] != "stage9_2_candidate_frozen_before_confirmation":
            raise RuntimeError("Stage-9.2 freeze is not active")
        if freeze["input_hashes"] != hashes:
            raise RuntimeError("Stage-9.2 inputs differ from freeze")

    definitions = config["designs"]
    legacy = ProtocolDesign(
        "legacy_epoch_only",
        int(definitions["legacy_epoch_only"]["restore_session_tag_bits"]),
        int(definitions["legacy_epoch_only"]["update_session_tag_bits"]),
        int(definitions["legacy_epoch_only"]["ack_session_tag_bits"]),
    )
    restore = ProtocolDesign(
        "restore_tag16_bounded",
        int(definitions["restore_tag16_bounded"]["restore_session_tag_bits"]),
        int(definitions["restore_tag16_bounded"]["update_session_tag_bits"]),
        int(definitions["restore_tag16_bounded"]["ack_session_tag_bits"]),
    )
    full = ProtocolDesign(
        "full_tag16",
        int(definitions["full_tag16"]["restore_session_tag_bits"]),
        int(definitions["full_tag16"]["update_session_tag_bits"]),
        int(definitions["full_tag16"]["ack_session_tag_bits"]),
    )
    model = config["model_check"]
    depth = int(
        model["development_depth"]
        if args.mode == "development"
        else model["confirmation_depth"]
    )
    common = {
        "depth": depth,
        "serial_bits": int(model["reduced_serial_bits"]),
        "maximum_states": int(model["maximum_states"]),
    }
    checks = {
        "legacy_epoch_only_unbounded": bounded_model_check(
            legacy, allow_ancient_collision=True, **common
        ),
        "restore_tag16_bounded": bounded_model_check(
            restore, allow_ancient_collision=False, **common
        ),
        "restore_tag16_unbounded": bounded_model_check(
            restore, allow_ancient_collision=True, **common
        ),
        "full_tag16_unbounded": bounded_model_check(
            full, allow_ancient_collision=True, **common
        ),
    }
    expected = model["expected_decision_pattern"]
    decisions = {}
    for name, result in checks.items():
        expectation = expected[name]
        passed = bool(
            (expectation == "counterexample_required" and result.counterexample_found)
            or (expectation == "no_counterexample_required" and not result.counterexample_found and not result.truncated)
        )
        decisions[name] = {
            "expectation": expectation,
            "passed": passed,
            "result": result.to_dict(),
        }

    cost_config = config["bit_cost_scenarios"]
    cost_results = {
        str(restores): protocol_identity_costs(
            scene_count=int(cost_config["session_scenes"]),
            ordinary_updates=int(cost_config["ordinary_updates"]),
            restore_count=int(restores),
            compact_update_bits=int(cost_config["base_compact_update_bits"]),
            activation_bits=int(cost_config["base_activation_bits"]),
            ack_bits=int(cost_config["base_ack_bits"]),
        )
        for restores in cost_config["restore_counts"]
    }
    cost_passed = all(
        values["restore_tag16_bounded"]["incremental_bits_per_scene"] < 1.0
        for values in cost_results.values()
    )
    result = {
        "version": "1.0",
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "status": f"stage9_2_{args.mode}_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": hashes,
        "model_depth": depth,
        "serial_bits_in_model": int(model["reduced_serial_bits"]),
        "checks": decisions,
        "bit_cost_scenarios": cost_results,
        "primary_gate": {
            "all_model_decisions_match_registration": all(
                item["passed"] for item in decisions.values()
            ),
            "restore_tag_cost_gate_passed": cost_passed,
            "passed": bool(
                all(item["passed"] for item in decisions.values())
                and cost_passed
            ),
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["primary_gate"], ensure_ascii=False))
    for name, item in decisions.items():
        check = item["result"]
        print(
            name,
            "counterexample=",
            check["counterexample_found"],
            "states=",
            check["explored_state_count"],
            "path=",
            check["counterexample_path"],
        )


if __name__ == "__main__":
    main()
