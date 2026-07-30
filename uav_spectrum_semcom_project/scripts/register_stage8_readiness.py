"""Validate and freeze Stage-8 external/hardware readiness protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def validate_external(config: dict[str, Any]) -> None:
    governance = config["access_governance"]
    if int(governance["minimum_new_independent_units"]) < 20:
        raise ValueError("external Final requires at least 20 independent units")
    if governance["post_access_parameter_retuning"] is not False:
        raise ValueError("post-access retuning must be forbidden")
    if governance["post_access_code_changes"] is not False:
        raise ValueError("post-access code changes must be forbidden")
    if governance["synthetic_units_count_toward_final"] is not False:
        raise ValueError("synthetic units cannot count toward external Final")
    if config["registered_task"]["channel_counts"] != [8, 16, 32, 64]:
        raise ValueError("the frozen four-N task family must be preserved")
    gate = config["primary_gate"]
    if gate["all_channel_counts_must_pass"] is not True:
        raise ValueError("all four channel counts must pass")
    if gate["wrong_context_action_executions_equals"] != 0:
        raise ValueError("wrong-context actions must remain zero")
    if float(gate["relative_total_bit_saving_ci_lower_strictly_greater_than"]) != 0.0:
        raise ValueError("bit-saving lower bound must be strictly positive")

    freeze_spec = config["candidate_freeze"]
    freeze_path = ROOT / freeze_spec["path"]
    if _sha256(freeze_path) != freeze_spec["sha256"]:
        raise ValueError("candidate freeze hash mismatch")
    freeze = _load(freeze_path)
    if freeze["status"] != freeze_spec["required_status"]:
        raise ValueError("candidate is not frozen")
    if freeze["candidate_id"] != config["candidate_id"]:
        raise ValueError("candidate identity mismatch")
    if len(freeze["artifact_hashes"]) != int(freeze_spec["required_artifact_count"]):
        raise ValueError("candidate artifact count mismatch")


def validate_hardware(config: dict[str, Any]) -> None:
    target = config["execution_target"]
    if target["single_thread_required"] is not True:
        raise ValueError("single-thread board evidence is required")
    if target["gpu_required"] is not False:
        raise ValueError("the frozen codec cannot require a GPU")
    workload = config["workloads"]
    if workload["channel_counts"] != [8, 16, 32, 64]:
        raise ValueError("hardware test must cover all four channel counts")
    if int(workload["measured_scenes_per_n"]) < 10000:
        raise ValueError("hardware soak must cover at least 10000 scenes per N")
    if int(workload["repetitions"]) < 5:
        raise ValueError("hardware timing requires at least five repetitions")
    gate = config["engineering_gate"]
    if gate["all_channel_counts_must_pass"] is not True:
        raise ValueError("hardware gate must apply to all channel counts")
    if gate["wrong_context_action_executions_equals"] != 0:
        raise ValueError("hardware safety violations must remain zero")


def register(
    external_path: Path,
    hardware_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    external = _load(external_path)
    hardware = _load(hardware_path)
    validate_external(external)
    validate_hardware(hardware)
    result = {
        "version": "1.0",
        "status": "REGISTERED_NOT_EXECUTABLE",
        "registration_id": "stage8_readiness_registration_v1",
        "candidate_id": external["candidate_id"],
        "source_git_commit_at_registration": _git_commit(),
        "protocol_hashes": {
            str(external_path.relative_to(ROOT)).replace("\\", "/"): _sha256(external_path),
            str(hardware_path.relative_to(ROOT)).replace("\\", "/"): _sha256(hardware_path),
        },
        "candidate_freeze": external["candidate_freeze"],
        "external_final": {
            "minimum_new_independent_units": external["access_governance"][
                "minimum_new_independent_units"
            ],
            "signal_access_allowed_now": False,
            "blockers": external["execution_blockers"],
        },
        "hardware": {
            "target_selected": all(
                target is not None
                for target in hardware["execution_target"].values()
                if not isinstance(target, bool)
            ),
            "blockers": hardware["execution_blockers"],
        },
        "decision": {
            "candidate_integrity_passed": True,
            "external_protocol_registered": True,
            "hardware_protocol_registered": True,
            "execution_authorized": False,
        },
        "claim_boundary": [
            "This registration does not access any new signal values.",
            "This registration is not an external Final result.",
            "This registration is not a target-board benchmark.",
            "Execution remains blocked until registries and environments are frozen.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external",
        type=Path,
        default=ROOT / "configs/stage8_external_final_readiness_v1.json",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "configs/stage8_hardware_readiness_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_readiness_registration_v1/result.json",
    )
    args = parser.parse_args()
    result = register(args.external, args.hardware, args.output)
    print(json.dumps(result["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
