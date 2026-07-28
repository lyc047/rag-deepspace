#!/usr/bin/env python
"""Run the frozen non-confirmatory Stage-6 external repair validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

import run_stage5_external_final as stage5_final  # noqa: E402
from run_stage6_external_final import (  # noqa: E402
    evaluate_n,
    load_pilot,
    prepare_n,
    verify_frozen_evidence,
)
from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.aerpaw_helikite_repair import (  # noqa: E402
    load_helikite_zip_power_sweep_compatible,
)
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    validate_external_final_catalog,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6_external_repair_validation_v1.json"
)


def verify_repair_snapshot(snapshot: dict) -> list[str]:
    errors = []
    files = snapshot.get("files", [])
    if (
        snapshot.get("status")
        != "stage6_external_repair_validation_code_frozen"
        or canonical_json_sha256(files)
        != snapshot.get("executable_snapshot_sha256")
    ):
        errors.append("repair snapshot identity or canonical hash is invalid")
    for row in files:
        path = PROJECT_DIR / row["path"]
        if not path.is_file():
            errors.append(f"missing repair-frozen file: {row['path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"repair-frozen file size changed: {row['path']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"repair-frozen file SHA-256 changed: {row['path']}")
    audit = snapshot.get("deviation_audit", {})
    if (
        audit.get("changed_parent_frozen_file_count") != 0
        or audit.get("added_repair_adapter_file_count") != 1
        or audit.get("algorithm_or_threshold_change_detected") is not False
    ):
        errors.append("repair snapshot deviation audit is invalid")
    return errors


def verify_inputs(
    config: dict,
    protocol: dict,
    registry: dict,
    state: dict,
    snapshot: dict,
    catalog: dict,
) -> list[str]:
    errors = verify_repair_snapshot(snapshot)
    errors.extend(verify_frozen_evidence(protocol))
    errors.extend(validate_external_final_catalog(catalog, protocol, registry))
    for entry in config["parent_final"].values():
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = PROJECT_DIR / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            errors.append(f"bound parent artifact changed: {entry['path']}")
    access = config["parent_final"]["access_state"]
    if (
        state.get("status") != access["required_status"]
        or state.get("access_count") != access["required_access_count"]
        or state.get("final_signal_values_accessed") is not True
        or state.get("retry_on_this_final_permitted") is not False
        or state.get("reset_permitted") is not False
    ):
        errors.append("parent Final failure/access state is inconsistent")
    if len(catalog.get("scenes", [])) != int(
        config["parent_final"]["catalog"]["required_scene_count"]
    ):
        errors.append("repair validation requires the complete 200-scene catalog")
    for entry in registry["final_candidates"]:
        archive = Path(registry["local_root"]) / entry["archive"]
        if (
            not archive.is_file()
            or archive.stat().st_size != int(entry["archive_size_bytes"])
            or sha256_file(archive) != entry["archive_sha256"]
        ):
            errors.append(f"archive integrity failed: {archive}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = config["parent_final"]
    protocol_path = PROJECT_DIR / parent["protocol"]["path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry_path = PROJECT_DIR / protocol["registry"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state = json.loads(
        (PROJECT_DIR / parent["access_state"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    catalog_path = PROJECT_DIR / parent["catalog"]["path"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    snapshot_path = args.snapshot or PROJECT_DIR / config["execution"][
        "freeze_snapshot"
    ]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    output = args.output or PROJECT_DIR / config["execution"]["raw_output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite repair-validation output")
    errors = verify_inputs(
        config, protocol, registry, state, snapshot, catalog
    )
    if errors:
        raise ValueError("repair-validation input audit failed: " + "; ".join(errors))

    parent_loader = stage5_final.load_helikite_zip_power_sweep
    stage5_final.load_helikite_zip_power_sweep = (
        load_helikite_zip_power_sweep_compatible
    )
    try:
        records = stage5_final.load_scene_arrays(catalog, protocol)
    finally:
        stage5_final.load_helikite_zip_power_sweep = parent_loader
    if len(records) != int(
        config["parent_final"]["catalog"]["required_scene_count"]
    ):
        raise ValueError("adapter did not load the complete registered catalog")
    pilot = load_pilot(protocol)
    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    n_results = {}
    wrong_actions = 0.0
    for n_position, n_channels in enumerate(n_values):
        prepared = prepare_n(n_channels, protocol, pilot, records)
        if not prepared["deployable"]:
            n_results[str(n_channels)] = {
                "n_channels": n_channels,
                "status": "not_deployable_in_current_codec",
                "representation": prepared["representation"],
                "outer_groups": {},
            }
            continue
        n_result = evaluate_n(
            n_channels, n_position, protocol, records, prepared
        )
        n_results[str(n_channels)] = n_result
        wrong_actions += sum(
            float(
                group["semantic_workpoint"]["summary"][
                    "wrong_codebook_decode_count"
                ]
            )
            for group in n_result["outer_groups"].values()
        )
    result = {
        "version": "1.0",
        "status": "stage6_external_repair_validation_execution_complete",
        "classification": config["classification"],
        "validation_config_sha256": sha256_file(args.config),
        "parent_protocol_sha256": sha256_file(protocol_path),
        "registry_sha256": sha256_file(registry_path),
        "catalog_sha256": canonical_json_sha256(catalog),
        "parent_access_receipt_sha256": state["receipt_sha256"],
        "repair_code_snapshot_sha256": snapshot[
            "executable_snapshot_sha256"
        ],
        "scene_count": len(records),
        "campaign_ids": sorted(
            {record["campaign_id"] for record in records}
        ),
        "n_results": n_results,
        "checks": {
            "all_registered_n_retained": set(n_results)
            == {str(value) for value in n_values},
            "wrong_codebook_actions_must_be_zero": wrong_actions == 0.0,
            "all_methods_used_identical_scene_catalog": True,
            "complete_registered_200_scene_catalog_used": len(records) == 200,
            "parent_final_access_state_modified": False,
            "adapter_only_repair": True,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "governance": {
            "external_confirmatory_final": False,
            "single_access_final": False,
            "post_result_parameter_tuning_permitted": False,
            "registered_final_rules_are_descriptive_only": True,
            "parent_final_access_count": state["access_count"],
        },
        "claim_boundary": (
            "This is a non-confirmatory external repair validation. It can "
            "describe cross-campaign robustness and application-bit behavior "
            "after a deterministic data-layout repair, but it cannot replace "
            "a new single-access external Final."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "scene_count": len(records),
                "campaign_ids": result["campaign_ids"],
                "n_values": n_values,
                "classification": config["classification"],
                "parent_final_state_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
