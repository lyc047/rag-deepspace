#!/usr/bin/env python
"""Merge independently executed N shards of one registered Stage-6R scan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite merged scan result")
    shards = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    if not shards:
        raise ValueError("at least one shard is required")
    identity_fields = (
        "experiment_id",
        "config_sha256",
        "input_hashes",
        "external_access_status_before_run",
        "external_access_count_before_run",
        "registered_n_channels",
        "claim_boundary",
    )
    first = shards[0]
    for shard in shards[1:]:
        for field in identity_fields:
            if shard[field] != first[field]:
                raise ValueError(f"shard identity mismatch: {field}")
    registered = list(map(int, first["registered_n_channels"]))
    executed = [
        int(value)
        for shard in shards
        for value in shard["executed_n_channels"]
    ]
    if sorted(executed) != sorted(registered) or len(set(executed)) != len(
        executed
    ):
        raise ValueError("shards do not partition the registered N values")
    conditions: dict[str, dict] = {}
    for shard in shards:
        for condition_name, values in shard["conditions"].items():
            destination = conditions.setdefault(condition_name, {})
            overlap = set(destination) & set(values)
            if overlap:
                raise ValueError(f"duplicate N shard: {sorted(overlap)}")
            destination.update(values)
    primary = conditions["primary_fault"]
    common_target_n = []
    positive_savings_n = []
    for key, value in primary.items():
        common = [
            row
            for row in value["matched_clean"]
            if row["common_eligible"]
        ]
        if common:
            common_target_n.append(int(key))
        if any(
            row.get("paired_savings_percentage", -math.inf) > 0.0
            for row in common
        ):
            positive_savings_n.append(int(key))
    checks = {
        "all_registered_n_values_executed": (
            sorted(map(int, primary)) == sorted(registered)
        ),
        "all_bit_identities_valid": all(
            shard["checks"]["all_bit_identities_valid"] for shard in shards
        ),
        "all_wrong_codebook_decode_counts_zero": all(
            value["maximum_wrong_codebook_decode_count"] == 0.0
            for condition in conditions.values()
            for value in condition.values()
        ),
        "primary_fault_common_target_n": sorted(common_target_n),
        "primary_fault_positive_savings_n": sorted(positive_savings_n),
        "full_scan_entry_by_positive_savings": bool(positive_savings_n),
        "shard_identity_checks_passed": True,
    }
    result = {
        "version": first["version"],
        "status": "stage6r_matched_clean_scan_complete",
        "verification_status": "ANALYZED",
        "experiment_id": first["experiment_id"],
        "config_sha256": first["config_sha256"],
        "input_hashes": first["input_hashes"],
        "external_access_status_before_run": first[
            "external_access_status_before_run"
        ],
        "external_access_count_before_run": first[
            "external_access_count_before_run"
        ],
        "registered_n_channels": registered,
        "executed_n_channels": registered,
        "execution": {
            "mode": "independent_n_shards",
            "shard_files": [str(path) for path in args.inputs],
            "sum_shard_elapsed_seconds": float(
                sum(float(shard["elapsed_seconds"]) for shard in shards)
            ),
        },
        "conditions": {
            name: {
                key: values[key]
                for key in sorted(values, key=lambda item: int(item))
            }
            for name, values in conditions.items()
        },
        "checks": checks,
        "environment": first["environment"],
        "claim_boundary": first["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
