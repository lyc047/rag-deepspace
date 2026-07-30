#!/usr/bin/env python
"""Reprice S7.4B event-update identity widths without loading raw data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file


CONFIG = (
    PROJECT_DIR
    / "configs/stage7_s7_4c0_semi_stateless_header_audit_v1.json"
)
OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_4c0_semi_stateless_header_audit_v1/result.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite registered audit")
    started = time.perf_counter()
    config = _read(args.config)
    source_path = PROJECT_DIR / config["inputs"]["s7_4b_result"]
    reproduction_path = (
        PROJECT_DIR / config["inputs"]["s7_4b_reproduction"]
    )
    source = _read(source_path)
    reproduction = _read(reproduction_path)
    source_compare = dict(source)
    reproduction_compare = dict(reproduction)
    source_compare.pop("elapsed_seconds")
    reproduction_compare.pop("elapsed_seconds")
    if (
        source_compare != reproduction_compare
        or source["decision"]["system_gate_passed"]
    ):
        raise ValueError("S7.4B verification or branch trigger mismatch")

    measured_width = int(
        config["audit"]["measured_identity_increment_bits"]
    )
    reference = str(config["audit"]["reference_policy"])
    minimum_reduction = float(
        config["gate"]["minimum_total_bit_reduction_percent"]
    )
    maximum_clean_loss = float(
        config["gate"]["maximum_clean_loss_percentage_points"]
    )
    rows = {}
    passing_counts: dict[str, int] = {}
    for n, n_result in source["n_results"].items():
        rows[n] = {}
        reference_bits = n_result["policy_metrics"][reference][
            "total_bits_per_scene"
        ]
        for policy in config["audit"]["source_policies"]:
            rows[n][policy] = {}
            policy_metrics = n_result["policy_metrics"][policy]
            total = policy_metrics["total_bits_per_scene"]
            identity = policy_metrics[
                "event_context_incremental_bits_per_scene"
            ]
            clean = n_result["paired_comparisons_vs_fixed10"][policy][
                "paired_clean_gain_percentage_points"
            ]
            for width in config["audit"][
                "counterfactual_identity_increment_bits"
            ]:
                width = int(width)
                saved_fraction = 1.0 - width / measured_width
                adjusted_mean = float(total["mean"]) - saved_fraction * float(
                    identity["mean"]
                )
                point_reduction = 100.0 * (
                    float(reference_bits["mean"]) - adjusted_mean
                ) / float(reference_bits["mean"])
                adjusted_upper = float(total["ci_upper"]) - saved_fraction * (
                    float(identity["ci_lower"])
                )
                conservative_reduction_lower = 100.0 * (
                    float(reference_bits["ci_lower"]) - adjusted_upper
                ) / float(reference_bits["ci_lower"])
                key = str(width)
                point_pass = (
                    point_reduction >= minimum_reduction
                    and float(clean["mean"]) >= -maximum_clean_loss
                )
                conservative_pass = (
                    conservative_reduction_lower >= minimum_reduction
                    and float(clean["ci_lower"]) >= -maximum_clean_loss
                    and width > 0
                )
                rows[n][policy][key] = {
                    "adjusted_total_bits_per_scene": adjusted_mean,
                    "point_total_bit_reduction_percent": point_reduction,
                    "conservative_total_bit_reduction_lower_percent": (
                        conservative_reduction_lower
                    ),
                    "paired_clean_gain_percentage_points": clean,
                    "point_gate_passed": point_pass,
                    "conservative_gate_passed": conservative_pass,
                }
                policy_key = f"{policy}_header{width}"
                passing_counts[policy_key] = (
                    passing_counts.get(policy_key, 0)
                    + int(conservative_pass)
                )
    required = int(config["gate"]["minimum_passing_n_count"])
    eligible = {
        key: value
        for key, value in passing_counts.items()
        if not key.endswith("header0")
    }
    best = max(eligible, key=lambda key: (eligible[key], key))
    passed = eligible[best] >= required
    result = {
        "version": "1.0",
        "status": "stage7_s7_4c0_semi_stateless_header_audit_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "s7_4b_result": sha256_file(source_path),
            "s7_4b_reproduction": sha256_file(reproduction_path),
        },
        "raw_signal_files_loaded": 0,
        "n_results": rows,
        "decision": {
            "passing_n_count": passing_counts,
            "best_nonzero_candidate": best,
            "required_n_count": required,
            "gate_passed": passed,
            "next_action": (
                config["gate"]["if_passes"]
                if passed
                else config["gate"]["if_fails"]
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment_snapshot([]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
