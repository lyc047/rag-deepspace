#!/usr/bin/env python
"""Audit extracted AERPAW sweeps and optionally freeze a pilot exclusion list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    build_permanently_excluded_pilot_manifest,
    discover_aerpaw_pairs,
    load_aerpaw_power_sweep,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted ResultsLW1Feb2022_SigMF directory")
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--pilot-count", type=int, default=0)
    parser.add_argument("--validate-count", type=int, default=10)
    parser.add_argument("--dataset-id", default="aerpaw-lw1-feb2022-dryad-hmgqnk9zn")
    args = parser.parse_args()

    pairs, discovery_errors = discover_aerpaw_pairs(args.root)
    validation_errors: list[str] = []
    validated = []
    for pair in pairs[: max(args.validate_count, 0)]:
        try:
            sweep = load_aerpaw_power_sweep(pair.meta_path, pair.data_path)
            validated.append(
                {
                    "stem": pair.stem,
                    "site": sweep.site,
                    "capture_datetime": sweep.capture_datetime,
                    "n_bins": sweep.n_bins,
                    "frequency_span_mhz": [float(sweep.frequencies_mhz[0]), float(sweep.frequencies_mhz[-1])],
                    "power_range_dbm": [float(sweep.powers_dbm.min()), float(sweep.powers_dbm.max())],
                }
            )
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            validation_errors.append(f"{pair.stem}: {exc}")

    result = {
        "audit_version": "1.0",
        "dataset_id": args.dataset_id,
        "root": str(args.root.resolve()),
        "discovered_pair_count": len(pairs),
        "discovery_errors": discovery_errors,
        "validated_sample_count": len(validated),
        "validation_errors": validation_errors,
        "validated_examples": validated,
        "format_facts": {
            "sigmf_payload_semantics": "real_float32_power_dbm_per_frequency_bin",
            "raw_complex_iq": False,
            "independent_occupancy_labels_in_deposit": False,
            "lw1_receiver_count": 1,
        },
        "stage4_compatibility": {
            "pilot_ingestion": "supported",
            "C1_external_resource_proxy": "conditional_on_pilot_frozen_power_to_occupancy_rule",
            "selective_G2_four_node_final": "unsupported_by_LW1_only",
            "automatic_final_registration": False,
        },
        "final_access_consumed": False,
    }
    atomic_write_json(args.audit_output, result)

    if args.pilot_output is not None or args.pilot_count:
        if args.pilot_output is None or args.pilot_count < 1:
            raise ValueError("--pilot-output and a positive --pilot-count must be provided together")
        pilot = build_permanently_excluded_pilot_manifest(
            pairs, dataset_root=args.root, count=args.pilot_count, dataset_id=args.dataset_id
        )
        atomic_write_json(args.pilot_output, pilot)
    print(json.dumps({"audit_output": str(args.audit_output), "pair_count": len(pairs), "errors": len(discovery_errors) + len(validation_errors), "final_access_consumed": False}, indent=2))


if __name__ == "__main__":
    main()
