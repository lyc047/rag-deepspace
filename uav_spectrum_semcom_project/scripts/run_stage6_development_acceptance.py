#!/usr/bin/env python
"""Audit Stage-6 development evidence before external-Final registration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)


def _all_true(mapping: dict) -> bool:
    return all(bool(value) for value in mapping.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_development_acceptance_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/development_acceptance_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "development_acceptance_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite development acceptance")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    decision_rule = protocol["decision_rule"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["frozen_algorithm_or_parameters_may_be_modified"]
        or decision_rule["passing_authorizes_external_final_execution"]
        or not decision_rule[
            "passing_authorizes_stage6_final_protocol_registration_only"
        ]
    ):
        raise ValueError("invalid development acceptance governance")
    evidence = {}
    evidence_hash_checks = {}
    for name, specification in protocol["locked_evidence"].items():
        path = PROJECT_DIR / specification["path"]
        evidence_hash_checks[name] = (
            sha256_file(path) == specification["sha256"]
        )
        evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    if not all(evidence_hash_checks.values()):
        raise ValueError("locked acceptance evidence changed")
    gates = protocol["acceptance_gates"]
    complexity_primary = evidence["complexity_primary"]
    complexity_reproduction = evidence["complexity_reproduction"]
    timing_cell_count = 0
    timing_within = 0
    timing_outside = []
    memory_exact = True
    for n_channels, primary_row in complexity_primary["results"].items():
        reproduced_row = complexity_reproduction["results"][n_channels]
        memory_exact = memory_exact and (
            primary_row["end_to_end_peak_traced_bytes"]
            == reproduced_row["end_to_end_peak_traced_bytes"]
        )
        for metric, primary_metric in primary_row["timings"].items():
            timing_cell_count += 1
            primary_median = float(primary_metric["median"])
            reproduced_median = float(
                reproduced_row["timings"][metric]["median"]
            )
            relative = abs(primary_median - reproduced_median) / max(
                abs(primary_median),
                abs(reproduced_median),
                1e-12,
            )
            if relative < 0.25:
                timing_within += 1
            else:
                timing_outside.append(
                    {
                        "n_channels": int(n_channels),
                        "metric": metric,
                        "primary_median": primary_median,
                        "reproduction_median": reproduced_median,
                        "relative_difference": relative,
                    }
                )
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    collected_count = sum(
        int(match.group(1))
        for line in collect.stdout.splitlines()
        if (match := re.search(r":\s+(\d+)$", line))
    )
    regression = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    access_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
    access_state = json.loads(access_path.read_text(encoding="utf-8"))
    component = evidence["component_ablation"]
    fault = evidence["fault_stress_primary"]
    query = evidence["query_stress_primary"]
    deterministic_pairs_equal = (
        protocol["locked_evidence"]["component_ablation"]["sha256"]
        == protocol["locked_evidence"][
            "component_ablation_reproduction"
        ]["sha256"]
        and protocol["locked_evidence"]["fault_stress_primary"]["sha256"]
        == protocol["locked_evidence"][
            "fault_stress_reproduction"
        ]["sha256"]
        and protocol["locked_evidence"]["query_stress_primary"]["sha256"]
        == protocol["locked_evidence"][
            "query_stress_reproduction"
        ]["sha256"]
    )
    gate_checks = {
        "architecture_freeze_checks_all_true": _all_true(
            evidence["architecture_freeze"]["checks"]
        ),
        "component_ablation_checks_all_true": _all_true(
            component["checks"]
        ),
        "learned_auxiliary_cross_n_support_minimum": (
            component["component_support"]["learned_update_protection"][
                "support_count"
            ]
            >= int(gates["learned_auxiliary_cross_n_support_minimum"])
        ),
        "fault_stress_checks_all_true": _all_true(fault["checks"]),
        "query_stress_checks_all_true": _all_true(query["checks"]),
        "all_query_epsilon_safe": query["checks"][
            "all_query_regrets_within_frozen_epsilon"
        ],
        "wrong_codebook_decode_count_zero": fault["checks"][
            "wrong_codebook_decode_count_equals_zero"
        ],
        "deterministic_primary_reproduction_hashes_equal": (
            deterministic_pairs_equal
        ),
        "complexity_timing_cells_within_25_percent_minimum": (
            timing_within
            >= int(
                gates[
                    "complexity_timing_cells_within_25_percent_minimum"
                ]
            )
        ),
        "complexity_timing_cell_count": (
            timing_cell_count == int(gates["complexity_timing_cell_count"])
        ),
        "complexity_memory_exact_across_reproduction": (
            memory_exact
            == bool(
                gates["complexity_memory_exact_across_reproduction"]
            )
        ),
        "minimum_full_regression_test_count": (
            collected_count
            >= int(gates["minimum_full_regression_test_count"])
        ),
        "full_regression_must_pass": (
            regression.returncode == 0
            if gates["full_regression_must_pass"]
            else True
        ),
        "external_final_access_count_must_equal": (
            access_state["access_count"]
            == int(gates["external_final_access_count_must_equal"])
            and not access_state["final_signal_values_accessed"]
            and not access_state["final_method_outputs_accessed"]
        ),
    }
    passed = all(gate_checks.values())
    verdict = (
        "ready_for_stage6_final_protocol_registration_with_caution"
        if passed
        else "not_ready_for_stage6_final_protocol_registration"
    )
    result = {
        "version": "1.0",
        "status": "stage6_development_acceptance_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "verdict": verdict,
        "all_acceptance_gates_passed": passed,
        "evidence_hash_checks": evidence_hash_checks,
        "gate_checks": gate_checks,
        "complexity_reproducibility": {
            "classification": "partially_reproducible",
            "timing_cell_count": timing_cell_count,
            "timing_cells_within_25_percent": timing_within,
            "timing_cells_outside_25_percent": timing_outside,
            "memory_exact": memory_exact,
        },
        "regression_validation": {
            "collected_test_count": collected_count,
            "pytest_exit_code": regression.returncode,
            "passed": regression.returncode == 0,
        },
        "caution_flags": protocol["caution_flags"],
        "supported_development_claims": [
            "The frozen joint task codebook satisfies the 0.2 dB constraint for all three registered demand ratios on the excluded development activity.",
            "The learned duplicate-update protection is a small auxiliary reliability contribution with cross-N interval support in three of four N values.",
            "Fixed heartbeat and context-belief recovery are the dominant reliability components under the registered synthetic mixed-fault model.",
            "The frozen desktop Python implementation remains computationally modest over N = 8 to 64 in the measured environment.",
            "Receiver context reset is the dominant registered synthetic-fault vulnerability."
        ],
        "prohibited_claims_before_external_final": [
            "cross-activity or real-UAV mobility generalization",
            "same-clean-rate end-to-end bit saving of at least 20 percent",
            "measured field failure probabilities or measured UAV link reliability",
            "machine learning as the primary source of Stage-6 gain",
            "embedded latency, memory, or energy guarantees",
            "external Final success"
        ],
        "authorization": {
            "stage6_final_protocol_registration": passed,
            "external_final_execution": False,
            "algorithm_or_parameter_change": False,
        },
        "final_access_count": access_state["access_count"],
        "environment": environment_snapshot(["numpy", "scikit-learn"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": verdict,
                "gate_checks": gate_checks,
                "complexity_reproducibility": result[
                    "complexity_reproducibility"
                ],
                "regression_validation": result[
                    "regression_validation"
                ],
                "authorization": result["authorization"],
                "final_access_count": access_state["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
