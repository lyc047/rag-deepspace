#!/usr/bin/env python
"""Measure bounded offline codebook-fit scaling after S7.5A."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage7_s7_5a_complexity_benchmark import _spectra
from spectrum_semcom.final_holdout import atomic_write_json
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage7_s7_5b_offline_fit_scaling_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_5b_offline_fit_scaling_v1/result.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite scaling result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    task = config["task"]
    rows = {}
    started = time.perf_counter()
    for n_position, n_channels in enumerate(task["n_channels"]):
        n_channels = int(n_channels)
        queries = tuple(
            SpectrumTaskQuery(
                max(1, min(n_channels, int(round(n_channels * ratio))))
            )
            for ratio in task["demand_ratios"]
        )
        rng = np.random.default_rng(
            int(task["random_seed"]) + n_position * 1009
        )
        spectra = _spectra(
            rng, max(map(int, task["scene_counts"])), n_channels
        )
        all_states = [
            build_task_state(
                row, queries, epsilon_db=float(task["epsilon_db"])
            )
            for row in spectra
        ]
        rows[str(n_channels)] = {}
        for scene_count in map(int, task["scene_counts"]):
            rows[str(n_channels)][str(scene_count)] = {}
            for limit in map(int, task["max_codewords"]):
                fit_started = time.perf_counter()
                codebook = fit_greedy_task_codebook(
                    all_states[:scene_count], max_codewords=limit
                )
                elapsed = time.perf_counter() - fit_started
                rows[str(n_channels)][str(scene_count)][str(limit)] = {
                    "fit_seconds": elapsed,
                    "selected_codewords": len(codebook.codewords),
                    "exact_action_tuple_count": (
                        codebook.exact_action_tuple_count
                    ),
                    "training_coverage_fraction": (
                        codebook.training_covered_count / scene_count
                    ),
                }
                print(
                    f"S7.5B N={n_channels} scenes={scene_count} "
                    f"K={limit} fit={elapsed:.4f}s",
                    flush=True,
                )
    gates = config["gates"]
    n_checks = {}
    for n in map(str, task["n_channels"]):
        at_64 = rows[n]["64"]["3"]["fit_seconds"]
        at_1024 = rows[n]["1024"]["3"]["fit_seconds"]
        n_checks[n] = {
            "fit_64_passed": (
                at_64
                <= float(
                    gates["maximum_fit_seconds_at_64_calibration_scenes"]
                )
            ),
            "fit_1024_passed": (
                at_1024
                <= float(
                    gates[
                        "maximum_bounded_fit_seconds_at_1024_scenes"
                    ]
                )
            ),
        }
        n_checks[n]["n_gate_passed"] = all(n_checks[n].values())
    passing = sum(
        int(value["n_gate_passed"]) for value in n_checks.values()
    )
    required = int(gates["minimum_passing_n_count"])
    result = {
        "version": "1.0",
        "status": "stage7_s7_5b_offline_fit_scaling_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "n_results": rows,
        "n_checks": n_checks,
        "decision": {
            "passing_n_count": passing,
            "required_n_count": required,
            "gate_passed": passing >= required,
            "next_action": (
                "freeze_complexity_evidence_and_write_report"
                if passing >= required
                else "optimize_or_formally_bound_offline_fit_before_engineering_claim"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
