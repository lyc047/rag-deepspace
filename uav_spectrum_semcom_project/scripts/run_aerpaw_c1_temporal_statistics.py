#!/usr/bin/env python
"""Evaluate the preregistered C1 temporal confirmation gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy, paired_cluster_bootstrap  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/aerpaw_c1_temporal_holdout_protocol_v1.json")
    parser.add_argument("--metrics", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_confirmation_v1/scene_metrics.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_confirmation_v1/statistics.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError("refusing to overwrite temporal confirmation statistics")
    expected = int(protocol["confirmation_statistics"]["scene_count"])
    methods = ("baseline_fixed4", "c1_v3_tail_ranking_fixed4")
    indexed = {(row["scene_id"], row["method"]): row for row in payload["rows"]}
    scene_ids = sorted({row["scene_id"] for row in payload["rows"]})
    if len(scene_ids) != expected or len(indexed) != expected * 2 or any((scene, method) not in indexed for scene in scene_ids for method in methods):
        raise ValueError("metrics are not a complete paired confirmation payload")
    baseline = [indexed[(scene, methods[0])] for scene in scene_ids]
    proposed = [indexed[(scene, methods[1])] for scene in scene_ids]
    if any(left["cluster_id"] != right["cluster_id"] or left["stratum"] != right["stratum"] for left, right in zip(baseline, proposed)):
        raise ValueError("paired scene metadata differ by method")
    stats_protocol = protocol["confirmation_statistics"]
    inference = paired_cluster_bootstrap(
        np.asarray([row["regret_db"] for row in proposed]),
        np.asarray([row["regret_db"] for row in baseline]),
        np.asarray([row["actual_bits"] for row in proposed]),
        np.asarray([row["actual_bits"] for row in baseline]),
        [row["cluster_id"] for row in baseline],
        repetitions=int(stats_protocol["paired_cluster_bootstrap_repetitions"]),
        seed=int(stats_protocol["bootstrap_seed"]),
        cvar_alpha=float(stats_protocol["cvar_alpha"]),
    )
    bit_margin = 0.01 * inference["baseline_mean_actual_bits"]
    gates = {
        "mean_regret_superiority": inference["mean_regret_upper_bound"] < 0.0,
        "cvar_superiority": inference["cvar_upper_bound"] < 0.0,
        "fixed_application_payload_equality": all(left["nominal_application_bits"] == right["nominal_application_bits"] for left, right in zip(baseline, proposed)),
        "actual_bit_noninferiority_within_1pct": inference["actual_bit_upper_bound"] <= bit_margin,
    }
    descriptive = {}
    for method, rows in zip(methods, (baseline, proposed)):
        regret = np.asarray([row["regret_db"] for row in rows])
        descriptive[method] = {"mean_regret_db": float(np.mean(regret)), "cvar_0_9_regret_db": empirical_cvar_numpy(regret, 0.9), "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows]))}
    strata = {}
    for stratum in sorted({row["stratum"] for row in baseline}):
        mask = np.asarray([row["stratum"] == stratum for row in baseline])
        left = np.asarray([row["regret_db"] for row in baseline])[mask]
        right = np.asarray([row["regret_db"] for row in proposed])[mask]
        strata[stratum] = {"scenes": int(mask.sum()), "baseline_mean_regret_db": float(left.mean()), "proposed_mean_regret_db": float(right.mean()), "mean_difference_db": float(np.mean(right-left))}
    result = {
        "version": "1.0",
        "status": "temporal_confirmation_statistics_complete",
        "metrics_sha256": sha256_file(args.metrics),
        "protocol_sha256": sha256_file(args.protocol),
        "descriptive": descriptive,
        "paired_cluster_bootstrap": inference,
        "bit_noninferiority_margin": bit_margin,
        "gates": gates,
        "overall_success": all(gates.values()),
        "stratum_sensitivity": strata,
        "multiplicity_note": "Mean and CVaR are a conjunction claim evaluated as an intersection-union test; both one-sided gates must pass.",
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "overall_success": result["overall_success"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
