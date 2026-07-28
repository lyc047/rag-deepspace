#!/usr/bin/env python
"""Decompose frozen Stage-6 task-codebook regret by demand ratio."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_query_demand_stress_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/query_demand_stress_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "query_demand_stress_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite query demand stress")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    rules = protocol["analysis_rules"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not rules["fit_one_joint_codebook_for_all_three_demands"]
        or not rules["do_not_refit_a_separate_codebook_per_demand"]
        or rules["results_may_modify_frozen_architecture_or_parameters"]
    ):
        raise ValueError("invalid query demand stress governance")
    freeze = protocol["architecture_freeze"]
    if sha256_file(PROJECT_DIR / freeze["result"]) != freeze["sha256"]:
        raise ValueError("architecture freeze changed")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("query demand cache changed")
    access_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    all_results = {}
    for n_value in grid["n_channels"]:
        n_channels = int(n_value)
        demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
        queries = tuple(SpectrumTaskQuery(demand) for demand in demands)
        states = [
            build_task_state(
                values,
                queries,
                epsilon_db=float(grid["epsilon_db"]),
            )
            for values in cache[f"channel_power_n{n_channels}"]
        ]
        codebook = fit_greedy_task_codebook(
            [states[int(index)] for index in train_indices]
        )
        decisions = [
            encode_task_state(codebook, states[int(index)])
            for index in evaluation_indices
        ]
        by_query = {}
        for query_position, (ratio, demand) in enumerate(
            zip(ratios, demands)
        ):
            regrets = np.asarray(
                [
                    decision.regret_by_query_db[query_position]
                    for decision in decisions
                ],
                dtype=np.float64,
            )
            candidate_count = n_channels - demand + 1
            by_query[format(ratio, ".2f")] = {
                "demand_ratio": ratio,
                "demand_channels": demand,
                "candidate_block_count": candidate_count,
                "standalone_exact_index_bits": int(
                    math.ceil(math.log2(candidate_count))
                ),
                "mean_regret_db": float(np.mean(regrets)),
                "cvar_0_9_regret_db": empirical_cvar_numpy(regrets, 0.9),
                "p95_regret_db": float(np.quantile(regrets, 0.95)),
                "maximum_regret_db": float(np.max(regrets)),
                "epsilon_safe_rate": float(
                    np.mean(regrets <= float(grid["epsilon_db"]) + 1e-12)
                ),
                "exact_optimum_rate": float(
                    np.mean(regrets <= 1e-12)
                ),
            }
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "joint_codebook_size": len(codebook.codewords),
            "joint_symbol_width_bits": codebook.symbol_width_bits,
            "joint_exact_index_bundle_bits": int(
                sum(
                    row["standalone_exact_index_bits"]
                    for row in by_query.values()
                )
            ),
            "joint_fallback_count": int(
                sum(decision.uses_fallback for decision in decisions)
            ),
            "joint_fallback_rate": float(
                np.mean([decision.uses_fallback for decision in decisions])
            ),
            "query_results": by_query,
        }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_query_regrets_within_frozen_epsilon": all(
            query["epsilon_safe_rate"] == 1.0
            and query["maximum_regret_db"]
            <= float(grid["epsilon_db"]) + 1e-12
            for result in all_results.values()
            for query in result["query_results"].values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"query demand stress failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_query_demand_stress_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_scene_count": int(len(evaluation_indices)),
        },
        "results": all_results,
        "checks": checks,
        "governance": {
            "development_only": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "results": all_results,
                "checks": checks,
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
