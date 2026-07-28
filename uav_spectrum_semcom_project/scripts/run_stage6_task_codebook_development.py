#!/usr/bin/env python
"""Evaluate the Stage-6 analytic task codebook on excluded pilot data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_query_bundle import query_bundle_payload_width  # noqa: E402
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)


def grouped_train_evaluation_indices(
    cluster_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    groups = sorted(set(np.asarray(cluster_ids).astype(str)))
    if len(groups) < 2 or not 0.0 < train_fraction < 1.0:
        raise ValueError("grouped split requires at least two groups and valid ratio")
    random.Random(seed).shuffle(groups)
    train_count = max(1, min(len(groups) - 1, int(len(groups) * train_fraction)))
    train_groups = tuple(sorted(groups[:train_count]))
    evaluation_groups = tuple(sorted(groups[train_count:]))
    values = np.asarray(cluster_ids).astype(str)
    train = np.flatnonzero(np.isin(values, train_groups))
    evaluation = np.flatnonzero(np.isin(values, evaluation_groups))
    if train.size + evaluation.size != values.size:
        raise AssertionError("grouped split lost scenes")
    return train, evaluation, train_groups, evaluation_groups


def evaluate_channel_scale(
    powers: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    n_channels: int,
    demand_ratios: tuple[float, ...],
    epsilon_db: float,
    header_bits: int,
) -> dict:
    demands = tuple(int(round(n_channels * ratio)) for ratio in demand_ratios)
    if any(
        abs(demand / n_channels - ratio) > 1e-12
        for demand, ratio in zip(demands, demand_ratios)
    ):
        raise ValueError("demand ratio does not map exactly to channel count")
    queries = tuple(SpectrumTaskQuery(demand) for demand in demands)
    states = [
        build_task_state(values, queries, epsilon_db=epsilon_db)
        for values in powers
    ]
    training = [states[int(index)] for index in train_indices]
    evaluation = [states[int(index)] for index in evaluation_indices]
    codebook = fit_greedy_task_codebook(training)
    decisions = [encode_task_state(codebook, state) for state in evaluation]
    regrets = np.asarray(
        [decision.max_regret_db for decision in decisions],
        dtype=np.float64,
    )
    payloads = np.asarray(
        [decision.payload_bits for decision in decisions],
        dtype=np.float64,
    )
    fallbacks = np.asarray(
        [decision.uses_fallback for decision in decisions],
        dtype=np.bool_,
    )
    exact_payload = query_bundle_payload_width(n_channels, demands)
    exact_application = int(header_bits) + int(exact_payload)
    semantic_application = float(int(header_bits) + np.mean(payloads))
    return {
        "n_channels": n_channels,
        "demands": list(demands),
        "training_scene_count": len(training),
        "evaluation_scene_count": len(evaluation),
        "training_exact_action_tuple_count": codebook.exact_action_tuple_count,
        "codebook_size": len(codebook.codewords),
        "codebook_symbol_width_bits": codebook.symbol_width_bits,
        "training_coverage_rate": (
            codebook.training_covered_count / codebook.training_scene_count
        ),
        "evaluation_codeword_coverage_rate": float(np.mean(~fallbacks)),
        "evaluation_fallback_rate": float(np.mean(fallbacks)),
        "evaluation_mean_max_query_regret_db": float(np.mean(regrets)),
        "evaluation_cvar_0_9_max_query_regret_db": empirical_cvar_numpy(
            regrets, 0.9
        ),
        "evaluation_maximum_max_query_regret_db": float(np.max(regrets)),
        "exact_query_bundle_payload_bits": int(exact_payload),
        "exact_query_bundle_application_bits": exact_application,
        "semantic_mean_payload_bits": float(np.mean(payloads)),
        "semantic_mean_application_bits": semantic_application,
        "application_bit_reduction_pct_vs_exact_query_bundle": float(
            100.0
            * (exact_application - semantic_application)
            / exact_application
        ),
        "payload_bit_reduction_pct_vs_exact_query_bundle": float(
            100.0 * (exact_payload - np.mean(payloads)) / exact_payload
        ),
        "codewords": [
            {
                "codeword_id": codeword.codeword_id,
                "decoder_actions": list(codeword.decoder_actions),
                "training_coverage_count": codeword.training_coverage_count,
            }
            for codeword in codebook.codewords
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_task_codebook_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage6/task_codebook_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "task_codebook_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 development result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("Stage-6 development governance is invalid")
    cache_path = PROJECT_DIR / protocol["development_source"]["cache"]
    if sha256_file(cache_path) != protocol["development_source"]["cache_sha256"]:
        raise ValueError("Stage-6 development cache hash mismatch")
    access_state_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
    access_before = json.loads(access_state_path.read_text(encoding="utf-8"))
    if (
        access_before["access_count"] != 0
        or access_before["final_signal_values_accessed"]
    ):
        raise ValueError("external Final access state is not pristine")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train, evaluation, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    results = {}
    for n_channels in grid["n_channels"]:
        n = int(n_channels)
        results[str(n)] = evaluate_channel_scale(
            cache[f"channel_power_n{n}"],
            train,
            evaluation,
            n_channels=n,
            demand_ratios=ratios,
            epsilon_db=float(grid["epsilon_db"]),
            header_bits=int(grid["application_header_bits"]),
        )
    access_after = json.loads(access_state_path.read_text(encoding="utf-8"))
    checks = {
        "training_coverage_equals_one": all(
            abs(value["training_coverage_rate"] - 1.0) <= 1e-12
            for value in results.values()
        ),
        "all_evaluation_actions_respect_epsilon": all(
            value["evaluation_maximum_max_query_regret_db"]
            <= float(grid["epsilon_db"]) + 1e-12
            for value in results.values()
        ),
        "external_final_access_state_unchanged": access_before == access_after,
        "external_final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Stage-6 development check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_task_codebook_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_sha256": protocol["development_source"]["cache_sha256"],
            "scene_count": int(len(cache["scene_ids"])),
            "cluster_count": int(len(set(cache["cluster_ids"].astype(str)))),
            "train_scene_count": int(train.size),
            "evaluation_scene_count": int(evaluation.size),
            "train_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
        },
        "task_grid": grid,
        "n_results": results,
        "checks": checks,
        "governance": {
            "development_only": True,
            "pilot_permanently_excluded_from_final": True,
            "external_final_archives_opened": False,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
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
                "checks": checks,
                "results": {
                    n: {
                        "codebook_size": value["codebook_size"],
                        "coverage": value[
                            "evaluation_codeword_coverage_rate"
                        ],
                        "mean_application_bits": value[
                            "semantic_mean_application_bits"
                        ],
                        "application_bit_reduction_pct": value[
                            "application_bit_reduction_pct_vs_exact_query_bundle"
                        ],
                        "maximum_regret_db": value[
                            "evaluation_maximum_max_query_regret_db"
                        ],
                    }
                    for n, value in results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
