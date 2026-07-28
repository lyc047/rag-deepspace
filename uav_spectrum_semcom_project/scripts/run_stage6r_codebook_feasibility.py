#!/usr/bin/env python
"""Run Stage-6R fixed-action codebook feasibility and oracle scans."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
)
from spectrum_semcom.stage6r_regret_codebook import (  # noqa: E402
    admissible_union_candidates,
    campaign_balanced_weights,
    evaluate_selected_actions,
    exhaustive_action_tuples,
    greedy_select_actions,
    optimal_action_candidates,
    regret_matrix,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_codebook_feasibility_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/codebook_feasibility_v1/result.json"
)
PILOT_CAMPAIGN = "aerpaw_packapalooza_2023"


def _load_combined(config: dict, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    pilot_path = PROJECT_DIR / config["inputs"]["pilot_2023_cache"]
    external_path = PROJECT_DIR / config["inputs"]["external_2024_2025_cache"]
    with np.load(pilot_path, allow_pickle=False) as pilot:
        pilot_power = pilot[f"channel_power_n{n_channels}"].astype(
            np.float64
        )
    with np.load(external_path, allow_pickle=False) as external:
        external_power = external[f"channel_power_n{n_channels}"].astype(
            np.float64
        )
        external_campaigns = external["campaign_ids"].astype(str)
    powers = np.concatenate((pilot_power, external_power), axis=0)
    campaigns = np.concatenate(
        (
            np.full(pilot_power.shape[0], PILOT_CAMPAIGN),
            external_campaigns,
        )
    )
    return powers, campaigns


def _queries(config: dict, n_channels: int) -> tuple[SpectrumTaskQuery, ...]:
    return tuple(
        SpectrumTaskQuery(
            int(round(n_channels * float(ratio)))
        )
        for ratio in config["task"]["demand_ratios"]
    )


def _candidate_indices(
    *,
    method: str,
    train_states: list,
    exhaustive_actions: tuple[tuple[int, ...], ...],
    action_to_index: dict[tuple[int, ...], int],
) -> np.ndarray:
    if method in {"optimal_only_unbalanced", "optimal_only_balanced"}:
        values = optimal_action_candidates(train_states)
    elif method == "admissible_union_balanced":
        values = admissible_union_candidates(train_states)
    elif method in {"exhaustive_balanced", "oracle_holdout_exhaustive"}:
        return np.arange(len(exhaustive_actions), dtype=int)
    else:
        raise ValueError(f"unsupported Stage-6R method: {method}")
    return np.asarray([action_to_index[value] for value in values], dtype=int)


def _fit_one(
    *,
    method: str,
    fit_indices: np.ndarray,
    fit_campaigns: np.ndarray,
    states: list,
    all_regrets: np.ndarray,
    exhaustive_actions: tuple[tuple[int, ...], ...],
    action_to_index: dict[tuple[int, ...], int],
    epsilon_db: float,
    maximum_k: int,
) -> tuple[np.ndarray, int]:
    fit_states = [states[int(index)] for index in fit_indices]
    candidates = _candidate_indices(
        method=method,
        train_states=fit_states,
        exhaustive_actions=exhaustive_actions,
        action_to_index=action_to_index,
    )
    weights = (
        np.ones(fit_indices.size, dtype=np.float64)
        if method == "optimal_only_unbalanced"
        else campaign_balanced_weights(fit_campaigns)
    )
    selected = greedy_select_actions(
        regrets=all_regrets[np.ix_(fit_indices, candidates)],
        actions=[exhaustive_actions[int(index)] for index in candidates],
        epsilon_db=epsilon_db,
        weights=weights,
        max_codewords=maximum_k,
    )
    selected_global = np.asarray(
        [action_to_index[value] for value in selected.actions], dtype=int
    )
    return selected_global, int(candidates.size)


def _aggregate(n_result: dict, methods: list[str], k_values: list[int]) -> dict:
    output = {}
    for method in methods:
        output[method] = {}
        for k_value in k_values:
            rows = [
                fold["methods"][method]["k_results"][str(k_value)]
                for fold in n_result["folds"].values()
            ]
            heldout_coverage = np.asarray(
                [row["heldout"]["coverage_rate"] for row in rows]
            )
            heldout_escape = np.asarray(
                [row["heldout"]["escape_rate"] for row in rows]
            )
            output[method][str(k_value)] = {
                "fold_count": len(rows),
                "macro_train_coverage": float(
                    np.mean(
                        [row["fit"]["coverage_rate"] for row in rows]
                    )
                ),
                "macro_heldout_coverage": float(
                    np.mean(heldout_coverage)
                ),
                "worst_heldout_coverage": float(
                    np.min(heldout_coverage)
                ),
                "macro_heldout_escape_rate": float(
                    np.mean(heldout_escape)
                ),
                "worst_heldout_escape_rate": float(
                    np.max(heldout_escape)
                ),
                "macro_payload_savings_percentage": float(
                    np.mean(
                        [
                            row["heldout"][
                                "payload_savings_percentage"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "all_unsafe_coded_action_counts_zero": all(
                    row["heldout"]["unsafe_coded_action_count"] == 0
                    for row in rows
                ),
            }
    return output


def _decision_summary(config: dict, n_results: dict) -> dict:
    k_max = str(max(map(int, config["task"]["k_values"])))
    structural_n = []
    representation_n = []
    candidate_improvements = []
    strong_n = []
    ai_entry_n = []
    analytic_methods = [
        value
        for value in config["methods"]
        if value != "oracle_holdout_exhaustive"
    ]
    for n_key, n_result in n_results.items():
        aggregate = n_result["aggregate_by_method_k"]
        oracle = aggregate["oracle_holdout_exhaustive"][k_max]
        exhaustive = aggregate["exhaustive_balanced"][k_max]
        if oracle["macro_heldout_escape_rate"] > 0.50:
            structural_n.append(int(n_key))
        if (
            oracle["macro_heldout_escape_rate"] <= 0.20
            and exhaustive["macro_heldout_escape_rate"] > 0.50
        ):
            representation_n.append(int(n_key))
        for k_value in map(int, config["task"]["k_values"]):
            key = str(k_value)
            improvement = (
                aggregate["admissible_union_balanced"][key][
                    "macro_heldout_coverage"
                ]
                - aggregate["optimal_only_balanced"][key][
                    "macro_heldout_coverage"
                ]
            )
            if improvement >= 0.10 - 1e-12:
                candidate_improvements.append(
                    {
                        "n_channels": int(n_key),
                        "k": k_value,
                        "absolute_macro_coverage_improvement": float(
                            improvement
                        ),
                    }
                )
        if any(
            aggregate["exhaustive_balanced"][str(k_value)][
                "macro_heldout_escape_rate"
            ]
            <= 0.20
            for k_value in map(int, config["task"]["k_values"])
            if k_value <= 16
        ):
            strong_n.append(int(n_key))
        strongest_analytic = max(
            aggregate[method][k_max]["macro_heldout_coverage"]
            for method in analytic_methods
        )
        oracle_coverage = oracle["macro_heldout_coverage"]
        if (
            oracle_coverage >= 0.80
            and oracle_coverage - strongest_analytic >= 0.10
        ):
            ai_entry_n.append(int(n_key))
    structural = len(structural_n) >= 3
    strong = len(strong_n) >= 3
    if structural:
        next_route = "conditional_or_hierarchical_codebook_before_ai_vq"
    elif ai_entry_n:
        next_route = "regret_aware_candidate_ranking_ai_prototype"
    elif candidate_improvements:
        next_route = "strengthen_analytic_admissible_set_cover_first"
    else:
        next_route = "reassess_task_representation_and_half_state_protocol"
    return {
        "single_layer_fixed_codebook_structurally_inadequate": structural,
        "n_values_with_oracle_escape_above_0_50_at_k64": structural_n,
        "representation_generalization_problem_n_values": representation_n,
        "candidate_generation_problem_detected": bool(
            candidate_improvements
        ),
        "candidate_generation_improvements": candidate_improvements,
        "strong_feasibility_passed": strong,
        "strong_feasibility_n_values": strong_n,
        "ai_entry_gate_n_values": ai_entry_n,
        "recommended_next_route": next_route,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6R feasibility result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    external_cache = PROJECT_DIR / config["inputs"][
        "external_2024_2025_cache"
    ]
    if not external_cache.is_file():
        raise FileNotFoundError(
            "build the Stage-6R airborne development cache first"
        )
    state = json.loads(
        (
            PROJECT_DIR / config["inputs"]["external_access_state"]
        ).read_text(encoding="utf-8")
    )
    if (
        state.get("status")
        != config["inputs"]["required_external_access_status"]
        or state.get("access_count") != 1
        or state.get("reset_permitted") is not False
    ):
        raise ValueError("consumed external-data state changed")

    epsilon = float(config["task"]["epsilon_db"])
    k_values = list(map(int, config["task"]["k_values"]))
    maximum_k = max(k_values)
    methods = list(config["methods"])
    n_results = {}
    for n_channels in map(int, config["task"]["n_channels"]):
        powers, campaigns = _load_combined(config, n_channels)
        expected_campaigns = set(config["splitting"]["campaign_ids"])
        if set(campaigns.tolist()) != expected_campaigns:
            raise ValueError("airborne campaign set differs from protocol")
        queries = _queries(config, n_channels)
        states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in powers
        ]
        actions = exhaustive_action_tuples(states)
        action_to_index = {
            value: index for index, value in enumerate(actions)
        }
        all_regrets = regret_matrix(states, actions)
        exact_bits = int(
            sum(
                math.ceil(
                    math.log2(
                        n_channels - query.demand_channels + 1
                    )
                )
                for query in queries
            )
        )
        folds = {}
        for holdout in config["splitting"]["campaign_ids"]:
            test_indices = np.flatnonzero(campaigns == holdout)
            train_indices = np.flatnonzero(campaigns != holdout)
            fold = {
                "holdout_campaign": holdout,
                "training_campaigns": sorted(
                    set(campaigns[train_indices].tolist())
                ),
                "training_scene_count": int(train_indices.size),
                "heldout_scene_count": int(test_indices.size),
                "methods": {},
            }
            for method in methods:
                fit_indices = (
                    test_indices
                    if method == "oracle_holdout_exhaustive"
                    else train_indices
                )
                fit_campaigns = campaigns[fit_indices]
                selected, candidate_count = _fit_one(
                    method=method,
                    fit_indices=fit_indices,
                    fit_campaigns=fit_campaigns,
                    states=states,
                    all_regrets=all_regrets,
                    exhaustive_actions=actions,
                    action_to_index=action_to_index,
                    epsilon_db=epsilon,
                    maximum_k=maximum_k,
                )
                if selected.size < 1:
                    raise ValueError(f"{method} selected an empty codebook")
                method_result = {
                    "candidate_action_count": candidate_count,
                    "selected_action_count_at_max_k": int(selected.size),
                    "selected_actions_at_max_k": [
                        list(actions[int(index)]) for index in selected
                    ],
                    "k_results": {},
                }
                for k_value in k_values:
                    prefix = selected[: min(k_value, selected.size)]
                    fit_metrics = evaluate_selected_actions(
                        regrets=all_regrets[fit_indices],
                        selected_candidate_indices=prefix,
                        epsilon_db=epsilon,
                        exact_action_payload_bits=exact_bits,
                        cvar_alpha=float(config["task"]["cvar_alpha"]),
                    )
                    heldout_metrics = evaluate_selected_actions(
                        regrets=all_regrets[test_indices],
                        selected_candidate_indices=prefix,
                        epsilon_db=epsilon,
                        exact_action_payload_bits=exact_bits,
                        cvar_alpha=float(config["task"]["cvar_alpha"]),
                    )
                    method_result["k_results"][str(k_value)] = {
                        "requested_k": k_value,
                        "actual_k": int(prefix.size),
                        "fit": fit_metrics,
                        "heldout": heldout_metrics,
                        "training_test_coverage_gap": float(
                            fit_metrics["coverage_rate"]
                            - heldout_metrics["coverage_rate"]
                        ),
                    }
                fold["methods"][method] = method_result
                print(
                    "Stage-6R feasibility "
                    f"N={n_channels} holdout={holdout} "
                    f"method={method} complete",
                    flush=True,
                )
            folds[holdout] = fold
        n_result = {
            "n_channels": n_channels,
            "demands": [
                int(query.demand_channels) for query in queries
            ],
            "scene_count": len(states),
            "campaign_counts": {
                value: int(np.count_nonzero(campaigns == value))
                for value in sorted(set(campaigns.tolist()))
            },
            "legal_action_tuple_count": len(actions),
            "exact_action_payload_bits": exact_bits,
            "folds": folds,
        }
        n_result["aggregate_by_method_k"] = _aggregate(
            n_result, methods, k_values
        )
        n_results[str(n_channels)] = n_result
        print(f"Stage-6R feasibility N={n_channels} complete", flush=True)

    result = {
        "version": "1.0",
        "status": "stage6r_codebook_feasibility_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "pilot_2023_cache": sha256_file(
                PROJECT_DIR / config["inputs"]["pilot_2023_cache"]
            ),
            "external_2024_2025_cache": sha256_file(external_cache),
            "external_access_state": sha256_file(
                PROJECT_DIR
                / config["inputs"]["external_access_state"]
            ),
        },
        "n_results": n_results,
        "decision_summary": _decision_summary(config, n_results),
        "checks": {
            "all_grid_points_retained": True,
            "random_scene_split_used": False,
            "all_methods_use_identical_outer_folds": True,
            "all_unsafe_coded_action_counts_zero": all(
                row["all_unsafe_coded_action_counts_zero"]
                for n_result in n_results.values()
                for method in methods
                for row in n_result["aggregate_by_method_k"][
                    method
                ].values()
            ),
            "external_confirmatory_final": False,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": (
            "This development experiment estimates fixed-action codebook "
            "feasibility on already-consumed 2023-2025 airborne data. It does "
            "not constitute a new external Final or an end-to-end system-bit "
            "claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "decision_summary": result["decision_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
