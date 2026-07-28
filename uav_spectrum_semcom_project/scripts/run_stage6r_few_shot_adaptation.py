#!/usr/bin/env python
"""Run the registered Stage-6R chronological few-shot adaptation study."""

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
    campaign_balanced_weights,
    evaluate_selected_actions,
    exhaustive_action_tuples,
    greedy_select_actions,
    regret_matrix,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage6r_few_shot_adaptation_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/few_shot_adaptation_v1/result.json"
)
PILOT_CAMPAIGN = "aerpaw_packapalooza_2023"


def _load_combined(
    config: dict, n_channels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pilot_path = PROJECT_DIR / config["inputs"]["pilot_2023_cache"]
    external_path = PROJECT_DIR / config["inputs"]["external_2024_2025_cache"]
    with np.load(pilot_path, allow_pickle=False) as pilot:
        pilot_power = pilot[f"channel_power_n{n_channels}"].astype(np.float64)
        pilot_time = pilot["timestamps_utc"].astype(str)
    with np.load(external_path, allow_pickle=False) as external:
        external_power = external[f"channel_power_n{n_channels}"].astype(
            np.float64
        )
        external_campaigns = external["campaign_ids"].astype(str)
        external_time = external["timestamps_local"].astype(str)
    return (
        np.concatenate((pilot_power, external_power), axis=0),
        np.concatenate(
            (
                np.full(pilot_power.shape[0], PILOT_CAMPAIGN),
                external_campaigns,
            )
        ),
        np.concatenate((pilot_time, external_time)),
    )


def _queries(config: dict, n_channels: int) -> tuple[SpectrumTaskQuery, ...]:
    return tuple(
        SpectrumTaskQuery(int(round(n_channels * float(ratio))))
        for ratio in config["task"]["demand_ratios"]
    )


def _fit(
    *,
    fit_indices: np.ndarray,
    fit_campaigns: np.ndarray,
    all_regrets: np.ndarray,
    actions: tuple[tuple[int, ...], ...],
    epsilon_db: float,
    k: int,
    balanced: bool,
) -> np.ndarray:
    weights = (
        campaign_balanced_weights(fit_campaigns)
        if balanced
        else np.ones(fit_indices.size, dtype=np.float64)
    )
    selected = greedy_select_actions(
        regrets=all_regrets[fit_indices],
        actions=actions,
        epsilon_db=epsilon_db,
        weights=weights,
        max_codewords=k,
    )
    lookup = {value: index for index, value in enumerate(actions)}
    return np.asarray([lookup[value] for value in selected.actions], dtype=int)


def _bank_choice(
    *,
    bank: dict[str, np.ndarray],
    calibration_indices: np.ndarray,
    all_regrets: np.ndarray,
    epsilon_db: float,
) -> tuple[str, np.ndarray, float]:
    rows = []
    for name, selected in bank.items():
        values = all_regrets[np.ix_(calibration_indices, selected)]
        best = np.min(values, axis=1)
        covered = best <= epsilon_db + 1e-12
        coverage = float(np.mean(covered))
        mean_regret = float(np.mean(best))
        rows.append((-coverage, mean_regret, name, selected))
    _, _, name, selected = min(rows, key=lambda row: row[:3])
    return name, selected, float(-min(rows, key=lambda row: row[:3])[0])


def _metrics_with_accounting(
    *,
    method: str,
    metrics: dict,
    exact_bits: int,
    calibration_count: int,
    evaluation_count: int,
    selected_count: int,
    install_header_bits: int,
    bank_header_bits: int,
    adapted: bool,
    oracle: bool,
) -> dict:
    calibration_bits = calibration_count * exact_bits
    if oracle:
        control_bits = 0
    elif adapted:
        control_bits = install_header_bits + selected_count * exact_bits
    elif method == "preinstalled_bank_selector" or method == "bank_or_adapt":
        control_bits = bank_header_bits
    else:
        control_bits = 0
    evaluation_bits = (
        evaluation_count * metrics["expected_semantic_payload_bits_per_scene"]
    )
    total_scenes = calibration_count + evaluation_count
    exact_total = total_scenes * exact_bits
    total_bits = calibration_bits + control_bits + evaluation_bits
    output = dict(metrics)
    output.update(
        {
            "calibration_exact_payload_bits": int(calibration_bits),
            "adaptation_control_bits": int(control_bits),
            "evaluation_semantic_payload_bits": float(evaluation_bits),
            "full_session_accounted_bits": float(total_bits),
            "full_session_bits_per_scene": float(total_bits / total_scenes),
            "full_session_exact_baseline_bits": int(exact_total),
            "full_session_payload_savings_percentage": float(
                100.0 * (1.0 - total_bits / exact_total)
            ),
        }
    )
    return output


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        key = (
            int(row["k"]),
            int(row["calibration_budget"]),
            str(row["method"]),
        )
        groups.setdefault(key, []).append(row)
    output = {}
    for (k, budget, method), values in groups.items():
        key = f"K{k}_B{budget}_{method}"
        coverage = np.asarray(
            [row["future"]["coverage_rate"] for row in values],
            dtype=np.float64,
        )
        savings = np.asarray(
            [
                row["future"]["full_session_payload_savings_percentage"]
                for row in values
            ],
            dtype=np.float64,
        )
        output[key] = {
            "k": k,
            "calibration_budget": budget,
            "method": method,
            "fold_count": len(values),
            "macro_future_coverage": float(np.mean(coverage)),
            "worst_activity_future_coverage": float(np.min(coverage)),
            "macro_full_session_payload_savings_percentage": float(
                np.mean(savings)
            ),
            "worst_full_session_payload_savings_percentage": float(
                np.min(savings)
            ),
            "all_unsafe_coded_action_counts_zero": all(
                row["future"]["unsafe_coded_action_count"] == 0
                for row in values
            ),
            "adapted_fold_count": int(
                sum(bool(row["adapted"]) for row in values)
            ),
        }
    return output


def _decision_summary(n_results: dict) -> dict:
    viable_n = []
    bank_n = []
    reject_global_n = []
    best_by_n = {}
    for n_key, result in n_results.items():
        aggregates = list(result["aggregate"].values())
        viable = [
            row
            for row in aggregates
            if row["method"] in {"calibration_only_adapt", "bank_or_adapt"}
            and row["k"] <= 4
            and row["calibration_budget"] <= 8
            and row["macro_future_coverage"] >= 0.90
            and row["worst_activity_future_coverage"] >= 0.80
            and row["macro_full_session_payload_savings_percentage"] > 0.0
            and row["all_unsafe_coded_action_counts_zero"]
        ]
        banks = [
            row
            for row in aggregates
            if row["method"] == "preinstalled_bank_selector"
            and row["k"] <= 4
            and row["calibration_budget"] <= 8
            and row["macro_future_coverage"] >= 0.90
            and row["worst_activity_future_coverage"] >= 0.80
            and row["macro_full_session_payload_savings_percentage"] > 0.0
            and row["all_unsafe_coded_action_counts_zero"]
        ]
        global_rows = [
            row
            for row in aggregates
            if row["method"] == "zero_shot_global"
            and row["k"] == 8
        ]
        if viable:
            viable_n.append(int(n_key))
            best = min(
                viable,
                key=lambda row: (
                    row["calibration_budget"],
                    row["k"],
                    -row["worst_activity_future_coverage"],
                ),
            )
            best_by_n[n_key] = best
        if banks:
            bank_n.append(int(n_key))
        if global_rows and max(
            row["worst_activity_future_coverage"] for row in global_rows
        ) < 0.80:
            reject_global_n.append(int(n_key))
    adaptation_viable = len(viable_n) >= 3
    bank_viable = len(bank_n) >= 3
    return {
        "few_shot_adaptation_viable": adaptation_viable,
        "few_shot_viable_n_values": viable_n,
        "best_gate_passing_configuration_by_n": best_by_n,
        "preinstalled_bank_reuse_viable": bank_viable,
        "bank_reuse_viable_n_values": bank_n,
        "reject_direct_global_vq": len(reject_global_n) >= 3,
        "direct_global_vq_reject_n_values": reject_global_n,
        "ai_ranker_entry_gate": bool(adaptation_viable and not bank_viable),
        "recommended_next_route": (
            "regret_aware_few_shot_candidate_ranker"
            if adaptation_viable and not bank_viable
            else "preinstalled_context_codebook_selector"
            if bank_viable
            else "hierarchical_self_contained_action_representation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6R adaptation result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    state_path = PROJECT_DIR / config["inputs"]["external_access_state"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("status")
        != config["inputs"]["required_external_access_status"]
        or state.get("access_count") != 1
        or state.get("reset_permitted") is not False
    ):
        raise ValueError("consumed external-data state changed")

    epsilon = float(config["task"]["epsilon_db"])
    cvar_alpha = float(config["task"]["cvar_alpha"])
    campaigns_expected = list(config["splitting"]["campaign_ids"])
    install_header = int(
        config["protocol_accounting"]["adapted_codebook_install_header_bits"]
    )
    bank_header = int(
        config["protocol_accounting"][
            "preinstalled_bank_selection_header_bits"
        ]
    )
    threshold = float(
        config["bank_or_adapt"][
            "minimum_calibration_coverage_to_reuse_bank"
        ]
    )
    n_results = {}
    for n_channels in map(int, config["task"]["n_channels"]):
        powers, campaigns, timestamps = _load_combined(config, n_channels)
        if set(campaigns.tolist()) != set(campaigns_expected):
            raise ValueError("airborne campaign set differs from protocol")
        queries = _queries(config, n_channels)
        states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in powers
        ]
        actions = exhaustive_action_tuples(states)
        all_regrets = regret_matrix(states, actions)
        exact_bits = int(
            sum(
                math.ceil(
                    math.log2(n_channels - query.demand_channels + 1)
                )
                for query in queries
            )
        )
        n_rows = []
        for holdout in campaigns_expected:
            heldout = np.flatnonzero(campaigns == holdout)
            heldout = heldout[np.argsort(timestamps[heldout], kind="stable")]
            training = np.flatnonzero(campaigns != holdout)
            for k in map(int, config["task"]["k_values"]):
                global_selected = _fit(
                    fit_indices=training,
                    fit_campaigns=campaigns[training],
                    all_regrets=all_regrets,
                    actions=actions,
                    epsilon_db=epsilon,
                    k=k,
                    balanced=True,
                )
                bank = {}
                for source in sorted(set(campaigns[training].tolist())):
                    source_indices = np.flatnonzero(campaigns == source)
                    bank[source] = _fit(
                        fit_indices=source_indices,
                        fit_campaigns=campaigns[source_indices],
                        all_regrets=all_regrets,
                        actions=actions,
                        epsilon_db=epsilon,
                        k=k,
                        balanced=False,
                    )
                for budget in map(
                    int, config["task"]["calibration_scene_budgets"]
                ):
                    if budget >= heldout.size:
                        continue
                    calibration = heldout[:budget]
                    future = heldout[budget:]
                    adapted_selected = _fit(
                        fit_indices=calibration,
                        fit_campaigns=campaigns[calibration],
                        all_regrets=all_regrets,
                        actions=actions,
                        epsilon_db=epsilon,
                        k=k,
                        balanced=False,
                    )
                    bank_name, bank_selected, bank_calibration_coverage = (
                        _bank_choice(
                            bank=bank,
                            calibration_indices=calibration,
                            all_regrets=all_regrets,
                            epsilon_db=epsilon,
                        )
                    )
                    if bank_calibration_coverage >= threshold:
                        hybrid_selected = bank_selected
                        hybrid_adapted = False
                        hybrid_source = bank_name
                    else:
                        hybrid_selected = adapted_selected
                        hybrid_adapted = True
                        hybrid_source = "new_adapted_codebook"
                    oracle_selected = _fit(
                        fit_indices=future,
                        fit_campaigns=campaigns[future],
                        all_regrets=all_regrets,
                        actions=actions,
                        epsilon_db=epsilon,
                        k=k,
                        balanced=False,
                    )
                    method_rows = [
                        (
                            "zero_shot_global",
                            global_selected,
                            False,
                            "global_training_codebook",
                            False,
                        ),
                        (
                            "preinstalled_bank_selector",
                            bank_selected,
                            False,
                            bank_name,
                            False,
                        ),
                        (
                            "calibration_only_adapt",
                            adapted_selected,
                            True,
                            "new_adapted_codebook",
                            False,
                        ),
                        (
                            "bank_or_adapt",
                            hybrid_selected,
                            hybrid_adapted,
                            hybrid_source,
                            False,
                        ),
                        (
                            "oracle_future_upper_bound",
                            oracle_selected,
                            False,
                            "future_oracle",
                            True,
                        ),
                    ]
                    for method, selected, adapted, source, oracle in method_rows:
                        base = evaluate_selected_actions(
                            regrets=all_regrets[future],
                            selected_candidate_indices=selected,
                            epsilon_db=epsilon,
                            exact_action_payload_bits=exact_bits,
                            cvar_alpha=cvar_alpha,
                        )
                        future_metrics = _metrics_with_accounting(
                            method=method,
                            metrics=base,
                            exact_bits=exact_bits,
                            calibration_count=budget,
                            evaluation_count=future.size,
                            selected_count=selected.size,
                            install_header_bits=install_header,
                            bank_header_bits=bank_header,
                            adapted=adapted,
                            oracle=oracle,
                        )
                        n_rows.append(
                            {
                                "holdout_campaign": holdout,
                                "k": k,
                                "calibration_budget": budget,
                                "future_scene_count": int(future.size),
                                "method": method,
                                "selected_action_count": int(selected.size),
                                "selected_actions": [
                                    list(actions[int(index)])
                                    for index in selected
                                ],
                                "adapted": bool(adapted),
                                "codebook_source": source,
                                "bank_calibration_coverage": (
                                    bank_calibration_coverage
                                    if method
                                    in {
                                        "preinstalled_bank_selector",
                                        "bank_or_adapt",
                                    }
                                    else None
                                ),
                                "future": future_metrics,
                            }
                        )
            print(
                f"Stage-6R adaptation N={n_channels} "
                f"holdout={holdout} complete",
                flush=True,
            )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": [
                int(query.demand_channels) for query in queries
            ],
            "exact_action_payload_bits": exact_bits,
            "legal_action_tuple_count": len(actions),
            "rows": n_rows,
            "aggregate": _aggregate(n_rows),
        }
    result = {
        "version": "1.0",
        "status": "stage6r_few_shot_adaptation_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "pilot_2023_cache": sha256_file(
                PROJECT_DIR / config["inputs"]["pilot_2023_cache"]
            ),
            "external_2024_2025_cache": sha256_file(
                PROJECT_DIR / config["inputs"]["external_2024_2025_cache"]
            ),
            "external_access_state": sha256_file(state_path),
        },
        "n_results": n_results,
        "decision_summary": _decision_summary(n_results),
        "checks": {
            "all_unsafe_coded_action_counts_zero": all(
                row["future"]["unsafe_coded_action_count"] == 0
                for n_result in n_results.values()
                for row in n_result["rows"]
            ),
            "external_access_count_unchanged": state.get("access_count") == 1,
            "chronological_prefix_split_used": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
