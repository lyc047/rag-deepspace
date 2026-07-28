#!/usr/bin/env python
"""Evaluate the Stage-6R spectral-context OOD codebook adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_stage6r_few_shot_adaptation import (  # noqa: E402
    _fit,
    _load_combined,
    _metrics_with_accounting,
    _queries,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6r_context_adapter import (  # noqa: E402
    leave_one_context_novelty_threshold,
    nearest_context,
    robust_spectral_shape_features,
)
from spectrum_semcom.stage6r_regret_codebook import (  # noqa: E402
    evaluate_selected_actions,
    exhaustive_action_tuples,
    regret_matrix,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage6r_context_adapter_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/context_adapter_v1/result.json"
)


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        key = (row["k"], row["calibration_budget"], row["method"])
        groups.setdefault(key, []).append(row)
    output = {}
    for (k, budget, method), values in groups.items():
        coverage = np.asarray(
            [row["future"]["coverage_rate"] for row in values]
        )
        savings = np.asarray(
            [
                row["future"]["full_session_payload_savings_percentage"]
                for row in values
            ]
        )
        output[f"K{k}_B{budget}_{method}"] = {
            "k": int(k),
            "calibration_budget": int(budget),
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
            "ood_adaptation_count": int(
                sum(bool(row["ood_detected"]) for row in values)
            ),
        }
    return output


def _passes(row: dict, gate: dict) -> bool:
    return bool(
        row["k"] <= int(gate["maximum_k"])
        and row["calibration_budget"]
        <= int(gate["maximum_calibration_scenes"])
        and row["macro_future_coverage"]
        >= float(gate["minimum_macro_future_coverage"])
        and row["worst_activity_future_coverage"]
        >= float(gate["minimum_worst_activity_future_coverage"])
        and row["macro_full_session_payload_savings_percentage"]
        > float(
            gate[
                "minimum_macro_full_session_payload_savings_percentage"
            ]
        )
        and row["all_unsafe_coded_action_counts_zero"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6R context result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_config_path = PROJECT_DIR / config["inputs"]["parent_config"]
    parent_result_path = PROJECT_DIR / config["inputs"]["parent_result"]
    parent = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    campaigns_expected = list(parent["splitting"]["campaign_ids"])
    epsilon = float(parent["task"]["epsilon_db"])
    cvar_alpha = float(parent["task"]["cvar_alpha"])
    install_header = int(
        parent["protocol_accounting"][
            "adapted_codebook_install_header_bits"
        ]
    )
    bank_header = int(
        parent["protocol_accounting"][
            "preinstalled_bank_selection_header_bits"
        ]
    )
    multiplier = float(config["representation"]["threshold_multiplier"])
    n_results = {}
    for n_channels in map(int, parent["task"]["n_channels"]):
        powers, campaigns, timestamps = _load_combined(parent, n_channels)
        queries = _queries(parent, n_channels)
        states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in powers
        ]
        actions = exhaustive_action_tuples(states)
        regrets = regret_matrix(states, actions)
        exact_bits = int(
            parent_result["n_results"][str(n_channels)][
                "exact_action_payload_bits"
            ]
        )
        rows = []
        for holdout in campaigns_expected:
            heldout = np.flatnonzero(campaigns == holdout)
            heldout = heldout[
                np.argsort(timestamps[heldout], kind="stable")
            ]
            training_names = sorted(set(campaigns_expected) - {holdout})
            training = np.flatnonzero(campaigns != holdout)
            for k in map(int, parent["task"]["k_values"]):
                global_selected = _fit(
                    fit_indices=training,
                    fit_campaigns=campaigns[training],
                    all_regrets=regrets,
                    actions=actions,
                    epsilon_db=epsilon,
                    k=k,
                    balanced=True,
                )
                bank = {}
                prototypes = {}
                for source in training_names:
                    source_indices = np.flatnonzero(campaigns == source)
                    bank[source] = _fit(
                        fit_indices=source_indices,
                        fit_campaigns=campaigns[source_indices],
                        all_regrets=regrets,
                        actions=actions,
                        epsilon_db=epsilon,
                        k=k,
                        balanced=False,
                    )
                    prototypes[source] = robust_spectral_shape_features(
                        powers[source_indices]
                    )
                threshold = (
                    multiplier
                    * leave_one_context_novelty_threshold(prototypes)
                )
                for budget in map(
                    int, parent["task"]["calibration_scene_budgets"]
                ):
                    if budget >= heldout.size:
                        continue
                    calibration = heldout[:budget]
                    future = heldout[budget:]
                    feature = robust_spectral_shape_features(
                        powers[calibration]
                    )
                    source, distance = nearest_context(
                        feature, prototypes
                    )
                    adapted = bool(distance > threshold)
                    if adapted:
                        selected = _fit(
                            fit_indices=calibration,
                            fit_campaigns=campaigns[calibration],
                            all_regrets=regrets,
                            actions=actions,
                            epsilon_db=epsilon,
                            k=k,
                            balanced=False,
                        )
                        selected_source = "new_adapted_codebook"
                    else:
                        selected = bank[source]
                        selected_source = source
                    base = evaluate_selected_actions(
                        regrets=regrets[future],
                        selected_candidate_indices=selected,
                        epsilon_db=epsilon,
                        exact_action_payload_bits=exact_bits,
                        cvar_alpha=cvar_alpha,
                    )
                    future_metrics = _metrics_with_accounting(
                        method="bank_or_adapt",
                        metrics=base,
                        exact_bits=exact_bits,
                        calibration_count=budget,
                        evaluation_count=future.size,
                        selected_count=selected.size,
                        install_header_bits=install_header,
                        bank_header_bits=bank_header,
                        adapted=adapted,
                        oracle=False,
                    )
                    rows.append(
                        {
                            "holdout_campaign": holdout,
                            "k": k,
                            "calibration_budget": budget,
                            "method": "spectral_context_ood_adapter",
                            "context_distance": distance,
                            "training_only_ood_threshold": threshold,
                            "ood_detected": adapted,
                            "codebook_source": selected_source,
                            "selected_action_count": int(selected.size),
                            "future": future_metrics,
                        }
                    )
            print(
                f"Stage-6R context N={n_channels} "
                f"holdout={holdout} complete",
                flush=True,
            )
        aggregate = _aggregate(rows)
        gate_rows = [
            row
            for row in aggregate.values()
            if row["method"] == "spectral_context_ood_adapter"
            and _passes(row, config["primary_gate"])
        ]
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "rows": rows,
            "aggregate": aggregate,
            "gate_passed": bool(gate_rows),
            "best_gate_passing_configuration": (
                min(
                    gate_rows,
                    key=lambda row: (
                        row["calibration_budget"],
                        row["k"],
                        -row["worst_activity_future_coverage"],
                    ),
                )
                if gate_rows
                else None
            ),
        }
    passed_n = [
        int(key)
        for key, value in n_results.items()
        if value["gate_passed"]
    ]
    result = {
        "version": "1.0",
        "status": "stage6r_context_adapter_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "parent_config": sha256_file(parent_config_path),
            "parent_result": sha256_file(parent_result_path),
        },
        "n_results": n_results,
        "decision_summary": {
            "retention_gate_passed": len(passed_n) >= 3,
            "gate_passing_n_values": passed_n,
            "recommended_next_route": (
                "train_regret_aware_codeword_ranker_with_context_ood_gate"
                if len(passed_n) >= 3
                else "improve_context_representation_before_ranker"
            ),
        },
        "checks": {
            "all_unsafe_coded_action_counts_zero": all(
                row["future"]["unsafe_coded_action_count"] == 0
                for value in n_results.values()
                for row in value["rows"]
            ),
            "threshold_uses_holdout_data": False,
            "chronological_prefix_split_used": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
