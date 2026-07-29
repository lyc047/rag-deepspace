#!/usr/bin/env python
"""Evaluate the Stage-6R temporal-consistency context gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

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
    context_signature,
    leave_one_context_novelty_threshold,
    robust_spectral_shape_features,
    stable_signature,
)
from spectrum_semcom.stage6r_regret_codebook import (  # noqa: E402
    evaluate_selected_actions,
    exhaustive_action_tuples,
    regret_matrix,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_temporal_context_gate_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/temporal_context_gate_v1/result.json"
)


def _exact_fallback_metrics(
    *, exact_bits: int, calibration_count: int, evaluation_count: int
) -> dict:
    total = calibration_count + evaluation_count
    return {
        "scene_count": evaluation_count,
        "codeword_count": 0,
        "symbol_width_bits": 0,
        "covered_count": 0,
        "coverage_rate": 0.0,
        "escape_count": evaluation_count,
        "escape_rate": 1.0,
        "used_codeword_count": 0,
        "codeword_utilization_rate": 0.0,
        "codeword_usage_entropy_bits": 0.0,
        "covered_mean_max_regret_db": 0.0,
        "covered_cvar_max_regret_db": 0.0,
        "covered_maximum_max_regret_db": 0.0,
        "exact_action_payload_bits": exact_bits,
        "expected_semantic_payload_bits_per_scene": float(exact_bits),
        "exact_payload_bits_per_scene": exact_bits,
        "payload_savings_percentage": 0.0,
        "unsafe_coded_action_count": 0,
        "usage_counts": [],
        "calibration_exact_payload_bits": calibration_count * exact_bits,
        "adaptation_control_bits": 0,
        "evaluation_semantic_payload_bits": evaluation_count * exact_bits,
        "full_session_accounted_bits": total * exact_bits,
        "full_session_bits_per_scene": float(exact_bits),
        "full_session_exact_baseline_bits": total * exact_bits,
        "full_session_payload_savings_percentage": 0.0,
    }


def _aggregate(rows: list[dict], k: int) -> dict:
    selected = [row for row in rows if row["k"] == k]
    coverage = np.asarray(
        [row["future"]["coverage_rate"] for row in selected],
        dtype=np.float64,
    )
    savings = np.asarray(
        [
            row["future"]["full_session_payload_savings_percentage"]
            for row in selected
        ],
        dtype=np.float64,
    )
    calibration = np.asarray(
        [row["calibration_scenes"] for row in selected], dtype=np.float64
    )
    return {
        "k": k,
        "fold_count": len(selected),
        "macro_compact_future_coverage": float(np.mean(coverage)),
        "worst_activity_compact_future_coverage": float(np.min(coverage)),
        "macro_full_session_payload_savings_percentage": float(
            np.mean(savings)
        ),
        "worst_full_session_payload_savings_percentage": float(
            np.min(savings)
        ),
        "mean_calibration_scenes": float(np.mean(calibration)),
        "maximum_calibration_scenes": int(np.max(calibration)),
        "unresolved_fold_count": int(
            sum(not row["resolved"] for row in selected)
        ),
        "ood_adaptation_fold_count": int(
            sum(row["decision"] == "OOD" for row in selected)
        ),
        "bank_reuse_fold_count": int(
            sum(
                isinstance(row["decision"], str)
                and row["decision"].startswith("BANK:")
                for row in selected
            )
        ),
        "all_unsafe_coded_action_counts_zero": all(
            row["future"]["unsafe_coded_action_count"] == 0
            for row in selected
        ),
    }


def _parent_context_aggregate(
    context_result: dict, n_channels: int, k: int, budget: int
) -> dict:
    key = f"K{k}_B{budget}_spectral_context_ood_adapter"
    return context_result["n_results"][str(n_channels)]["aggregate"][key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite temporal-gate result")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adaptation_config_path = (
        PROJECT_DIR / config["inputs"]["adaptation_config"]
    )
    adaptation_result_path = (
        PROJECT_DIR / config["inputs"]["adaptation_result"]
    )
    context_config_path = PROJECT_DIR / config["inputs"]["context_config"]
    context_result_path = PROJECT_DIR / config["inputs"]["context_result"]
    adaptation = json.loads(
        adaptation_config_path.read_text(encoding="utf-8")
    )
    adaptation_result = json.loads(
        adaptation_result_path.read_text(encoding="utf-8")
    )
    context_result = json.loads(
        context_result_path.read_text(encoding="utf-8")
    )
    campaigns_expected = list(adaptation["splitting"]["campaign_ids"])
    epsilon = float(adaptation["task"]["epsilon_db"])
    cvar_alpha = float(adaptation["task"]["cvar_alpha"])
    install_header = int(
        adaptation["protocol_accounting"][
            "adapted_codebook_install_header_bits"
        ]
    )
    bank_header = int(
        adaptation["protocol_accounting"][
            "preinstalled_bank_selection_header_bits"
        ]
    )
    window = int(config["task"]["window_size_scenes"])
    required = int(
        config["task"]["required_consecutive_matching_windows"]
    )
    checkpoints = list(map(int, config["task"]["checkpoints_scenes"]))
    minimum_bank_coverage = float(
        config["task"]["minimum_accumulated_bank_coverage"]
    )
    k_values = [
        int(config["task"]["secondary_k"]),
        int(config["task"]["primary_k"]),
    ]
    n_results = {}
    for n_channels in map(int, adaptation["task"]["n_channels"]):
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in powers
        ]
        actions = exhaustive_action_tuples(states)
        regrets = regret_matrix(states, actions)
        exact_bits = int(
            adaptation_result["n_results"][str(n_channels)][
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
            prototypes = {}
            for source in training_names:
                source_indices = np.flatnonzero(campaigns == source)
                prototypes[source] = robust_spectral_shape_features(
                    powers[source_indices]
                )
            novelty_threshold = leave_one_context_novelty_threshold(
                prototypes
            )
            for k in k_values:
                bank = {}
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
                signatures = []
                trace = []
                resolution = None
                for checkpoint in checkpoints:
                    if checkpoint > heldout.size:
                        break
                    decision_window = heldout[
                        checkpoint - window : checkpoint
                    ]
                    signature, source, distance = context_signature(
                        powers[decision_window],
                        prototypes,
                        novelty_threshold,
                    )
                    signatures.append(signature)
                    stable = stable_signature(signatures, required)
                    trace_row = {
                        "checkpoint_scenes": checkpoint,
                        "signature": signature,
                        "nearest_source": source,
                        "context_distance": distance,
                        "novelty_threshold": novelty_threshold,
                        "stable_signature": stable,
                        "accumulated_bank_coverage": None,
                        "accepted": False,
                    }
                    if stable is not None:
                        calibration = heldout[:checkpoint]
                        if stable == "OOD":
                            resolution = {
                                "checkpoint": checkpoint,
                                "decision": stable,
                                "adapted": True,
                                "selected": _fit(
                                    fit_indices=calibration,
                                    fit_campaigns=campaigns[calibration],
                                    all_regrets=regrets,
                                    actions=actions,
                                    epsilon_db=epsilon,
                                    k=k,
                                    balanced=False,
                                ),
                            }
                            trace_row["accepted"] = True
                        else:
                            bank_source = stable.removeprefix("BANK:")
                            selected = bank[bank_source]
                            calibration_coverage = float(
                                np.mean(
                                    np.min(
                                        regrets[
                                            np.ix_(calibration, selected)
                                        ],
                                        axis=1,
                                    )
                                    <= epsilon + 1e-12
                                )
                            )
                            trace_row[
                                "accumulated_bank_coverage"
                            ] = calibration_coverage
                            if (
                                calibration_coverage
                                >= minimum_bank_coverage
                            ):
                                resolution = {
                                    "checkpoint": checkpoint,
                                    "decision": stable,
                                    "adapted": False,
                                    "selected": selected,
                                }
                                trace_row["accepted"] = True
                    trace.append(trace_row)
                    if resolution is not None:
                        break
                if resolution is None:
                    calibration_count = min(
                        int(config["task"]["maximum_calibration_scenes"]),
                        heldout.size,
                    )
                    future_indices = heldout[calibration_count:]
                    future = _exact_fallback_metrics(
                        exact_bits=exact_bits,
                        calibration_count=calibration_count,
                        evaluation_count=int(future_indices.size),
                    )
                    decision = "UNRESOLVED_EXACT_FALLBACK"
                    adapted = False
                    selected_actions = []
                else:
                    calibration_count = int(resolution["checkpoint"])
                    future_indices = heldout[calibration_count:]
                    selected = np.asarray(
                        resolution["selected"], dtype=int
                    )
                    base = evaluate_selected_actions(
                        regrets=regrets[future_indices],
                        selected_candidate_indices=selected,
                        epsilon_db=epsilon,
                        exact_action_payload_bits=exact_bits,
                        cvar_alpha=cvar_alpha,
                    )
                    future = _metrics_with_accounting(
                        method="bank_or_adapt",
                        metrics=base,
                        exact_bits=exact_bits,
                        calibration_count=calibration_count,
                        evaluation_count=int(future_indices.size),
                        selected_count=int(selected.size),
                        install_header_bits=install_header,
                        bank_header_bits=bank_header,
                        adapted=bool(resolution["adapted"]),
                        oracle=False,
                    )
                    decision = str(resolution["decision"])
                    adapted = bool(resolution["adapted"])
                    selected_actions = [
                        list(actions[int(index)]) for index in selected
                    ]
                rows.append(
                    {
                        "holdout_campaign": holdout,
                        "k": k,
                        "resolved": resolution is not None,
                        "calibration_scenes": calibration_count,
                        "future_scene_count": int(future_indices.size),
                        "decision": decision,
                        "adapted": adapted,
                        "selected_actions": selected_actions,
                        "decision_trace": trace,
                        "future": future,
                    }
                )
            print(
                f"Stage-6R temporal gate N={n_channels} "
                f"holdout={holdout} complete",
                flush=True,
            )
        aggregates = {str(k): _aggregate(rows, k) for k in k_values}
        primary_k = int(config["task"]["primary_k"])
        primary = aggregates[str(primary_k)]
        gate = config["primary_gate"]
        parent_b2 = _parent_context_aggregate(
            context_result, n_channels, primary_k, 2
        )
        parent_b4 = _parent_context_aggregate(
            context_result, n_channels, primary_k, 4
        )
        comparison = {
            "worst_coverage_gain_over_single_prefix_B2": float(
                primary["worst_activity_compact_future_coverage"]
                - parent_b2["worst_activity_future_coverage"]
            ),
            "macro_coverage_loss_vs_single_prefix_B4": float(
                parent_b4["macro_future_coverage"]
                - primary["macro_compact_future_coverage"]
            ),
            "single_prefix_B2": parent_b2,
            "single_prefix_B4": parent_b4,
            "single_prefix_B8": _parent_context_aggregate(
                context_result, n_channels, primary_k, 8
            ),
        }
        gate_passed = bool(
            primary["macro_compact_future_coverage"]
            >= float(gate["minimum_macro_compact_future_coverage"])
            and primary["worst_activity_compact_future_coverage"]
            >= float(
                gate["minimum_worst_activity_compact_future_coverage"]
            )
            and primary["mean_calibration_scenes"]
            <= float(gate["maximum_mean_calibration_scenes"])
            and primary["unresolved_fold_count"]
            <= int(gate["maximum_unresolved_fold_count"])
            and primary[
                "macro_full_session_payload_savings_percentage"
            ]
            > float(
                gate[
                    "minimum_macro_full_session_payload_savings_percentage"
                ]
            )
            and primary["all_unsafe_coded_action_counts_zero"]
            and comparison[
                "worst_coverage_gain_over_single_prefix_B2"
            ]
            >= float(
                config["secondary_checks"][
                    "minimum_worst_coverage_gain_over_single_prefix_B2"
                ]
            )
            and comparison["macro_coverage_loss_vs_single_prefix_B4"]
            <= float(
                config["secondary_checks"][
                    "maximum_macro_coverage_loss_vs_single_prefix_B4"
                ]
            )
        )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "rows": rows,
            "aggregate_by_k": aggregates,
            "primary_comparison": comparison,
            "primary_gate_passed": gate_passed,
        }
    passed_n = [
        int(key)
        for key, value in n_results.items()
        if value["primary_gate_passed"]
    ]
    all_required = bool(
        config["secondary_checks"]["all_n_values_required_for_primary_gate"]
    )
    retained = (
        len(passed_n) == len(n_results)
        if all_required
        else len(passed_n) >= 3
    )
    result = {
        "version": "1.0",
        "status": "stage6r_temporal_context_gate_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "adaptation_config": sha256_file(adaptation_config_path),
            "adaptation_result": sha256_file(adaptation_result_path),
            "context_config": sha256_file(context_config_path),
            "context_result": sha256_file(context_result_path),
        },
        "n_results": n_results,
        "decision_summary": {
            "retention_gate_passed": retained,
            "gate_passing_n_values": passed_n,
            "recommended_next_route": (
                "regret_aware_candidate_ranker_with_temporal_context"
                if retained
                else "revise_temporal_context_before_candidate_ranker"
            ),
        },
        "checks": {
            "all_unsafe_coded_action_counts_zero": all(
                row["future"]["unsafe_coded_action_count"] == 0
                for value in n_results.values()
                for row in value["rows"]
            ),
            "future_suffix_used_for_online_decision": False,
            "unresolved_sessions_fail_closed": True,
            "new_final_data_accessed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision_summary": result["decision_summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
