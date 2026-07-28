#!/usr/bin/env python
"""Analyze the frozen Stage-6 external Final at matched clean."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from analyze_stage6_grouped_validation import (  # noqa: E402
    _bootstrap_plan,
    _leave_one_site_out,
    _metric_estimate,
    _paired_savings,
)
from analyze_stage6_task_scale_sensitivity import (  # noqa: E402
    _adjust_rows,
    _exact_row,
    _semantic_row,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    verify_stage6_code_snapshot,
)
from spectrum_semcom.stage6_task_scale_sensitivity import (  # noqa: E402
    codebook_install_break_even_scenes,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json"
)
DEFAULT_STATE = (
    PROJECT_DIR / "configs/stage6_external_final_access_state.json"
)
DEFAULT_SNAPSHOT = (
    PROJECT_DIR / "results/stage6/external_final_freeze_v1/code_snapshot.json"
)
DEFAULT_RAW = (
    PROJECT_DIR / "results/stage6/external_final_v1/final_raw.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6/external_final_v1/final_analysis.json"
)


def analyze_n(
    *,
    n_channels: int,
    n_position: int,
    raw_n: dict,
    protocol: dict,
) -> dict:
    if raw_n["status"] != "evaluated":
        return {
            "n_channels": n_channels,
            "status": raw_n["status"],
            "representation": raw_n["representation"],
            "matched_clean_comparisons": [],
        }
    group_ids = list(raw_n["configured_outer_group_ids"])
    groups = [raw_n["outer_groups"][value] for value in group_ids]
    campaigns = [value["site_id"] for value in groups]
    semantic_rows = [_semantic_row(value) for value in groups]
    evaluation = protocol["system_evaluation"]
    replicates = int(evaluation["paired_bootstrap_replicates"])
    confidence = float(evaluation["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    trajectory_count = len(semantic_rows[0]["trajectory_metrics"])
    seed = int(protocol["statistics"]["bootstrap_seed"]) + n_position * 100_000
    group_draws, trajectory_draws = _bootstrap_plan(
        len(groups), trajectory_count, replicates, seed
    )
    semantic_estimates = {}
    for metric in (
        "clean_rate",
        "availability_rate",
        "actual_bits_per_scene",
        "resource_equivalent_bits_per_scene",
        "effective_mean_regret_db",
        "effective_cvar_0_9_regret_db",
        "conditional_mean_regret_db",
        "conditional_cvar_0_9_regret_db",
    ):
        semantic_estimates[metric] = _metric_estimate(
            semantic_rows,
            metric,
            group_draws,
            trajectory_draws,
            alpha,
        )[0]
    exact_by_attempts = {}
    for attempts in evaluation["exact_reference_open_loop_attempts"]:
        rows = [_exact_row(value, int(attempts)) for value in groups]
        exact_by_attempts[int(attempts)] = {
            "rows": rows,
            "clean_rate": _metric_estimate(
                rows,
                "clean_rate",
                group_draws,
                trajectory_draws,
                alpha,
            )[0],
            "bits_per_scene": _metric_estimate(
                rows,
                "actual_bits_per_scene",
                group_draws,
                trajectory_draws,
                alpha,
            )[0],
        }
    targets = [
        float(evaluation["primary_matched_clean_target"]),
        *map(float, evaluation["auxiliary_matched_clean_targets"]),
    ]
    comparisons = []
    wrong_actions = sum(
        float(
            value["semantic_workpoint"]["summary"][
                "wrong_codebook_decode_count"
            ]
        )
        for value in groups
    )
    for target_position, target in enumerate(dict.fromkeys(targets)):
        eligible = [
            (attempts, value)
            for attempts, value in exact_by_attempts.items()
            if value["clean_rate"]["confidence_interval_lower"] >= target
        ]
        base = {
            "target_clean_rate": target,
            "semantic_clean_rate": semantic_estimates["clean_rate"],
            "semantic_bits_per_scene": semantic_estimates[
                "actual_bits_per_scene"
            ],
        }
        if (
            semantic_estimates["clean_rate"]["confidence_interval_lower"]
            < target
            or not eligible
        ):
            comparisons.append(
                base
                | {
                    "status": "no_common_eligible_workpoint",
                    "eligible_exact_attempts": [
                        value[0] for value in eligible
                    ],
                    "passes_primary_acceptance": False,
                    "passes_strong_effect": False,
                }
            )
            continue
        attempts, exact = min(
            eligible,
            key=lambda value: (
                value[1]["bits_per_scene"]["point_estimate"],
                value[0],
            ),
        )
        savings = _paired_savings(
            semantic_rows,
            exact["rows"],
            "actual_bits_per_scene",
            group_draws,
            trajectory_draws,
            alpha,
        )
        leave_one_campaign = _leave_one_site_out(
            group_ids,
            campaigns,
            semantic_rows,
            exact["rows"],
            replicates=replicates,
            seed=seed + 10_000 + target_position * 100,
            alpha=alpha,
        )
        direction_stable = all(
            value["resource_savings_percentage"]["point_estimate"] > 0.0
            for value in leave_one_campaign
        )
        accepted = (
            semantic_estimates["clean_rate"]["confidence_interval_lower"]
            >= target
            and savings["confidence_interval_lower"] > 0.0
            and direction_stable
            and wrong_actions == 0.0
        )
        strong = accepted and savings["point_estimate"] >= 15.0
        representation = raw_n["representation"]
        install_bits = int(representation["context_install_bits"])
        amortized = {}
        for horizon in evaluation["amortized_codebook_horizons_scenes"]:
            adjusted = _adjust_rows(
                semantic_rows,
                "actual_bits_per_scene",
                install_bits / int(horizon),
            )
            amortized[str(int(horizon))] = {
                "semantic_bits_per_scene": _metric_estimate(
                    adjusted,
                    "actual_bits_per_scene",
                    group_draws,
                    trajectory_draws,
                    alpha,
                )[0],
                "savings_percentage": _paired_savings(
                    adjusted,
                    exact["rows"],
                    "actual_bits_per_scene",
                    group_draws,
                    trajectory_draws,
                    alpha,
                ),
            }
        comparisons.append(
            base
            | {
                "status": "matched",
                "selected_exact_attempts": attempts,
                "exact_clean_rate": exact["clean_rate"],
                "exact_bits_per_scene": exact["bits_per_scene"],
                "savings_percentage": savings,
                "amortized": amortized,
                "codebook_install_break_even_scenes": (
                    codebook_install_break_even_scenes(
                        semantic_estimates["actual_bits_per_scene"][
                            "point_estimate"
                        ],
                        exact["bits_per_scene"]["point_estimate"],
                        install_bits,
                    )
                ),
                "leave_one_campaign_out": leave_one_campaign,
                "leave_one_campaign_out_direction_stable": direction_stable,
                "wrong_codebook_action_count": wrong_actions,
                "passes_primary_acceptance": accepted,
                "passes_strong_effect": strong,
            }
        )
    return {
        "n_channels": n_channels,
        "status": "analyzed",
        "outer_group_count": len(groups),
        "campaign_ids": campaigns,
        "representation": raw_n["representation"],
        "semantic_metrics": semantic_estimates,
        "campaign_strata": {
            value["site_id"]: {
                "scene_count": value["evaluated_scene_count"],
                "semantic_summary": value["semantic_workpoint"]["summary"],
                "exact_summaries": [
                    {
                        "attempts": item["task_packet_open_loop_attempts"],
                        "summary": item["summary"],
                    }
                    for item in value["exact_workpoints"]
                ],
            }
            for value in groups
        },
        "matched_clean_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 Final analysis")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    errors = verify_stage6_code_snapshot(PROJECT_DIR, snapshot)
    if errors:
        raise ValueError("frozen Stage-6 code changed: " + "; ".join(errors))
    if (
        state.get("access_count") != 1
        or state.get("final_signal_values_accessed") is not True
        or state.get("reset_permitted") is not False
    ):
        raise ValueError("Stage-6 Final access receipt is invalid")
    if raw.get("status") != "stage6_external_final_execution_complete":
        raise ValueError("Stage-6 Final raw result is incomplete")
    if raw.get("code_snapshot_sha256") != snapshot.get(
        "executable_snapshot_sha256"
    ):
        raise ValueError("raw result is not bound to frozen snapshot")
    if state.get("raw_result_sha256") != sha256_file(args.raw):
        raise ValueError("raw result hash differs from access state")
    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    analysis_by_n = {
        str(n_channels): analyze_n(
            n_channels=n_channels,
            n_position=n_position,
            raw_n=raw["n_results"][str(n_channels)],
            protocol=protocol,
        )
        for n_position, n_channels in enumerate(n_values)
    }
    target = float(
        protocol["system_evaluation"]["primary_matched_clean_target"]
    )
    primary_by_n = {}
    for n_channels in n_values:
        comparisons = analysis_by_n[str(n_channels)][
            "matched_clean_comparisons"
        ]
        matches = [
            value
            for value in comparisons
            if abs(float(value["target_clean_rate"]) - target) < 1e-12
        ]
        primary_by_n[str(n_channels)] = (
            matches[0]
            if matches
            else {
                "status": "missing",
                "passes_primary_acceptance": False,
                "passes_strong_effect": False,
            }
        )
    passing_n = [
        n_channels
        for n_channels in n_values
        if primary_by_n[str(n_channels)].get(
            "passes_primary_acceptance", False
        )
    ]
    primary_n = int(
        protocol["decision_rules"]["primary_configuration"]["n_channels"]
    )
    all_campaigns = set(raw["campaign_ids"]) == set(
        protocol["sampling"]["campaign_allocations"]
    )
    wrong_zero = bool(
        raw["checks"]["wrong_codebook_actions_must_be_zero"]
    )
    scale_support = len(passing_n) >= 3
    primary_pass = primary_by_n[str(primary_n)].get(
        "passes_primary_acceptance", False
    )
    stage6_supported = bool(
        primary_pass and scale_support and all_campaigns and wrong_zero
    )
    result = {
        "version": "1.0",
        "status": "stage6_external_final_analysis_complete",
        "verification_status": "VERIFIED",
        "protocol_sha256": sha256_file(args.protocol),
        "raw_result_sha256": sha256_file(args.raw),
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "access_receipt_sha256": state["receipt_sha256"],
        "bootstrap": {
            "replicates": int(
                protocol["system_evaluation"][
                    "paired_bootstrap_replicates"
                ]
            ),
            "confidence_level": float(
                protocol["system_evaluation"]["confidence_level"]
            ),
            "outer_unit": "campaign",
            "inner_unit": "paired_link_trajectory",
        },
        "analysis_by_n": analysis_by_n,
        "decision_summary": {
            "primary_n_channels": primary_n,
            "primary_target_clean_rate": target,
            "primary_configuration_passed": bool(primary_pass),
            "n_values_passing_primary_acceptance": passing_n,
            "n_values_passing_count": len(passing_n),
            "scale_support_passed": scale_support,
            "all_three_campaigns_present": all_campaigns,
            "wrong_codebook_actions_zero": wrong_zero,
            "stage6_external_value_supported": stage6_supported,
            "post_final_tuning_on_this_final_permitted": False,
        },
        "checks": {
            "access_count": 1,
            "all_registered_n_retained": raw["checks"][
                "all_registered_n_retained"
            ],
            "all_methods_used_identical_scene_catalog": raw["checks"][
                "all_methods_used_identical_scene_catalog"
            ],
            "frozen_snapshot_verified": True,
            "raw_result_hash_verified": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    updated_state = {
        **state,
        "status": "access_consumed_final_analysis_complete",
        "final_method_outputs_accessed": True,
        "analysis_sha256": sha256_file(args.output),
        "stage6_external_value_supported": stage6_supported,
    }
    atomic_write_json(args.state, updated_state)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stage6_external_value_supported": stage6_supported,
                "primary_configuration_passed": bool(primary_pass),
                "n_values_passing_primary_acceptance": passing_n,
                "post_final_tuning_permitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
