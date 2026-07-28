#!/usr/bin/env python
"""Analyze Stage-6 epsilon and task-scale sensitivity at matched clean."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from analyze_stage6_grouped_validation import (  # noqa: E402
    _bootstrap_plan,
    _leave_one_site_out,
    _metric_estimate,
    _paired_savings,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage6_task_scale_sensitivity import (  # noqa: E402
    codebook_install_break_even_scenes,
    registered_grid_points,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_task_scale_sensitivity_v1.json"
)
DEFAULT_RESULT = (
    PROJECT_DIR / "results/stage6/task_scale_sensitivity_grouped_v1/result.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6/task_scale_sensitivity_analysis_v1/analysis.json"
)


def _adjust_rows(rows: list[dict], metric: str, addition: float) -> list[dict]:
    adjusted = copy.deepcopy(rows)
    for row in adjusted:
        for trajectory in row["trajectory_metrics"]:
            trajectory[metric] = float(trajectory[metric] + addition)
        row["summary"][metric] = float(row["summary"][metric] + addition)
    return adjusted


def _exact_row(group: dict, attempts: int) -> dict:
    matches = [
        value
        for value in group["exact_workpoints"]
        if int(value["task_packet_open_loop_attempts"]) == int(attempts)
    ]
    if len(matches) != 1:
        raise ValueError("exact sensitivity workpoint is not unique")
    return matches[0] | {
        "evaluated_scene_count": group["evaluated_scene_count"]
    }


def _semantic_row(group: dict) -> dict:
    return group["semantic_workpoint"] | {
        "evaluated_scene_count": group["evaluated_scene_count"]
    }


def _reference_comparison(r1_analysis: dict, n_channels: int) -> dict:
    matches = [
        value
        for value in r1_analysis["analysis_by_n"][str(n_channels)][
            "comparisons"
        ]
        if value["variant_id"] == "v2_durable_codebook_compact_refresh"
        and value["deployment_mode"]
        == "preconfigured_codebook_empty_action_state"
        and int(value["requested_session_scene_count"]) == 500
        and abs(float(value["target_clean_rate"]) - 0.9) < 1e-12
    ]
    if len(matches) != 1:
        raise ValueError("frozen R1 reference comparison is not unique")
    return matches[0]


def analyze(protocol: dict, result: dict, r1_analysis: dict) -> dict:
    system = protocol["system_evaluation"]
    replicates = int(system["paired_bootstrap_replicates"])
    confidence = float(system["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    points = registered_grid_points(protocol)
    analysis_by_n: dict[str, dict] = {}
    accepted_total = 0
    strong_total = 0
    reproduction_checks = []

    for n_position, n_channels in enumerate(protocol["task_grid"]["n_channels"]):
        n_value = int(n_channels)
        n_key = str(n_value)
        point_results = []
        n_points = [value for value in points if value.n_channels == n_value]
        for point_position, point in enumerate(n_points):
            raw = result["results"][n_key][point.grid_point_id]
            representation = raw["representation"]
            base = {
                "grid_point_id": point.grid_point_id,
                "n_channels": n_value,
                "epsilon_db": point.epsilon_db,
                "query_set_id": point.query_set_id,
                "query_count": len(point.demands),
                "demands": list(point.demands),
                "representation": representation,
            }
            if raw["system_status"] != "evaluated":
                point_results.append(
                    base
                    | {
                        "status": "not_deployable_in_current_codec",
                        "matched_clean_comparisons": [],
                    }
                )
                continue
            group_ids = [
                group_id
                for group_id in raw["configured_outer_group_ids"]
                if raw["outer_groups"][group_id]["status"] == "evaluated"
            ]
            groups = [raw["outer_groups"][value] for value in group_ids]
            sites = [value["site_id"] for value in groups]
            semantic_rows = [_semantic_row(value) for value in groups]
            trajectory_count = len(semantic_rows[0]["trajectory_metrics"])
            seed = (
                int(system["random_seed"])
                + 70_000_000
                + n_position * 1_000_000
                + point_position * 10_000
            )
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
                estimate, _, _, _ = _metric_estimate(
                    semantic_rows,
                    metric,
                    group_draws,
                    trajectory_draws,
                    alpha,
                )
                semantic_estimates[metric] = estimate
            exact_by_attempts = {}
            for attempts in system["exact_reference_open_loop_attempts"]:
                rows = [_exact_row(value, int(attempts)) for value in groups]
                clean, _, _, _ = _metric_estimate(
                    rows,
                    "clean_rate",
                    group_draws,
                    trajectory_draws,
                    alpha,
                )
                bits, _, _, _ = _metric_estimate(
                    rows,
                    "actual_bits_per_scene",
                    group_draws,
                    trajectory_draws,
                    alpha,
                )
                exact_by_attempts[int(attempts)] = {
                    "rows": rows,
                    "clean_rate": clean,
                    "bits_per_scene": bits,
                }
            comparisons = []
            for target_position, target_value in enumerate(
                system["matched_clean_targets"]
            ):
                target = float(target_value)
                eligible = [
                    (attempts, value)
                    for attempts, value in exact_by_attempts.items()
                    if value["clean_rate"]["confidence_interval_lower"]
                    >= target
                ]
                comparison = {
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
                        comparison
                        | {
                            "status": "no_common_eligible_workpoint",
                            "eligible_exact_attempts": [
                                value[0] for value in eligible
                            ],
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
                leave_one_site = _leave_one_site_out(
                    group_ids,
                    sites,
                    semantic_rows,
                    exact["rows"],
                    replicates=replicates,
                    seed=seed + 1_000 + target_position * 100,
                    alpha=alpha,
                )
                direction_stable = all(
                    value["resource_savings_percentage"]["point_estimate"]
                    > 0.0
                    for value in leave_one_site
                )
                accepted = (
                    semantic_estimates["clean_rate"][
                        "confidence_interval_lower"
                    ]
                    >= 0.9
                    and savings["confidence_interval_lower"] > 0.0
                    and direction_stable
                )
                strong = accepted and savings["point_estimate"] >= 15.0
                accepted_total += int(accepted)
                strong_total += int(strong)
                amortized = {}
                install_bits = int(representation["context_install_bits"])
                for horizon in system["amortized_codebook_horizons_scenes"]:
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
                break_even = codebook_install_break_even_scenes(
                    semantic_estimates["actual_bits_per_scene"]["point_estimate"],
                    exact["bits_per_scene"]["point_estimate"],
                    install_bits,
                )
                comparisons.append(
                    comparison
                    | {
                        "status": "matched",
                        "selected_exact_attempts": attempts,
                        "exact_clean_rate": exact["clean_rate"],
                        "exact_bits_per_scene": exact["bits_per_scene"],
                        "savings_percentage": savings,
                        "amortized": amortized,
                        "codebook_install_break_even_scenes": break_even,
                        "leave_one_site_out": leave_one_site,
                        "leave_one_site_out_direction_stable": direction_stable,
                        "passes_primary_acceptance": accepted,
                        "passes_strong_effect": strong,
                    }
                )
            point_analysis = base | {
                "status": "analyzed",
                "outer_group_count": len(group_ids),
                "site_ids": sorted(set(sites)),
                "semantic_metrics": semantic_estimates,
                "matched_clean_comparisons": comparisons,
            }
            point_results.append(point_analysis)
            primary = protocol["decision_rules"][
                "primary_reference_configuration"
            ]
            if (
                abs(point.epsilon_db - float(primary["epsilon_db"])) < 1e-12
                and point.query_set_id == primary["query_set_id"]
            ):
                current = [
                    value
                    for value in comparisons
                    if abs(value["target_clean_rate"] - 0.9) < 1e-12
                ][0]
                frozen = _reference_comparison(r1_analysis, n_value)
                deltas = {
                    "clean_rate": float(
                        current["semantic_clean_rate"]["point_estimate"]
                        - frozen["r1_clean_rate"]["point_estimate"]
                    ),
                    "bits_per_scene": float(
                        current["semantic_bits_per_scene"]["point_estimate"]
                        - frozen["r1_bits_per_scene"]["point_estimate"]
                    ),
                    "savings_percentage": float(
                        current["savings_percentage"]["point_estimate"]
                        - frozen["savings_percentage"]["point_estimate"]
                    ),
                }
                reproduction_checks.append(
                    {
                        "n_channels": n_value,
                        "deltas": deltas,
                        "passes_deterministic_tolerance": all(
                            abs(value) <= 1e-12 for value in deltas.values()
                        ),
                    }
                )
        analysis_by_n[n_key] = {"grid_points": point_results}

    broad_support = {}
    for n_channels in protocol["task_grid"]["n_channels"]:
        points_for_n = analysis_by_n[str(int(n_channels))]["grid_points"]
        supported_epsilons = []
        for value in points_for_n:
            if value["query_set_id"] != "multi_025_050_075":
                continue
            comparison = [
                item
                for item in value["matched_clean_comparisons"]
                if abs(item["target_clean_rate"] - 0.9) < 1e-12
            ]
            if comparison and comparison[0].get("passes_primary_acceptance", False):
                supported_epsilons.append(value["epsilon_db"])
        broad_support[str(int(n_channels))] = supported_epsilons
    n_meeting_broad_rule = sum(
        len(values) >= 3 for values in broad_support.values()
    )
    query_count_profile = {}
    for query_count in (1, 2, 3):
        comparisons = []
        for n_value in analysis_by_n.values():
            for point in n_value["grid_points"]:
                if point["query_count"] != query_count:
                    continue
                comparisons.extend(
                    item
                    for item in point["matched_clean_comparisons"]
                    if item["status"] == "matched"
                    and abs(item["target_clean_rate"] - 0.9) < 1e-12
                )
        query_count_profile[str(query_count)] = {
            "matched_comparison_count": len(comparisons),
            "accepted_comparison_count": sum(
                value.get("passes_primary_acceptance", False)
                for value in comparisons
            ),
            "mean_savings_point_estimate_percentage": (
                float(
                    np.mean(
                        [
                            value["savings_percentage"]["point_estimate"]
                            for value in comparisons
                        ]
                    )
                )
                if comparisons
                else None
            ),
        }
    return {
        "version": "1.0",
        "status": "task_scale_sensitivity_analysis_complete",
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence,
            "outer_unit": "site_x_local_date",
            "inner_unit": "paired_link_trajectory",
        },
        "analysis_by_n": analysis_by_n,
        "decision_summary": {
            "accepted_comparison_count": accepted_total,
            "strong_effect_comparison_count": strong_total,
            "multi_query_supported_epsilon_by_n": broad_support,
            "n_values_meeting_broad_rate_risk_rule": n_meeting_broad_rule,
            "broad_rate_risk_support_passed": n_meeting_broad_rule >= 3,
            "query_count_profile_at_target_clean_0_9": query_count_profile,
            "primary_reproduction_checks": reproduction_checks,
            "primary_reproduction_passed": all(
                value["passes_deterministic_tolerance"]
                for value in reproduction_checks
            )
            and len(reproduction_checks) == 4,
        },
        "checks": {
            "all_grid_points_retained": result["checks"][
                "all_grid_points_retained"
            ],
            "wrong_codebook_actions_must_be_zero": result["checks"][
                "wrong_codebook_actions_must_be_zero"
            ],
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != sha256_file(args.protocol):
        raise ValueError("sensitivity result does not match protocol")
    r1_entry = protocol["frozen_inputs"]["r1_grouped_analysis"]
    r1_analysis = json.loads(
        (PROJECT_DIR / r1_entry["path"]).read_text(encoding="utf-8")
    )
    value = analyze(protocol, result, r1_analysis)
    value["protocol_sha256"] = sha256_file(args.protocol)
    value["grouped_result_sha256"] = sha256_file(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, value)


if __name__ == "__main__":
    main()
