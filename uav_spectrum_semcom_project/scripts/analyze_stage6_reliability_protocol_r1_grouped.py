#!/usr/bin/env python
"""Analyze grouped Stage-6 R1 protocols against frozen exact references."""

from __future__ import annotations

import argparse
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


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_reliability_protocol_diagnostic_v1.json"
)


def _verified_json(entry: dict, label: str) -> dict:
    path = PROJECT_DIR / entry["path"]
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def _r1_row(
    group: dict,
    *,
    variant: str,
    deployment: str,
    length: int,
) -> dict:
    matches = [
        row
        for row in group["variant_workpoints"]
        if row["variant_id"] == variant
        and row["deployment_mode"] == deployment
        and int(row["requested_session_scene_count"]) == int(length)
    ]
    if len(matches) != 1:
        raise ValueError("R1 workpoint identity is not unique")
    return matches[0]


def _exact_row(group: dict, *, attempts: int, length: int) -> dict:
    matches = [
        row
        for row in group["exact_workpoints"]
        if int(row["task_packet_open_loop_attempts"]) == int(attempts)
        and int(row["requested_session_scene_count"]) == int(length)
    ]
    if len(matches) != 1:
        raise ValueError("exact workpoint identity is not unique")
    return matches[0]


def _per_group_profile(
    group_ids: list[str],
    sites: list[str],
    r1_rows: list[dict],
    exact_rows: list[dict],
) -> list[dict]:
    result = []
    for group_id, site, r1, exact in zip(
        group_ids, sites, r1_rows, exact_rows
    ):
        r1_bits = r1["summary"]["actual_bits_per_scene"]
        exact_bits = exact["summary"]["actual_bits_per_scene"]
        result.append(
            {
                "outer_group_id": group_id,
                "site_id": site,
                "r1_clean_rate": r1["summary"]["clean_rate"],
                "exact_clean_rate": exact["summary"]["clean_rate"],
                "r1_bits_per_scene": r1_bits,
                "exact_bits_per_scene": exact_bits,
                "savings_percentage": float(
                    100.0 * (exact_bits - r1_bits) / exact_bits
                ),
            }
        )
    return result


def analyze(protocol: dict, r1: dict, frozen: dict) -> dict:
    scope = protocol["evaluation_scope"]
    acceptance = protocol["primary_acceptance"]
    targets = [float(value) for value in scope["matched_clean_targets"]]
    replicates = int(scope["paired_bootstrap_replicates"])
    confidence = float(scope["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    variants = [
        item["variant_id"]
        for item in protocol["r1_variants"]
        if item["variant_id"].startswith(("v2_", "v3_", "v4_"))
    ]
    analysis_by_n = {}
    accepted_counts = {variant: 0 for variant in variants}
    strong_counts = {variant: 0 for variant in variants}

    for n_position, n_channels in enumerate(scope["n_channels"]):
        n_key = str(int(n_channels))
        r1_value = r1["results"][n_key]
        frozen_value = frozen["results"][n_key]
        group_ids = [
            key
            for key in r1_value["configured_outer_group_ids"]
            if r1_value["outer_groups"][key].get("status", "evaluated")
            == "evaluated"
        ]
        r1_groups = [r1_value["outer_groups"][key] for key in group_ids]
        exact_groups = [
            frozen_value["outer_groups"][key] for key in group_ids
        ]
        sites = [group["site_id"] for group in r1_groups]
        comparisons = []
        for length_position, length in enumerate(
            scope["session_scene_counts"]
        ):
            seed = (
                20260728
                + 40_000_000
                + n_position * 100_000
                + length_position * 10_000
            )
            group_draws, trajectory_draws = _bootstrap_plan(
                len(group_ids),
                int(scope["link_trajectories"]),
                replicates,
                seed,
            )
            exact_by_attempts = {}
            for attempts in (1, 2, 3):
                rows = [
                    _exact_row(
                        group, attempts=attempts, length=int(length)
                    )
                    for group in exact_groups
                ]
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
                exact_by_attempts[attempts] = {
                    "rows": rows,
                    "clean_rate": clean,
                    "bits_per_scene": bits,
                }
            for deployment in scope["deployment_modes"]:
                for variant_position, variant in enumerate(variants):
                    rows = [
                        _r1_row(
                            group,
                            variant=variant,
                            deployment=deployment,
                            length=int(length),
                        )
                        for group in r1_groups
                    ]
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
                    component_metrics = {}
                    for metric in (
                        "forward_bits_per_scene",
                        "feedback_bits_per_scene",
                        "task_frame_bits_per_scene",
                        "initial_install_bits_per_scene",
                        "recovery_install_bits_per_scene",
                        "heartbeat_bits_per_scene",
                        "compact_refresh_count",
                        "exact_refresh_count",
                        "actual_action_reset_count",
                        "actual_codebook_reset_count",
                        "missing_update_ack_uncertainty_count",
                        "failed_heartbeat_uncertainty_count",
                    ):
                        estimate, _, _, _ = _metric_estimate(
                            rows,
                            metric,
                            group_draws,
                            trajectory_draws,
                            alpha,
                        )
                        component_metrics[metric] = estimate
                    for target in targets:
                        eligible_exact = [
                            (attempts, item)
                            for attempts, item in exact_by_attempts.items()
                            if item["clean_rate"][
                                "confidence_interval_lower"
                            ]
                            >= target
                        ]
                        base = {
                            "variant_id": variant,
                            "deployment_mode": deployment,
                            "requested_session_scene_count": int(length),
                            "target_clean_rate": target,
                            "r1_clean_rate": clean,
                            "r1_bits_per_scene": bits,
                            "components": component_metrics,
                        }
                        if (
                            clean["confidence_interval_lower"] < target
                            or not eligible_exact
                        ):
                            comparisons.append(
                                base
                                | {
                                    "status": (
                                        "no_common_eligible_workpoint"
                                    ),
                                    "eligible_exact_attempts": [
                                        item[0] for item in eligible_exact
                                    ],
                                }
                            )
                            continue
                        exact_attempts, exact = min(
                            eligible_exact,
                            key=lambda item: (
                                item[1]["bits_per_scene"][
                                    "point_estimate"
                                ],
                                item[0],
                            ),
                        )
                        savings = _paired_savings(
                            rows,
                            exact["rows"],
                            "actual_bits_per_scene",
                            group_draws,
                            trajectory_draws,
                            alpha,
                        )
                        leave_one_site_out = _leave_one_site_out(
                            group_ids,
                            sites,
                            rows,
                            exact["rows"],
                            replicates=replicates,
                            seed=seed + 1_000 + variant_position * 10,
                            alpha=alpha,
                        )
                        leave_one_site_direction_stable = all(
                            item["resource_savings_percentage"][
                                "point_estimate"
                            ]
                            > 0.0
                            for item in leave_one_site_out
                        )
                        accepted = (
                            clean["confidence_interval_lower"]
                            >= float(
                                acceptance[
                                    "clean_lower_bound_minimum"
                                ]
                            )
                            and savings["confidence_interval_lower"]
                            > float(
                                acceptance[
                                    "actual_savings_lower_bound_percentage_must_exceed"
                                ]
                            )
                            and leave_one_site_direction_stable
                        )
                        strong = (
                            accepted
                            and savings["point_estimate"]
                            >= float(
                                acceptance[
                                    "strong_effect_point_savings_percentage"
                                ]
                            )
                        )
                        comparisons.append(
                            base
                            | {
                                "status": "matched",
                                "selected_exact_attempts": exact_attempts,
                                "exact_clean_rate": exact["clean_rate"],
                                "exact_bits_per_scene": exact[
                                    "bits_per_scene"
                                ],
                                "savings_percentage": savings,
                                "leave_one_site_out": leave_one_site_out,
                                "leave_one_site_out_direction_stable": (
                                    leave_one_site_direction_stable
                                ),
                                "passes_primary_acceptance": accepted,
                                "passes_strong_effect": strong,
                                "per_site_date_profile": _per_group_profile(
                                    group_ids,
                                    sites,
                                    rows,
                                    exact["rows"],
                                ),
                            }
                        )
                        accepted_counts[variant] += int(accepted)
                        strong_counts[variant] += int(strong)
        analysis_by_n[n_key] = {
            "outer_group_count": len(group_ids),
            "site_ids": sorted(set(sites)),
            "comparisons": comparisons,
            "accepted_comparison_count_by_variant": {
                variant: sum(
                    row.get("passes_primary_acceptance", False)
                    and row["variant_id"] == variant
                    for row in comparisons
                )
                for variant in variants
            },
        }
    v2 = "v2_durable_codebook_compact_refresh"
    v3 = "v3_context_independent_exact_refresh"
    v4 = "v4_event_triggered_exact_action_cache"
    if accepted_counts[v2] > 0 and accepted_counts[v4] == 0:
        decision = (
            "retain_task_semantic_codebook_as_primary_protocol_contribution"
        )
    elif accepted_counts[v2] > 0 and accepted_counts[v4] > 0:
        decision = (
            "narrow_claim_to_task_trigger_and_reliable_state_reuse_with_"
            "codebook_as_payload_optimization"
        )
    elif accepted_counts[v3] > 0 or accepted_counts[v4] > 0:
        decision = (
            "do_not_claim_existing_codebook_is_source_of_system_savings"
        )
    else:
        decision = (
            "stop_before_external_final_and_reframe_rate_reliability_tradeoff"
        )
    return {
        "version": "1.0",
        "status": "r1_grouped_analysis_complete",
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence,
            "outer_unit": "site_x_local_date",
            "inner_unit": "paired_link_trajectory",
        },
        "analysis_by_n": analysis_by_n,
        "decision_summary": {
            "accepted_comparison_count_by_variant": accepted_counts,
            "strong_effect_comparison_count_by_variant": strong_counts,
            "protocol_decision": decision,
        },
        "checks": {
            "wrong_codebook_actions_must_be_zero": r1["checks"][
                "wrong_codebook_actions_must_be_zero"
            ],
            "negative_reference_retained": True,
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "claim_boundary": (
            "R1 grouped development analysis. Passing this gate permits "
            "sensitivity analysis, not an external Final claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--r1-result", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    r1_path = (
        args.r1_result.resolve()
        if args.r1_result is not None
        else PROJECT_DIR / protocol["outputs"]["r1_grouped"]
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["r1_analysis"]
    )
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))
    if (
        r1["status"] != "r1_grouped_validation_complete"
        or r1["protocol_sha256"] != sha256_file(protocol_path)
    ):
        raise ValueError("R1 result is incomplete or incompatible")
    frozen = _verified_json(
        protocol["frozen_inputs"]["grouped_result"],
        "frozen grouped result",
    )
    result = analyze(protocol, r1, frozen)
    result["protocol_sha256"] = sha256_file(protocol_path)
    result["r1_result_sha256"] = sha256_file(r1_path)
    if output.exists():
        raise FileExistsError("refusing to overwrite R1 analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
