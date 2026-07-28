#!/usr/bin/env python
"""Analyze frozen Stage-6 grouped validation without retuning candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


DEFAULT_CANDIDATES = (
    PROJECT_DIR / "configs/stage6_grouped_validation_candidates_v1.json"
)
DEFAULT_GROUPED = (
    PROJECT_DIR
    / "results/stage6/matched_reliability_grouped_validation_v1/result.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6/matched_reliability_grouped_analysis_v1/analysis.json"
)


def _row_for(
    group: dict,
    *,
    method: str,
    scene_count: int,
    candidate_id: str | None = None,
    deployment_mode: str | None = None,
    attempts: int | None = None,
) -> dict:
    rows = group[
        "semantic_workpoints" if method == "semantic" else "exact_workpoints"
    ]
    matches = []
    for row in rows:
        if int(row["requested_session_scene_count"]) != int(scene_count):
            continue
        if method == "semantic":
            if (
                row["candidate_id"] == candidate_id
                and row["deployment_mode"] == deployment_mode
            ):
                matches.append(row)
        elif int(row["task_packet_open_loop_attempts"]) == int(attempts):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} row, found {len(matches)}"
        )
    return matches[0]


def _metric_matrix(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray(
        [
            [float(item[metric]) for item in row["trajectory_metrics"]]
            for row in rows
        ],
        dtype=np.float64,
    )


def _scene_weights(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [float(row["evaluated_scene_count"]) for row in rows],
        dtype=np.float64,
    )


def _point(matrix: np.ndarray, scene_weights: np.ndarray) -> float:
    group_means = np.mean(matrix, axis=1)
    return float(np.sum(group_means * scene_weights) / np.sum(scene_weights))


def _bootstrap_plan(
    group_count: int,
    trajectory_count: int,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, group_count, size=(replicates, group_count))
    trajectories = rng.integers(
        0, trajectory_count, size=(replicates, group_count)
    )
    return groups, trajectories


def _bootstrap_metric(
    matrix: np.ndarray,
    scene_weights: np.ndarray,
    group_draws: np.ndarray,
    trajectory_draws: np.ndarray,
) -> np.ndarray:
    sampled_values = matrix[group_draws, trajectory_draws]
    sampled_weights = scene_weights[group_draws]
    return np.sum(sampled_values * sampled_weights, axis=1) / np.sum(
        sampled_weights, axis=1
    )


def _interval(
    point: float,
    bootstrap: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    return {
        "point_estimate": float(point),
        "confidence_interval_lower": float(np.quantile(bootstrap, alpha)),
        "confidence_interval_upper": float(
            np.quantile(bootstrap, 1.0 - alpha)
        ),
    }


def _metric_estimate(
    rows: list[dict],
    metric: str,
    group_draws: np.ndarray,
    trajectory_draws: np.ndarray,
    alpha: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    matrix = _metric_matrix(rows, metric)
    weights = _scene_weights(rows)
    bootstrap = _bootstrap_metric(
        matrix, weights, group_draws, trajectory_draws
    )
    return _interval(_point(matrix, weights), bootstrap, alpha), bootstrap, matrix, weights


def _paired_savings(
    semantic_rows: list[dict],
    exact_rows: list[dict],
    metric: str,
    group_draws: np.ndarray,
    trajectory_draws: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    semantic = _metric_matrix(semantic_rows, metric)
    exact = _metric_matrix(exact_rows, metric)
    semantic_weights = _scene_weights(semantic_rows)
    exact_weights = _scene_weights(exact_rows)
    if not np.array_equal(semantic_weights, exact_weights):
        raise ValueError("paired methods evaluated different scene counts")
    semantic_point = _point(semantic, semantic_weights)
    exact_point = _point(exact, exact_weights)
    point = 100.0 * (exact_point - semantic_point) / exact_point
    semantic_boot = _bootstrap_metric(
        semantic, semantic_weights, group_draws, trajectory_draws
    )
    exact_boot = _bootstrap_metric(
        exact, exact_weights, group_draws, trajectory_draws
    )
    bootstrap = 100.0 * (exact_boot - semantic_boot) / exact_boot
    return _interval(point, bootstrap, alpha)


def _group_profile(
    group_ids: list[str],
    sites: list[str],
    semantic_rows: list[dict],
    exact_rows: list[dict],
) -> list[dict]:
    profile = []
    for group_id, site, semantic, exact in zip(
        group_ids, sites, semantic_rows, exact_rows
    ):
        semantic_bits = semantic["summary"][
            "resource_equivalent_bits_per_scene"
        ]
        exact_bits = exact["summary"]["resource_equivalent_bits_per_scene"]
        profile.append(
            {
                "outer_group_id": group_id,
                "site_id": site,
                "evaluated_scene_count": semantic[
                    "evaluated_scene_count"
                ],
                "semantic_clean_rate": semantic["summary"]["clean_rate"],
                "exact_clean_rate": exact["summary"]["clean_rate"],
                "semantic_resource_bits_per_scene": semantic_bits,
                "exact_resource_bits_per_scene": exact_bits,
                "resource_savings_percentage": float(
                    100.0 * (exact_bits - semantic_bits) / exact_bits
                ),
            }
        )
    return profile


def _leave_one_site_out(
    group_ids: list[str],
    sites: list[str],
    semantic_rows: list[dict],
    exact_rows: list[dict],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> list[dict]:
    values = []
    for site_position, excluded_site in enumerate(sorted(set(sites))):
        keep = [
            index
            for index, site in enumerate(sites)
            if site != excluded_site
        ]
        kept_semantic = [semantic_rows[index] for index in keep]
        kept_exact = [exact_rows[index] for index in keep]
        trajectory_count = len(
            kept_semantic[0]["trajectory_metrics"]
        )
        group_draws, trajectory_draws = _bootstrap_plan(
            len(keep),
            trajectory_count,
            replicates,
            seed + site_position,
        )
        savings = _paired_savings(
            kept_semantic,
            kept_exact,
            "resource_equivalent_bits_per_scene",
            group_draws,
            trajectory_draws,
            alpha,
        )
        semantic_clean, _, _, _ = _metric_estimate(
            kept_semantic,
            "clean_rate",
            group_draws,
            trajectory_draws,
            alpha,
        )
        exact_clean, _, _, _ = _metric_estimate(
            kept_exact,
            "clean_rate",
            group_draws,
            trajectory_draws,
            alpha,
        )
        values.append(
            {
                "excluded_site_id": excluded_site,
                "retained_outer_group_count": len(keep),
                "retained_outer_group_ids": [
                    group_ids[index] for index in keep
                ],
                "semantic_clean_rate": semantic_clean,
                "exact_clean_rate": exact_clean,
                "resource_savings_percentage": savings,
            }
        )
    return values


def analyze(candidates: dict, grouped: dict) -> dict:
    evaluation = candidates["evaluation_conditions"]
    targets = [
        float(value) for value in evaluation["candidate_target_clean_rates"]
    ]
    replicates = int(evaluation["paired_bootstrap_replicates"])
    confidence = float(evaluation["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    analysis_by_n = {}
    supported_primary = 0

    for n_position, n_key in enumerate(sorted(grouped["results"], key=int)):
        value = grouped["results"][n_key]
        group_ids = [
            key
            for key in value["configured_outer_group_ids"]
            if value["outer_groups"][key].get("status", "evaluated")
            == "evaluated"
        ]
        excluded_group_ids = [
            key
            for key in value["configured_outer_group_ids"]
            if value["outer_groups"][key].get("status", "evaluated")
            != "evaluated"
        ]
        groups = [value["outer_groups"][key] for key in group_ids]
        sites = [group["site_id"] for group in groups]
        trajectory_count = int(value["trajectory_count"])
        n_comparisons = []
        for length_position, scene_count in enumerate(
            evaluation["requested_session_scene_counts"]
        ):
            seed = (
                20260728
                + 20_000_000
                + n_position * 100_000
                + length_position * 10_000
            )
            group_draws, trajectory_draws = _bootstrap_plan(
                len(groups), trajectory_count, replicates, seed
            )
            exact_by_attempts = {}
            for attempts in evaluation[
                "exact_task_packet_open_loop_attempts"
            ]:
                rows = [
                    _row_for(
                        group,
                        method="exact",
                        scene_count=int(scene_count),
                        attempts=int(attempts),
                    )
                    for group in groups
                ]
                clean, clean_boot, _, _ = _metric_estimate(
                    rows,
                    "clean_rate",
                    group_draws,
                    trajectory_draws,
                    alpha,
                )
                resource, _, _, _ = _metric_estimate(
                    rows,
                    "resource_equivalent_bits_per_scene",
                    group_draws,
                    trajectory_draws,
                    alpha,
                )
                exact_by_attempts[int(attempts)] = {
                    "rows": rows,
                    "clean_rate": clean,
                    "clean_lower_bound": float(
                        np.quantile(clean_boot, alpha)
                    ),
                    "resource_bits_per_scene": resource,
                }
            for deployment_mode in evaluation["deployment_modes"]:
                for candidate_position, candidate in enumerate(
                    candidates["controller_candidates"][n_key]
                ):
                    semantic_rows = [
                        _row_for(
                            group,
                            method="semantic",
                            scene_count=int(scene_count),
                            candidate_id=candidate["candidate_id"],
                            deployment_mode=deployment_mode,
                        )
                        for group in groups
                    ]
                    semantic_clean, semantic_clean_boot, _, _ = (
                        _metric_estimate(
                            semantic_rows,
                            "clean_rate",
                            group_draws,
                            trajectory_draws,
                            alpha,
                        )
                    )
                    semantic_clean_lower = float(
                        np.quantile(semantic_clean_boot, alpha)
                    )
                    semantic_resource, _, _, _ = _metric_estimate(
                        semantic_rows,
                        "resource_equivalent_bits_per_scene",
                        group_draws,
                        trajectory_draws,
                        alpha,
                    )
                    semantic_actual, _, _, _ = _metric_estimate(
                        semantic_rows,
                        "actual_bits_per_scene",
                        group_draws,
                        trajectory_draws,
                        alpha,
                    )
                    for target in targets:
                        eligible_exact = [
                            (attempts, item)
                            for attempts, item in exact_by_attempts.items()
                            if item["clean_lower_bound"] >= target
                        ]
                        base = {
                            "candidate_id": candidate["candidate_id"],
                            "candidate_rank_from_coarse_grid": (
                                candidate_position + 1
                            ),
                            "is_primary_frozen_candidate": (
                                candidate_position == 0
                            ),
                            "deployment_mode": deployment_mode,
                            "requested_session_scene_count": int(scene_count),
                            "target_clean_rate": target,
                            "semantic_clean_rate": semantic_clean,
                            "semantic_resource_bits_per_scene": (
                                semantic_resource
                            ),
                            "semantic_actual_bits_per_scene": semantic_actual,
                        }
                        if (
                            semantic_clean_lower < target
                            or not eligible_exact
                        ):
                            n_comparisons.append(
                                base
                                | {
                                    "status": (
                                        "no_common_eligible_workpoint"
                                    ),
                                    "semantic_reaches_target": (
                                        semantic_clean_lower >= target
                                    ),
                                    "eligible_exact_attempts": [
                                        attempts
                                        for attempts, _ in eligible_exact
                                    ],
                                }
                            )
                            continue
                        selected_attempts, selected_exact = min(
                            eligible_exact,
                            key=lambda item: (
                                item[1]["resource_bits_per_scene"][
                                    "point_estimate"
                                ],
                                item[0],
                            ),
                        )
                        exact_rows = selected_exact["rows"]
                        resource_savings = _paired_savings(
                            semantic_rows,
                            exact_rows,
                            "resource_equivalent_bits_per_scene",
                            group_draws,
                            trajectory_draws,
                            alpha,
                        )
                        actual_savings = _paired_savings(
                            semantic_rows,
                            exact_rows,
                            "actual_bits_per_scene",
                            group_draws,
                            trajectory_draws,
                            alpha,
                        )
                        direction_agrees = (
                            resource_savings["point_estimate"] > 0.0
                            and actual_savings["point_estimate"] > 0.0
                        ) or (
                            resource_savings["point_estimate"] <= 0.0
                            and actual_savings["point_estimate"] <= 0.0
                        )
                        supported = (
                            resource_savings[
                                "confidence_interval_lower"
                            ]
                            > 0.0
                            and actual_savings[
                                "confidence_interval_lower"
                            ]
                            > 0.0
                            and direction_agrees
                        )
                        comparison = base | {
                            "status": "matched",
                            "selected_exact_attempts": selected_attempts,
                            "exact_clean_rate": selected_exact["clean_rate"],
                            "exact_resource_bits_per_scene": selected_exact[
                                "resource_bits_per_scene"
                            ],
                            "resource_savings_percentage": resource_savings,
                            "actual_savings_percentage": actual_savings,
                            "actual_and_resource_direction_agree": (
                                direction_agrees
                            ),
                            "positive_savings_supported": supported,
                            "per_site_date_profile": _group_profile(
                                group_ids,
                                sites,
                                semantic_rows,
                                exact_rows,
                            ),
                            "leave_one_site_out": _leave_one_site_out(
                                group_ids,
                                sites,
                                semantic_rows,
                                exact_rows,
                                replicates=replicates,
                                seed=seed
                                + 1_000
                                + candidate_position * 10,
                                alpha=alpha,
                            ),
                        }
                        n_comparisons.append(comparison)
                        if (
                            comparison["is_primary_frozen_candidate"]
                            and supported
                        ):
                            supported_primary += 1
        analysis_by_n[n_key] = {
            "outer_group_count": len(groups),
            "excluded_outer_group_ids": excluded_group_ids,
            "site_ids": sorted(set(sites)),
            "trajectory_count": trajectory_count,
            "comparisons": n_comparisons,
            "matched_comparison_count": sum(
                row["status"] == "matched" for row in n_comparisons
            ),
            "supported_comparison_count": sum(
                row.get("positive_savings_supported", False)
                for row in n_comparisons
            ),
            "supported_primary_candidate_comparison_count": sum(
                row.get("positive_savings_supported", False)
                and row["is_primary_frozen_candidate"]
                for row in n_comparisons
            ),
        }
    return {
        "version": "1.0",
        "status": "grouped_validation_analysis_complete",
        "grouped_validation_sha256": None,
        "candidate_protocol_sha256": grouped[
            "candidate_protocol_sha256"
        ],
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence,
            "outer_unit": "site_x_local_date",
            "inner_unit": "paired_link_trajectory",
            "aggregation": "scene_weighted_hierarchical_cluster_bootstrap",
        },
        "analysis_by_n": analysis_by_n,
        "checks": {
            "controller_candidates_were_not_reselected": True,
            "primary_candidate_is_coarse_rank_one": True,
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "decision_summary": {
            "supported_primary_candidate_comparison_count": (
                supported_primary
            ),
            "primary_development_success": supported_primary > 0,
        },
        "claim_boundary": (
            "Hierarchical grouped analysis of historically accessed 2022 "
            "development data. Confidence intervals include site/date and "
            "paired link-trajectory variation; they are not external Final."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES
    )
    parser.add_argument(
        "--grouped-validation", type=Path, default=DEFAULT_GROUPED
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = args.candidates.resolve()
    grouped_path = args.grouped_validation.resolve()
    output_path = args.output.resolve()
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    candidate_sha = sha256_file(candidate_path)
    grouped_sha = sha256_file(grouped_path)
    if (
        grouped["status"] != "grouped_validation_complete"
        or grouped["candidate_protocol_sha256"] != candidate_sha
        or not grouped["checks"]["external_final_access_count_remains_zero"]
    ):
        raise ValueError("grouped result is incomplete or incompatible")
    result = analyze(candidates, grouped)
    result["grouped_validation_sha256"] = grouped_sha
    if output_path.exists():
        raise FileExistsError("refusing to overwrite grouped analysis")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    print(output_path)


if __name__ == "__main__":
    main()
