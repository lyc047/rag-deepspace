#!/usr/bin/env python
"""Analyze S6.7b coarse-grid Pareto and matched-clean workpoints."""

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


DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "configs/stage6_matched_reliability_development_v1.json"
)


def _bootstrap_weights(
    trajectory_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        trajectory_count,
        size=(replicates, trajectory_count),
    )
    weights = np.zeros(
        (trajectory_count, replicates), dtype=np.float64
    )
    for replicate, row in enumerate(indices):
        weights[:, replicate] = (
            np.bincount(row, minlength=trajectory_count) / trajectory_count
        )
    return weights


def _matrix(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray(
        [
            [float(value[metric]) for value in row["trajectory_metrics"]]
            for row in rows
        ],
        dtype=np.float64,
    )


def _lower_bounds(
    rows: list[dict],
    metric: str,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    bootstrap = _matrix(rows, metric) @ weights
    return np.quantile(bootstrap, alpha, axis=1)


def _pareto_indices(rows: list[dict]) -> list[int]:
    clean = np.asarray(
        [row["summary"]["clean_rate"] for row in rows], dtype=np.float64
    )
    bits = np.asarray(
        [
            row["summary"]["resource_equivalent_bits_per_scene"]
            for row in rows
        ],
        dtype=np.float64,
    )
    keep = []
    for index in range(len(rows)):
        dominated = np.any(
            (bits <= bits[index] + 1e-12)
            & (clean >= clean[index] - 1e-12)
            & (
                (bits < bits[index] - 1e-12)
                | (clean > clean[index] + 1e-12)
            )
        )
        if not dominated:
            keep.append(index)
    return keep


def _interval(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    bootstrap = np.asarray(values, dtype=np.float64) @ weights
    return {
        "point_estimate": float(np.mean(values)),
        "confidence_interval_lower": float(
            np.quantile(bootstrap, alpha)
        ),
        "confidence_interval_upper": float(
            np.quantile(bootstrap, 1.0 - alpha)
        ),
    }


def _compact_workpoint(row: dict) -> dict:
    keys = (
        "workpoint_id",
        "heartbeat_silence_scenes",
        "maximum_state_age_minutes",
        "update_reservation_fraction",
        "task_update_open_loop_attempts",
        "deployment_mode",
        "requested_session_scene_count",
        "session_count",
        "evaluated_scene_count",
        "risk_threshold",
        "risk_candidate_count",
        "task_packet_open_loop_attempts",
        "packet_bits",
    )
    return {
        key: row[key] for key in keys if key in row
    } | {"summary": row["summary"]}


def _controller_signature(row: dict) -> tuple:
    return (
        row["heartbeat_silence_scenes"],
        float(row["maximum_state_age_minutes"]),
        float(row["update_reservation_fraction"]),
        int(row["task_update_open_loop_attempts"]),
    )


def analyze(protocol: dict, coarse: dict) -> dict:
    monte = protocol["monte_carlo"]
    targets = [
        float(value)
        for value in protocol["matched_clean"][
            "candidate_target_clean_rates"
        ]
    ]
    replicates = int(monte["paired_bootstrap_replicates"])
    confidence = float(monte["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    analysis_by_n = {}

    for n_position, n_channels in enumerate(
        protocol["task_grid"]["n_channels"]
    ):
        n_key = str(int(n_channels))
        value = coarse["results"][n_key]
        semantic = value["semantic_workpoints"]
        exact = value["exact_workpoints"]
        trajectory_count = len(semantic[0]["trajectory_metrics"])
        weights = _bootstrap_weights(
            trajectory_count,
            replicates,
            int(monte["seed"]) + 7_000_000 + n_position,
        )
        semantic_lower = _lower_bounds(
            semantic, "clean_rate", weights, alpha
        )
        exact_lower = _lower_bounds(exact, "clean_rate", weights, alpha)
        semantic_index = {
            row["workpoint_id"]: index for index, row in enumerate(semantic)
        }
        exact_index = {
            row["workpoint_id"]: index for index, row in enumerate(exact)
        }
        pareto = {}
        comparisons = []
        selected_pool = []
        for scene_count in protocol["session_grid"]["scene_counts"]:
            exact_rows = [
                row
                for row in exact
                if row["requested_session_scene_count"] == scene_count
            ]
            for deployment_mode in protocol["session_grid"][
                "deployment_modes"
            ]:
                semantic_rows = [
                    row
                    for row in semantic
                    if row["requested_session_scene_count"] == scene_count
                    and row["deployment_mode"] == deployment_mode
                ]
                pareto_rows = [
                    semantic_rows[index]
                    for index in _pareto_indices(semantic_rows)
                ]
                pareto[
                    f"{deployment_mode}|L={scene_count}"
                ] = [row["workpoint_id"] for row in pareto_rows]
                for target in targets:
                    eligible_semantic = [
                        row
                        for row in semantic_rows
                        if semantic_lower[
                            semantic_index[row["workpoint_id"]]
                        ]
                        >= target
                    ]
                    eligible_exact = [
                        row
                        for row in exact_rows
                        if exact_lower[
                            exact_index[row["workpoint_id"]]
                        ]
                        >= target
                    ]
                    if not eligible_semantic or not eligible_exact:
                        comparisons.append(
                            {
                                "target_clean_rate": target,
                                "deployment_mode": deployment_mode,
                                "requested_session_scene_count": scene_count,
                                "status": "no_common_eligible_workpoint",
                                "semantic_eligible_count": len(
                                    eligible_semantic
                                ),
                                "exact_eligible_count": len(eligible_exact),
                            }
                        )
                        continue
                    selected_semantic = min(
                        eligible_semantic,
                        key=lambda row: (
                            row["summary"][
                                "resource_equivalent_bits_per_scene"
                            ],
                            -row["summary"]["clean_rate"],
                        ),
                    )
                    selected_exact = min(
                        eligible_exact,
                        key=lambda row: (
                            row["summary"][
                                "resource_equivalent_bits_per_scene"
                            ],
                            -row["summary"]["clean_rate"],
                        ),
                    )
                    semantic_resource = _matrix(
                        [selected_semantic],
                        "resource_equivalent_bits_per_scene",
                    )[0]
                    exact_resource = _matrix(
                        [selected_exact],
                        "resource_equivalent_bits_per_scene",
                    )[0]
                    semantic_actual = _matrix(
                        [selected_semantic], "actual_bits_per_scene"
                    )[0]
                    exact_actual = _matrix(
                        [selected_exact], "actual_bits_per_scene"
                    )[0]
                    resource_savings = (
                        100.0
                        * (exact_resource - semantic_resource)
                        / exact_resource
                    )
                    actual_savings = (
                        100.0
                        * (exact_actual - semantic_actual)
                        / exact_actual
                    )
                    resource_interval = _interval(
                        resource_savings, weights, alpha
                    )
                    actual_interval = _interval(
                        actual_savings, weights, alpha
                    )
                    supported = (
                        resource_interval["confidence_interval_lower"] > 0.0
                        and actual_interval["confidence_interval_lower"] > 0.0
                    )
                    comparison = {
                        "target_clean_rate": target,
                        "deployment_mode": deployment_mode,
                        "requested_session_scene_count": scene_count,
                        "status": "matched",
                        "semantic_clean_lower_bound": float(
                            semantic_lower[
                                semantic_index[
                                    selected_semantic["workpoint_id"]
                                ]
                            ]
                        ),
                        "exact_clean_lower_bound": float(
                            exact_lower[
                                exact_index[selected_exact["workpoint_id"]]
                            ]
                        ),
                        "semantic": _compact_workpoint(selected_semantic),
                        "exact": _compact_workpoint(selected_exact),
                        "resource_savings_percentage": resource_interval,
                        "actual_savings_percentage": actual_interval,
                        "positive_savings_supported_in_coarse_grid": supported,
                    }
                    comparisons.append(comparison)
                    if resource_interval["point_estimate"] > 0.0:
                        selected_pool.append(
                            (
                                _controller_signature(selected_semantic),
                                selected_semantic["workpoint_id"],
                                target,
                                supported,
                                resource_interval[
                                    "confidence_interval_lower"
                                ],
                                resource_interval["point_estimate"],
                            )
                        )

        ranked = sorted(
            selected_pool,
            key=lambda item: (
                -item[3],
                -item[2],
                -item[4],
                -item[5],
                item[1],
            ),
        )
        candidates = []
        seen_signatures = set()
        for (
            signature,
            workpoint_id,
            target,
            supported,
            lower,
            point,
        ) in ranked:
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidates.append(
                {
                    "candidate_id": (
                        f"n{n_key}_controller_{len(candidates) + 1:02d}"
                    ),
                    "heartbeat_silence_scenes": signature[0],
                    "maximum_state_age_minutes": signature[1],
                    "update_reservation_fraction": signature[2],
                    "task_update_open_loop_attempts": signature[3],
                    "ranking_source_workpoint_id": workpoint_id,
                    "highest_supported_target_clean_rate": target,
                    "coarse_positive_savings_supported": supported,
                    "resource_savings_lower_bound_percentage": lower,
                    "resource_savings_point_percentage": point,
                }
            )
            if len(candidates) == int(
                protocol["study_stages"]["grouped_validation"][
                    "maximum_semantic_candidates_per_n"
                ]
            ):
                break
        analysis_by_n[n_key] = {
            "trajectory_count": trajectory_count,
            "pareto_workpoint_ids": pareto,
            "matched_clean_comparisons": comparisons,
            "frozen_grouped_validation_candidates": candidates,
            "supported_comparison_count": sum(
                row.get(
                    "positive_savings_supported_in_coarse_grid", False
                )
                for row in comparisons
            ),
        }
    return {
        "version": "1.0",
        "status": "coarse_grid_analysis_complete",
        "protocol_sha256": coarse["protocol_sha256"],
        "coarse_grid_sha256": None,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence,
            "unit": "paired_link_trajectory",
        },
        "analysis_by_n": analysis_by_n,
        "claim_boundary": (
            "Candidate selection on the excluded 2023 pilot only. "
            "Significance here is development evidence and must be checked "
            "on 2022 grouped validation before any external Final."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--coarse-grid",
        type=Path,
        default=(
            PROJECT_DIR
            / "results/stage6/matched_reliability_coarse_grid_v1/result.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_DIR
            / (
                "results/stage6/matched_reliability_coarse_analysis_v1/"
                "analysis.json"
            )
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite coarse analysis")
    protocol_path = args.protocol.resolve()
    coarse_path = args.coarse_grid.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
    if coarse["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("coarse result protocol hash mismatch")
    if coarse["status"] != "coarse_grid_complete":
        raise ValueError("coarse grid is incomplete")
    if not all(coarse["checks"].values()):
        raise ValueError("coarse grid failed a safety check")
    result = analyze(protocol, coarse)
    result["coarse_grid_sha256"] = sha256_file(coarse_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
