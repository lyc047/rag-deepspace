#!/usr/bin/env python
"""Run the registered Stage-6R activity and session-length diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_few_shot_adaptation import _load_combined, _queries  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_bit_accounting import (  # noqa: E402
    validate_breakdown_identity,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    simulate_calibrated_semantic_trajectory,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_activity_session_diagnostic_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/activity_session_diagnostic_v1/result.json"
)


def _mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, float]:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if rows.size < 1 or not np.all(np.isfinite(rows)):
        raise ValueError("summary values must be finite")
    rng = np.random.default_rng(int(seed))
    positions = rng.integers(
        0,
        rows.size,
        size=(int(replicates), rows.size),
    )
    boot = np.mean(rows[positions], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(rows)),
        "ci_lower": float(np.quantile(boot, alpha)),
        "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
    }


def _metric_arrays(results: list) -> dict[str, np.ndarray]:
    rows = []
    for result in results:
        validate_breakdown_identity(result.bit_breakdown)
        scenes = len(result.clean)
        bits = result.bit_breakdown
        rows.append(
            {
                "clean_rate": float(np.mean(result.clean)),
                "availability_rate": float(np.mean(result.available)),
                "total_bits_per_scene": float(
                    bits.actual_total_application_bits / scenes
                ),
                "recovery_install_bits_per_scene": float(
                    bits.recovery_install_bits / scenes
                ),
                "initial_install_bits_per_scene": float(
                    bits.initial_install_bits / scenes
                ),
                "heartbeat_bits_per_scene": float(
                    (
                        bits.heartbeat_request_bits
                        + bits.heartbeat_response_bits
                    )
                    / scenes
                ),
                "ack_bits_per_scene": float(bits.ack_bits / scenes),
                "task_bits_per_scene": float(bits.task_frame_bits / scenes),
                "wrong_codebook_decode_count": float(
                    result.wrong_codebook_decode_count
                ),
            }
        )
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _summaries(
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict:
    return {
        name: _mean_ci(
            values,
            seed=int(seed) + position * 1009,
            replicates=replicates,
            confidence=confidence,
        )
        for position, (name, values) in enumerate(arrays.items())
    }


def _paired_savings(
    semantic: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
) -> np.ndarray:
    return 100.0 * (
        exact["total_bits_per_scene"] - semantic["total_bits_per_scene"]
    ) / exact["total_bits_per_scene"]


def _chunks(
    indices: np.ndarray,
    *,
    target_length: int | None,
    calibration_count: int,
) -> list[tuple[np.ndarray, slice]]:
    if target_length is None:
        return [(indices, slice(0, indices.size))]
    rows = []
    for start in range(0, indices.size, int(target_length)):
        stop = min(start + int(target_length), indices.size)
        if stop - start > int(calibration_count):
            rows.append((indices[start:stop], slice(start, stop)))
    if not rows:
        raise ValueError("session segmentation produced no usable chunk")
    return rows


def _aggregate_macro(
    activity_arrays: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    semantic_names = tuple(next(iter(activity_arrays.values()))["semantic"])
    exact_names = tuple(next(iter(activity_arrays.values()))["exact"])
    semantic = {
        name: np.mean(
            [value["semantic"][name] for value in activity_arrays.values()],
            axis=0,
        )
        for name in semantic_names
    }
    exact = {
        name: np.mean(
            [value["exact"][name] for value in activity_arrays.values()],
            axis=0,
        )
        for name in exact_names
    }
    return semantic, exact, _paired_savings(semantic, exact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite activity diagnostic")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        name: PROJECT_DIR / config["inputs"][name]
        for name in (
            "adaptation_config",
            "temporal_gate_result",
            "matched_scan_config",
            "matched_scan_result",
            "external_final_access_state",
        )
    }
    adaptation = json.loads(
        paths["adaptation_config"].read_text(encoding="utf-8")
    )
    temporal = json.loads(
        paths["temporal_gate_result"].read_text(encoding="utf-8")
    )
    scan_config = json.loads(
        paths["matched_scan_config"].read_text(encoding="utf-8")
    )
    scan = json.loads(
        paths["matched_scan_result"].read_text(encoding="utf-8")
    )
    access = json.loads(
        paths["external_final_access_state"].read_text(encoding="utf-8")
    )
    target = float(config["frozen_selection"]["matched_clean_target"])
    primary_k = int(config["frozen_selection"]["primary_k"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    if trajectories != int(scan_config["monte_carlo"]["trajectories"]):
        raise ValueError("trajectory count differs from the matched scan")
    base_seed = int(scan_config["monte_carlo"]["seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    replicates = int(config["monte_carlo"]["bootstrap_replicates"])
    confidence = float(config["monte_carlo"]["confidence_level"])
    condition = scan_config["conditions"]["primary_fault"]
    codebook_epoch = int(scan_config["task"]["codebook_epoch"])
    ack_bits = int(cumulative_ack_payload_bits())
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=codebook_epoch,
            expected_update_epoch=0,
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=codebook_epoch,
            current_update_epoch=0,
        ).size
    )
    n_results = {}
    started = time.perf_counter()

    registered_n = list(map(int, scan_config["task"]["n_channels"]))
    for n_position, n_channels in enumerate(registered_n):
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(
                row,
                queries,
                epsilon_db=float(scan_config["task"]["epsilon_db"]),
            )
            for row in powers
        ]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(
                scan_config["task"]["exact_action_header_bits"]
            ),
        )
        scan_n = scan["conditions"]["primary_fault"][str(n_channels)]
        matched = next(
            row
            for row in scan_n["matched_clean"]
            if row["common_eligible"]
            and abs(float(row["target_clean_rate"]) - target) < 1e-12
        )
        semantic_workpoint = next(
            row
            for row in scan_n["semantic_workpoints"]
            if row["workpoint_id"] == matched["semantic_workpoint_id"]
        )
        exact_workpoint = next(
            row
            for row in scan_n["exact_workpoints"]
            if row["workpoint_id"] == matched["exact_workpoint_id"]
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == primary_k
        }
        activity_inputs = {}
        for campaign_position, campaign in enumerate(
            adaptation["splitting"]["campaign_ids"]
        ):
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            row = temporal_rows[campaign]
            activity_inputs[campaign] = {
                "indices": indices,
                "calibration_count": int(row["calibration_scenes"]),
                "selected_actions": row["selected_actions"],
                "decision": row["decision"],
                "random": np.random.default_rng(
                    base_seed
                    + n_position * 1_000_000
                    + campaign_position * 10_000
                ).random(
                    (trajectories, indices.size, RANDOM_STREAM_COUNT)
                ),
            }

        length_rows = []
        whole_internal = None
        for length_position, target_length in enumerate(
            config["session_length_scenes"]
        ):
            activity_results = {}
            activity_arrays = {}
            activity_output = {}
            for campaign_position, (campaign, value) in enumerate(
                activity_inputs.items()
            ):
                chunk_rows = _chunks(
                    value["indices"],
                    target_length=target_length,
                    calibration_count=value["calibration_count"],
                )
                semantic_runs = []
                exact_runs = []
                for trajectory in range(trajectories):
                    semantic_parts = []
                    exact_parts = []
                    for chunk, random_slice in chunk_rows:
                        random = value["random"][trajectory, random_slice, :]
                        semantic_parts.append(
                            simulate_calibrated_semantic_trajectory(
                                states,
                                timestamps,
                                chunk,
                                calibration_scene_count=value[
                                    "calibration_count"
                                ],
                                selected_actions=value["selected_actions"],
                                random_values=random,
                                condition=condition,
                                exact_packet_bits=exact_packet_bits,
                                epsilon_db=float(
                                    scan_config["task"]["epsilon_db"]
                                ),
                                max_age_minutes=float(
                                    semantic_workpoint[
                                        "maximum_state_age_minutes"
                                    ]
                                ),
                                ack_frame_bits=ack_bits,
                                outage_penalty_db=float(
                                    scan_config["task"][
                                        "outage_penalty_db"
                                    ]
                                ),
                                heartbeat_interval_scenes=(
                                    semantic_workpoint[
                                        "heartbeat_silence_scenes"
                                    ]
                                ),
                                heartbeat_request_frame_bits=(
                                    heartbeat_request_bits
                                ),
                                heartbeat_response_frame_bits=(
                                    heartbeat_response_bits
                                ),
                                calibration_open_loop_attempts=int(
                                    semantic_workpoint[
                                        "calibration_open_loop_attempts"
                                    ]
                                ),
                                task_open_loop_attempts=int(
                                    semantic_workpoint[
                                        "task_update_open_loop_attempts"
                                    ]
                                ),
                                codebook_epoch=codebook_epoch,
                            ).combined
                        )
                        exact_parts.append(
                            simulate_exact_trajectory(
                                states,
                                timestamps,
                                chunk,
                                random_values=random,
                                packet_loss_probability=float(
                                    condition["task_loss_probability"]
                                ),
                                receiver_reset_probability=float(
                                    condition[
                                        "receiver_context_reset_probability"
                                    ]
                                ),
                                task_open_loop_attempts=int(
                                    exact_workpoint[
                                        "task_packet_open_loop_attempts"
                                    ]
                                ),
                                packet_bits=exact_packet_bits,
                                epsilon_db=float(
                                    scan_config["task"]["epsilon_db"]
                                ),
                                outage_penalty_db=float(
                                    scan_config["task"][
                                        "outage_penalty_db"
                                    ]
                                ),
                            )
                        )
                    semantic_runs.append(
                        combine_matched_trajectory_results(semantic_parts)
                    )
                    exact_runs.append(
                        combine_matched_trajectory_results(exact_parts)
                    )
                semantic_arrays = _metric_arrays(semantic_runs)
                exact_arrays = _metric_arrays(exact_runs)
                savings = _paired_savings(semantic_arrays, exact_arrays)
                activity_results[campaign] = {
                    "semantic": semantic_runs,
                    "exact": exact_runs,
                }
                activity_arrays[campaign] = {
                    "semantic": semantic_arrays,
                    "exact": exact_arrays,
                }
                activity_output[campaign] = {
                    "decision": value["decision"],
                    "original_scene_count": int(value["indices"].size),
                    "evaluated_scene_count": int(
                        sum(chunk.size for chunk, _ in chunk_rows)
                    ),
                    "session_count": len(chunk_rows),
                    "calibration_scenes_per_session": value[
                        "calibration_count"
                    ],
                    "semantic": _summaries(
                        semantic_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "exact": _summaries(
                        exact_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                            + 31
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "paired_savings_percentage": _mean_ci(
                        savings,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                            + 67
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                }

            scene_semantic = []
            scene_exact = []
            for trajectory in range(trajectories):
                scene_semantic.append(
                    combine_matched_trajectory_results(
                        [
                            value["semantic"][trajectory]
                            for value in activity_results.values()
                        ]
                    )
                )
                scene_exact.append(
                    combine_matched_trajectory_results(
                        [
                            value["exact"][trajectory]
                            for value in activity_results.values()
                        ]
                    )
                )
            scene_semantic_arrays = _metric_arrays(scene_semantic)
            scene_exact_arrays = _metric_arrays(scene_exact)
            scene_savings = _paired_savings(
                scene_semantic_arrays, scene_exact_arrays
            )
            macro_semantic, macro_exact, macro_savings = _aggregate_macro(
                activity_arrays
            )
            row = {
                "session_length_scenes": target_length,
                "activity_results": activity_output,
                "scene_weighted": {
                    "semantic": _summaries(
                        scene_semantic_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 7001
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "exact": _summaries(
                        scene_exact_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 7002
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "paired_savings_percentage": _mean_ci(
                        scene_savings,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 7003
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                },
                "equal_activity_macro": {
                    "semantic": _summaries(
                        macro_semantic,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 8001
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "exact": _summaries(
                        macro_exact,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 8002
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "paired_savings_percentage": _mean_ci(
                        macro_savings,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + 8003
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                },
            }
            if target_length is None:
                leave_one_out = []
                names = list(activity_results)
                for excluded_position, excluded in enumerate(names):
                    semantic_runs = []
                    exact_runs = []
                    retained = [name for name in names if name != excluded]
                    for trajectory in range(trajectories):
                        semantic_runs.append(
                            combine_matched_trajectory_results(
                                [
                                    activity_results[name]["semantic"][
                                        trajectory
                                    ]
                                    for name in retained
                                ]
                            )
                        )
                        exact_runs.append(
                            combine_matched_trajectory_results(
                                [
                                    activity_results[name]["exact"][
                                        trajectory
                                    ]
                                    for name in retained
                                ]
                            )
                        )
                    semantic_arrays = _metric_arrays(semantic_runs)
                    exact_arrays = _metric_arrays(exact_runs)
                    leave_one_out.append(
                        {
                            "excluded_activity": excluded,
                            "retained_activities": retained,
                            "semantic_clean_rate": _mean_ci(
                                semantic_arrays["clean_rate"],
                                seed=(
                                    bootstrap_seed
                                    + n_position * 100_000
                                    + excluded_position * 100
                                    + 9001
                                ),
                                replicates=replicates,
                                confidence=confidence,
                            ),
                            "paired_savings_percentage": _mean_ci(
                                _paired_savings(
                                    semantic_arrays, exact_arrays
                                ),
                                seed=(
                                    bootstrap_seed
                                    + n_position * 100_000
                                    + excluded_position * 100
                                    + 9002
                                ),
                                replicates=replicates,
                                confidence=confidence,
                            ),
                        }
                    )
                row["leave_one_activity_out"] = leave_one_out
                whole_internal = {
                    "semantic_arrays": scene_semantic_arrays,
                    "exact_arrays": scene_exact_arrays,
                    "row": row,
                }
            length_rows.append(row)

        if whole_internal is None:
            raise ValueError("whole-activity session length is required")
        tolerance = float(
            config["diagnostic_checks"]["reproduction_absolute_tolerance"]
        )
        reproduced = {
            "semantic_clean_rate": bool(abs(
                float(
                    np.mean(
                        whole_internal["semantic_arrays"]["clean_rate"]
                    )
                )
                - float(matched["semantic_clean_rate"])
            )
            <= tolerance),
            "semantic_bits_per_scene": bool(abs(
                float(
                    np.mean(
                        whole_internal["semantic_arrays"][
                            "total_bits_per_scene"
                        ]
                    )
                )
                - float(matched["semantic_resource_bits_per_scene"])
            )
            <= tolerance),
            "exact_clean_rate": bool(abs(
                float(np.mean(whole_internal["exact_arrays"]["clean_rate"]))
                - float(matched["exact_clean_rate"])
            )
            <= tolerance),
            "exact_bits_per_scene": bool(abs(
                float(
                    np.mean(
                        whole_internal["exact_arrays"][
                            "total_bits_per_scene"
                        ]
                    )
                )
                - float(matched["exact_resource_bits_per_scene"])
            )
            <= tolerance),
        }
        whole_row = whole_internal["row"]
        loo_positive = all(
            row["paired_savings_percentage"]["ci_lower"] > 0.0
            for row in whole_row["leave_one_activity_out"]
        )
        weighted_saving = whole_row["scene_weighted"][
            "paired_savings_percentage"
        ]["mean"]
        macro_saving = whole_row["equal_activity_macro"][
            "paired_savings_percentage"
        ]["mean"]
        warning_threshold = float(
            config["diagnostic_checks"][
                "long_activity_dominance_warning_percentage_points"
            ]
        )
        break_even = [
            row["session_length_scenes"]
            for row in length_rows
            if row["session_length_scenes"] is not None
            and row["scene_weighted"]["paired_savings_percentage"][
                "ci_lower"
            ]
            > 0.0
        ]
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "frozen_matched_row": matched,
            "frozen_semantic_workpoint": {
                key: semantic_workpoint[key]
                for key in (
                    "workpoint_id",
                    "heartbeat_silence_scenes",
                    "maximum_state_age_minutes",
                    "task_update_open_loop_attempts",
                    "calibration_open_loop_attempts",
                )
            },
            "frozen_exact_workpoint": exact_workpoint,
            "session_length_results": length_rows,
            "checks": {
                "whole_activity_reproduction": reproduced,
                "whole_activity_reproduction_passed": all(
                    reproduced.values()
                ),
                "all_leave_one_activity_out_savings_lower_bounds_positive": (
                    loo_positive
                ),
                "activity_macro_and_scene_weighted_same_direction": bool(
                    np.sign(weighted_saving) == np.sign(macro_saving)
                ),
                "macro_weighted_savings_difference_percentage_points": float(
                    macro_saving - weighted_saving
                ),
                "long_activity_dominance_warning": bool(
                    abs(macro_saving - weighted_saving)
                    > warning_threshold
                ),
                "shortest_registered_positive_cold_start_session_length": (
                    min(break_even) if break_even else None
                ),
            },
        }
        print(f"S6R activity/session N={n_channels} complete", flush=True)

    result = {
        "version": "1.0",
        "status": "stage6r_activity_session_diagnostic_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "external_access_status_before_run": access.get("status"),
        "external_access_count_before_run": access.get("access_count"),
        "n_results": n_results,
        "checks": {
            "all_n_reproduce_matched_scan": all(
                row["checks"]["whole_activity_reproduction_passed"]
                for row in n_results.values()
            ),
            "all_wrong_codebook_decode_counts_zero": all(
                activity["semantic"]["wrong_codebook_decode_count"]["mean"]
                == 0.0
                for value in n_results.values()
                for length in value["session_length_results"]
                for activity in length["activity_results"].values()
            ),
            "external_final_access_count_unchanged": (
                access.get("access_count") == 1
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["checks"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
