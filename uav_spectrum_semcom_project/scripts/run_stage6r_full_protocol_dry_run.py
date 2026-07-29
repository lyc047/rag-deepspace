#!/usr/bin/env python
"""Run the registered Stage-6R full reliability-protocol dry-run."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_few_shot_adaptation import (  # noqa: E402
    _load_combined,
    _queries,
)
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
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    build_task_state,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    simulate_calibrated_semantic_trajectory,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_full_protocol_dry_run_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/full_protocol_dry_run_v1/result.json"
)


def _mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> tuple[float, float, float]:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    if rows.size < 1 or not np.all(np.isfinite(rows)):
        raise ValueError("confidence interval input must be finite")
    if rows.size == 1:
        value = float(rows[0])
        return value, value, value
    rng = np.random.default_rng(int(seed))
    positions = rng.integers(
        0,
        rows.size,
        size=(int(replicates), rows.size),
    )
    boot = np.mean(rows[positions], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.mean(rows)),
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    )


def _trajectory_values(result) -> dict[str, float]:
    validate_breakdown_identity(result.bit_breakdown)
    scenes = len(result.clean)
    bits = result.bit_breakdown
    return {
        "clean_rate": float(np.mean(result.clean)),
        "availability_rate": float(np.mean(result.available)),
        "actual_total_bits_per_scene": float(
            bits.actual_total_application_bits / scenes
        ),
        "resource_equivalent_bits_per_scene": float(
            bits.resource_equivalent_bits / scenes
        ),
        "exact_action_bits_per_scene": float(
            bits.exact_action_frame_bits / scenes
        ),
        "initial_install_bits_per_scene": float(
            bits.initial_install_bits / scenes
        ),
        "recovery_install_bits_per_scene": float(
            bits.recovery_install_bits / scenes
        ),
        "compact_update_bits_per_scene": float(
            bits.compact_update_bits / scenes
        ),
        "escape_exact_bits_per_scene": float(
            bits.escape_exact_bits / scenes
        ),
        "duplicate_update_bits_per_scene": float(
            bits.duplicate_update_bits / scenes
        ),
        "ack_bits_per_scene": float(bits.ack_bits / scenes),
        "heartbeat_bits_per_scene": float(
            (
                bits.heartbeat_request_bits
                + bits.heartbeat_response_bits
            )
            / scenes
        ),
        "effective_mean_regret_db": float(
            np.mean(result.effective_regret_db)
        ),
        "context_install_count_per_scene": float(
            result.context_install_count / scenes
        ),
        "compact_update_count_per_scene": float(
            result.compact_update_count / scenes
        ),
        "wrong_codebook_decode_count": float(
            result.wrong_codebook_decode_count
        ),
    }


def _summarize(
    values: list[dict[str, float]],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = {
        name: np.asarray([row[name] for row in values], dtype=np.float64)
        for name in values[0]
    }
    output = {}
    for position, (name, rows) in enumerate(arrays.items()):
        mean, lower, upper = _mean_ci(
            rows,
            seed=int(seed) + position * 1009,
            replicates=bootstrap_replicates,
            confidence=confidence,
        )
        output[name] = mean
        output[f"{name}_ci_lower"] = lower
        output[f"{name}_ci_upper"] = upper
    return output, arrays


def _workpoints(config: dict) -> list[dict]:
    grid = config["semantic_working_point_grid"]
    return [
        {
            "heartbeat_silence_scenes": heartbeat,
            "maximum_state_age_minutes": float(age),
            "task_update_open_loop_attempts": int(attempts),
            "calibration_open_loop_attempts": int(attempts),
        }
        for heartbeat, age, attempts in itertools.product(
            grid["heartbeat_silence_scenes"],
            grid["maximum_state_age_minutes"],
            grid["task_update_open_loop_attempts"],
        )
    ]


def _matched_rows(
    semantic: list[dict],
    exact: list[dict],
    arrays_by_id: dict[str, dict[str, np.ndarray]],
    *,
    targets: list[float],
    seed: int,
    bootstrap_replicates: int,
    confidence: float,
) -> list[dict]:
    rows = []
    for target_position, target in enumerate(targets):
        eligible_semantic = [
            row
            for row in semantic
            if row["summary"]["clean_rate_ci_lower"] >= float(target)
        ]
        eligible_exact = [
            row
            for row in exact
            if row["summary"]["clean_rate_ci_lower"] >= float(target)
        ]
        if not eligible_semantic or not eligible_exact:
            rows.append(
                {
                    "target_clean_rate": float(target),
                    "common_eligible": False,
                    "reason": (
                        "semantic_unavailable"
                        if not eligible_semantic
                        else "exact_unavailable"
                    ),
                }
            )
            continue
        semantic_best = min(
            eligible_semantic,
            key=lambda row: row["summary"][
                "resource_equivalent_bits_per_scene"
            ],
        )
        exact_best = min(
            eligible_exact,
            key=lambda row: row["summary"][
                "resource_equivalent_bits_per_scene"
            ],
        )
        semantic_bits = arrays_by_id[semantic_best["workpoint_id"]][
            "resource_equivalent_bits_per_scene"
        ]
        exact_bits = arrays_by_id[exact_best["workpoint_id"]][
            "resource_equivalent_bits_per_scene"
        ]
        paired_savings = 100.0 * (exact_bits - semantic_bits) / exact_bits
        mean, lower, upper = _mean_ci(
            paired_savings,
            seed=int(seed) + target_position * 7919,
            replicates=bootstrap_replicates,
            confidence=confidence,
        )
        rows.append(
            {
                "target_clean_rate": float(target),
                "common_eligible": True,
                "semantic_workpoint_id": semantic_best["workpoint_id"],
                "exact_workpoint_id": exact_best["workpoint_id"],
                "semantic_clean_rate": semantic_best["summary"]["clean_rate"],
                "exact_clean_rate": exact_best["summary"]["clean_rate"],
                "semantic_resource_bits_per_scene": semantic_best["summary"][
                    "resource_equivalent_bits_per_scene"
                ],
                "exact_resource_bits_per_scene": exact_best["summary"][
                    "resource_equivalent_bits_per_scene"
                ],
                "paired_savings_percentage": mean,
                "paired_savings_ci_lower": lower,
                "paired_savings_ci_upper": upper,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--n-channels",
        type=int,
        nargs="*",
        default=None,
        help="Optional execution shard; config and seed identities stay global.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite full-protocol dry-run")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adaptation_path = PROJECT_DIR / config["inputs"]["adaptation_config"]
    temporal_path = PROJECT_DIR / config["inputs"]["temporal_gate_result"]
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    access_state_path = (
        PROJECT_DIR / config["inputs"]["external_final_access_state"]
    )
    access_state = json.loads(access_state_path.read_text(encoding="utf-8"))
    primary_k = int(config["task"]["primary_k"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    bootstrap = int(config["monte_carlo"]["bootstrap_replicates"])
    confidence = float(config["monte_carlo"]["confidence_level"])
    base_seed = int(config["monte_carlo"]["seed"])
    ack_bits = int(cumulative_ack_payload_bits())
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=int(config["task"]["codebook_epoch"]),
            expected_update_epoch=0,
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=int(config["task"]["codebook_epoch"]),
            current_update_epoch=0,
        ).size
    )
    output_by_condition = {}
    started = time.perf_counter()

    registered_n = list(map(int, config["task"]["n_channels"]))
    executed_n = (
        registered_n
        if args.n_channels is None
        else list(map(int, args.n_channels))
    )
    if (
        not executed_n
        or len(executed_n) != len(set(executed_n))
        or any(value not in registered_n for value in executed_n)
    ):
        raise ValueError("execution shard contains an unregistered N value")
    for n_channels in executed_n:
        n_position = registered_n.index(n_channels)
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(
                row,
                queries,
                epsilon_db=float(config["task"]["epsilon_db"]),
            )
            for row in powers
        ]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(config["task"]["exact_action_header_bits"]),
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == primary_k
        }
        sessions = {}
        random_by_campaign = {}
        for campaign_position, campaign in enumerate(
            adaptation["splitting"]["campaign_ids"]
        ):
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            row = temporal_rows[campaign]
            if (
                not row["resolved"]
                or int(row["calibration_scenes"]) >= indices.size
                or not row["selected_actions"]
            ):
                raise ValueError(
                    f"unusable frozen temporal row for {campaign}"
                )
            sessions[campaign] = {
                "indices": indices,
                "calibration_scene_count": int(
                    row["calibration_scenes"]
                ),
                "selected_actions": row["selected_actions"],
                "decision": row["decision"],
            }
            random_by_campaign[campaign] = np.random.default_rng(
                base_seed
                + n_position * 1_000_000
                + campaign_position * 10_000
            ).random(
                (trajectories, indices.size, RANDOM_STREAM_COUNT)
            )

        for condition_position, (condition_name, condition) in enumerate(
            config["conditions"].items()
        ):
            semantic_rows = []
            exact_rows = []
            arrays_by_id = {}
            for workpoint_position, workpoint in enumerate(
                _workpoints(config)
            ):
                runs = []
                metadata = []
                for trajectory in range(trajectories):
                    campaign_runs = []
                    for campaign, session in sessions.items():
                        result = simulate_calibrated_semantic_trajectory(
                            states,
                            timestamps,
                            session["indices"],
                            calibration_scene_count=session[
                                "calibration_scene_count"
                            ],
                            selected_actions=session["selected_actions"],
                            random_values=random_by_campaign[campaign][
                                trajectory
                            ],
                            condition=condition,
                            exact_packet_bits=exact_packet_bits,
                            epsilon_db=float(
                                config["task"]["epsilon_db"]
                            ),
                            max_age_minutes=workpoint[
                                "maximum_state_age_minutes"
                            ],
                            ack_frame_bits=ack_bits,
                            outage_penalty_db=float(
                                config["task"]["outage_penalty_db"]
                            ),
                            heartbeat_interval_scenes=workpoint[
                                "heartbeat_silence_scenes"
                            ],
                            heartbeat_request_frame_bits=(
                                heartbeat_request_bits
                            ),
                            heartbeat_response_frame_bits=(
                                heartbeat_response_bits
                            ),
                            calibration_open_loop_attempts=workpoint[
                                "calibration_open_loop_attempts"
                            ],
                            task_open_loop_attempts=workpoint[
                                "task_update_open_loop_attempts"
                            ],
                            codebook_epoch=int(
                                config["task"]["codebook_epoch"]
                            ),
                        )
                        campaign_runs.append(result.combined)
                        if trajectory == 0:
                            metadata.append(
                                {
                                    "campaign": campaign,
                                    "decision": session["decision"],
                                    "scene_count": int(
                                        session["indices"].size
                                    ),
                                    "calibration_scene_count": int(
                                        session[
                                            "calibration_scene_count"
                                        ]
                                    ),
                                    "future_scene_count": (
                                        result.future_scene_count
                                    ),
                                    "codeword_count": len(
                                        result.codebook.codewords
                                    ),
                                    "install_packet_bits": (
                                        result.install_packet_bits
                                    ),
                                    "compact_frame_bits": (
                                        result.compact_frame_bits
                                    ),
                                }
                            )
                    runs.append(
                        combine_matched_trajectory_results(campaign_runs)
                    )
                values = [_trajectory_values(run) for run in runs]
                workpoint_id = (
                    f"semantic_{condition_name}_n{n_channels}_"
                    f"{workpoint_position:02d}"
                )
                summary, arrays = _summarize(
                    values,
                    seed=(
                        base_seed
                        + n_position * 100_000
                        + condition_position * 10_000
                        + workpoint_position * 100
                    ),
                    bootstrap_replicates=bootstrap,
                    confidence=confidence,
                )
                arrays_by_id[workpoint_id] = arrays
                semantic_rows.append(
                    {
                        "workpoint_id": workpoint_id,
                        **workpoint,
                        "summary": summary,
                        "session_metadata": metadata,
                    }
                )

            for attempts in map(
                int,
                config["exact_reference_grid"][
                    "task_packet_open_loop_attempts"
                ],
            ):
                runs = []
                for trajectory in range(trajectories):
                    campaign_runs = []
                    for campaign, session in sessions.items():
                        campaign_runs.append(
                            simulate_exact_trajectory(
                                states,
                                timestamps,
                                session["indices"],
                                random_values=random_by_campaign[campaign][
                                    trajectory
                                ],
                                packet_loss_probability=float(
                                    condition["task_loss_probability"]
                                ),
                                receiver_reset_probability=float(
                                    condition[
                                        "receiver_context_reset_probability"
                                    ]
                                ),
                                task_open_loop_attempts=attempts,
                                packet_bits=exact_packet_bits,
                                epsilon_db=float(
                                    config["task"]["epsilon_db"]
                                ),
                                outage_penalty_db=float(
                                    config["task"]["outage_penalty_db"]
                                ),
                            )
                        )
                    runs.append(
                        combine_matched_trajectory_results(campaign_runs)
                    )
                values = [_trajectory_values(run) for run in runs]
                workpoint_id = (
                    f"exact_{condition_name}_n{n_channels}_a{attempts}"
                )
                summary, arrays = _summarize(
                    values,
                    seed=(
                        base_seed
                        + n_position * 100_000
                        + condition_position * 10_000
                        + attempts * 1000
                    ),
                    bootstrap_replicates=bootstrap,
                    confidence=confidence,
                )
                arrays_by_id[workpoint_id] = arrays
                exact_rows.append(
                    {
                        "workpoint_id": workpoint_id,
                        "task_packet_open_loop_attempts": attempts,
                        "summary": summary,
                    }
                )

            matched = _matched_rows(
                semantic_rows,
                exact_rows,
                arrays_by_id,
                targets=list(
                    map(float, config["matched_clean"]["targets"])
                ),
                seed=(
                    base_seed
                    + n_position * 100_000
                    + condition_position * 10_000
                    + 777
                ),
                bootstrap_replicates=bootstrap,
                confidence=confidence,
            )
            condition_output = output_by_condition.setdefault(
                condition_name,
                {},
            )
            condition_output[str(n_channels)] = {
                "n_channels": n_channels,
                "total_scene_count": int(
                    sum(
                        session["indices"].size
                        for session in sessions.values()
                    )
                ),
                "exact_packet_bits": exact_packet_bits,
                "semantic_workpoints": semantic_rows,
                "exact_workpoints": exact_rows,
                "matched_clean": matched,
                "maximum_wrong_codebook_decode_count": float(
                    max(
                        row["summary"]["wrong_codebook_decode_count"]
                        for row in semantic_rows
                    )
                ),
            }
            print(
                f"S6R full protocol N={n_channels} "
                f"condition={condition_name} complete",
                flush=True,
            )

    primary = output_by_condition["primary_fault"]
    minimum_common_target = float(
        config.get("dry_run_gates", {}).get(
            "minimum_common_matched_clean_target_under_primary_fault",
            min(map(float, config["matched_clean"]["targets"])),
        )
    )
    common_target_n = []
    positive_savings_n = []
    for key, value in primary.items():
        common = [
            row
            for row in value["matched_clean"]
            if row["common_eligible"]
            and row["target_clean_rate"] >= minimum_common_target
        ]
        if common:
            common_target_n.append(int(key))
        if any(
            row.get("paired_savings_percentage", -math.inf) > 0.0
            for row in common
        ):
            positive_savings_n.append(int(key))
    checks = {
        "all_registered_n_values_executed": (
            len(primary) == len(registered_n)
        ),
        "all_bit_identities_valid": True,
        "all_wrong_codebook_decode_counts_zero": all(
            value["maximum_wrong_codebook_decode_count"] == 0.0
            for condition in output_by_condition.values()
            for value in condition.values()
        ),
        "primary_fault_common_target_n": common_target_n,
        "primary_fault_positive_savings_n": positive_savings_n,
        "full_scan_entry_by_positive_savings": bool(positive_savings_n),
    }
    result = {
        "version": "1.0",
        "status": "stage6r_full_protocol_dry_run_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            "adaptation_config": sha256_file(adaptation_path),
            "temporal_gate_result": sha256_file(temporal_path),
            "external_final_access_state": sha256_file(access_state_path),
        },
        "external_access_status_before_run": access_state.get("status"),
        "external_access_count_before_run": access_state.get("access_count"),
        "registered_n_channels": registered_n,
        "executed_n_channels": executed_n,
        "conditions": output_by_condition,
        "checks": checks,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
