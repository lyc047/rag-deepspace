#!/usr/bin/env python
"""Run checkpointed 2022 grouped validation for Stage-6 R1 protocols."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_grouped_validation import (  # noqa: E402
    _group_random,
    _prepare_n,
)
from run_stage6_matched_reliability_coarse_grid import (  # noqa: E402
    _session_random,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    build_grouped_sessions,
)
from spectrum_semcom.stage6_reliability_protocol_r1 import (  # noqa: E402
    R1_VARIANTS,
    combine_r1_trajectory_results,
    simulate_r1_trajectory,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_reliability_protocol_diagnostic_v1.json"
)


def _metric(value) -> dict[str, float]:
    scenes = len(value.clean)
    bits = value.bit_breakdown
    effective = np.asarray(value.effective_regret_db, dtype=np.float64)
    available_regret = np.asarray(
        value.available_regret_db, dtype=np.float64
    )
    result = {
        "clean_rate": float(np.mean(value.clean)),
        "availability_rate": float(np.mean(value.available)),
        "actual_bits_per_scene": float(
            bits.actual_total_application_bits / scenes
        ),
        "resource_equivalent_bits_per_scene": float(
            bits.resource_equivalent_bits / scenes
        ),
        "forward_bits_per_scene": float(
            bits.actual_forward_bits / scenes
        ),
        "feedback_bits_per_scene": float(
            bits.actual_feedback_bits / scenes
        ),
        "task_frame_bits_per_scene": float(
            bits.task_frame_bits / scenes
        ),
        "initial_install_bits_per_scene": float(
            bits.initial_install_bits / scenes
        ),
        "recovery_install_bits_per_scene": float(
            bits.recovery_install_bits / scenes
        ),
        "heartbeat_bits_per_scene": float(
            (
                bits.heartbeat_request_bits
                + bits.heartbeat_response_bits
            )
            / scenes
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "conditional_mean_regret_db": (
            float(np.mean(available_regret))
            if available_regret.size
            else 10.0
        ),
        "wrong_codebook_decode_count": float(
            value.wrong_codebook_decode_count
        ),
        "rejected_compact_count": float(value.rejected_compact_count),
    }
    for field in (
        "initial_install_count",
        "recovery_install_count",
        "compact_refresh_count",
        "exact_refresh_count",
        "duplicate_update_count",
        "ack_frame_count",
        "heartbeat_probe_count",
        "heartbeat_response_count",
        "heartbeat_failure_count",
        "actual_action_reset_count",
        "actual_codebook_reset_count",
        "missing_update_ack_uncertainty_count",
        "failed_heartbeat_uncertainty_count",
        "maximum_belief_size",
    ):
        result[field] = float(getattr(value, field))
    return result


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        field: float(np.mean([row[field] for row in rows]))
        for field in rows[0]
    }


def _verified_inputs(protocol_path: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol["frozen_inputs"]
    for key in (
        "grouped_candidate_protocol",
        "grouped_result",
        "grouped_analysis",
        "fixed_site_cache",
        "excluded_pilot_cache",
    ):
        entry = frozen[key]
        if sha256_file(PROJECT_DIR / entry["path"]) != entry["sha256"]:
            raise ValueError(f"{key} changed")
    if (
        protocol["governance"]["external_final_signal_values_may_be_loaded"]
        or not protocol["governance"][
            "external_final_access_count_must_remain_zero"
        ]
    ):
        raise ValueError("external Final boundary changed")
    candidates = json.loads(
        (
            PROJECT_DIR
            / frozen["grouped_candidate_protocol"]["path"]
        ).read_text(encoding="utf-8")
    )
    parent = json.loads(
        (
            PROJECT_DIR / candidates["parent_protocol"]["path"]
        ).read_text(encoding="utf-8")
    )
    with np.load(
        PROJECT_DIR / frozen["fixed_site_cache"]["path"],
        allow_pickle=False,
    ) as handle:
        fixed = {key: handle[key] for key in handle.files}
    with np.load(
        PROJECT_DIR / frozen["excluded_pilot_cache"]["path"],
        allow_pickle=False,
    ) as handle:
        pilot = {key: handle[key] for key in handle.files}
    return protocol, candidates, parent, fixed, pilot


def _new_checkpoint(
    protocol_sha: str,
    n_channels: int,
    group_ids: list[str],
    prepared: dict,
) -> dict:
    return {
        "version": "1.0",
        "status": "in_progress",
        "protocol_sha256": protocol_sha,
        "n_channels": int(n_channels),
        "demands": list(prepared["demands"]),
        "configured_outer_group_ids": group_ids,
        "completed_outer_group_ids": [],
        "outer_groups": {},
        "external_final_access_count": 0,
    }


def _run_n(
    n_channels: int,
    protocol_path: Path,
    checkpoint_path: Path,
) -> None:
    protocol, candidates, parent, fixed, pilot = _verified_inputs(
        protocol_path
    )
    protocol_sha = sha256_file(protocol_path)
    configured_n = [
        int(value) for value in protocol["evaluation_scope"]["n_channels"]
    ]
    if n_channels not in configured_n:
        raise ValueError("N outside R1 scope")
    prepared = _prepare_n(n_channels, parent, pilot, fixed)
    controller = candidates["controller_candidates"][str(n_channels)][0]
    group_ids = sorted(set(fixed["outer_group_ids"].astype(str)))
    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if (
            checkpoint["protocol_sha256"] != protocol_sha
            or checkpoint["configured_outer_group_ids"] != group_ids
            or int(checkpoint["n_channels"]) != n_channels
        ):
            raise ValueError("R1 checkpoint mismatch")
    else:
        checkpoint = _new_checkpoint(
            protocol_sha, n_channels, group_ids, prepared
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(checkpoint_path, checkpoint)
    completed = set(checkpoint["completed_outer_group_ids"])
    trajectory_count = int(
        protocol["evaluation_scope"]["link_trajectories"]
    )
    n_position = configured_n.index(n_channels)
    started = time.perf_counter()
    for group_position, group_id in enumerate(group_ids):
        if group_id in completed:
            continue
        indices = np.flatnonzero(
            fixed["outer_group_ids"].astype(str) == group_id
        ).astype(np.int64)
        site_id = str(fixed["site_ids"][indices[0]])
        if indices.size < int(
            protocol["evaluation_scope"]["minimum_remainder_scenes"]
        ):
            checkpoint["outer_groups"][group_id] = {
                "outer_group_id": group_id,
                "site_id": site_id,
                "scene_count": int(indices.size),
                "status": "excluded_too_short_for_minimum_session",
                "variant_workpoints": [],
                "elapsed_seconds": 0.0,
            }
        else:
            positions = {
                int(index): position
                for position, index in enumerate(indices)
            }
            common_random = _group_random(
                trajectory_count=trajectory_count,
                scene_count=int(indices.size),
                base_seed=int(parent["monte_carlo"]["seed"]),
                n_position=n_position,
                group_position=group_position,
            )
            sessions_by_length = {
                int(length): build_grouped_sessions(
                    indices,
                    fixed["session_group_ids"],
                    target_scene_count=int(length),
                    minimum_remainder_scenes=int(
                        protocol["evaluation_scope"][
                            "minimum_remainder_scenes"
                        ]
                    ),
                )
                for length in protocol["evaluation_scope"][
                    "session_scene_counts"
                ]
            }
            workpoints = []
            group_started = time.perf_counter()
            for length, sessions in sessions_by_length.items():
                for deployment in protocol["evaluation_scope"][
                    "deployment_modes"
                ]:
                    for variant in R1_VARIANTS:
                        trajectory_metrics = []
                        for trajectory in range(trajectory_count):
                            session_results = [
                                simulate_r1_trajectory(
                                    prepared["states"],
                                    fixed["timestamps_local"],
                                    session,
                                    variant=variant,
                                    sender_session=prepared["sender"],
                                    receiver_session=prepared["receiver"],
                                    install_packet=prepared[
                                        "install_packet"
                                    ],
                                    deployment_mode=deployment,
                                    condition=parent[
                                        "primary_fault_condition"
                                    ],
                                    random_values=_session_random(
                                        common_random[trajectory],
                                        session,
                                        positions,
                                    ),
                                    epsilon_db=float(
                                        parent["task_grid"]["epsilon_db"]
                                    ),
                                    max_age_minutes=float(
                                        controller[
                                            "maximum_state_age_minutes"
                                        ]
                                    ),
                                    ack_frame_bits=prepared["ack_bits"],
                                    outage_penalty_db=float(
                                        parent["task_grid"][
                                            "outage_penalty_db"
                                        ]
                                    ),
                                    heartbeat_interval_scenes=int(
                                        controller[
                                            "heartbeat_silence_scenes"
                                        ]
                                    ),
                                    heartbeat_request_frame_bits=prepared[
                                        "heartbeat_request_bits"
                                    ],
                                    heartbeat_response_frame_bits=prepared[
                                        "heartbeat_response_bits"
                                    ],
                                    task_open_loop_attempts=int(
                                        controller[
                                            "task_update_open_loop_attempts"
                                        ]
                                    ),
                                    compact_codeword_frame_bits=prepared[
                                        "compact_frame_bits"
                                    ],
                                    exact_action_frame_bits=prepared[
                                        "exact_packet_bits"
                                    ],
                                )
                                for session in sessions
                            ]
                            trajectory_metrics.append(
                                _metric(
                                    combine_r1_trajectory_results(
                                        session_results
                                    )
                                )
                            )
                        workpoints.append(
                            {
                                "variant_id": variant,
                                "deployment_mode": deployment,
                                "requested_session_scene_count": int(
                                    length
                                ),
                                "session_count": len(sessions),
                                "evaluated_scene_count": int(
                                    sum(len(session) for session in sessions)
                                ),
                                "summary": _summary(trajectory_metrics),
                                "trajectory_metrics": trajectory_metrics,
                            }
                        )
            checkpoint["outer_groups"][group_id] = {
                "outer_group_id": group_id,
                "site_id": site_id,
                "scene_count": int(indices.size),
                "status": "evaluated",
                "variant_workpoints": workpoints,
                "elapsed_seconds": float(
                    time.perf_counter() - group_started
                ),
            }
        checkpoint["completed_outer_group_ids"].append(group_id)
        checkpoint["elapsed_seconds_current_process"] = float(
            time.perf_counter() - started
        )
        if len(checkpoint["completed_outer_group_ids"]) == len(group_ids):
            checkpoint["status"] = "complete"
        atomic_write_json(checkpoint_path, checkpoint)
        print(
            f"R1 N={n_channels} group={group_id} "
            f"{len(checkpoint['completed_outer_group_ids'])}/{len(group_ids)}",
            flush=True,
        )


def _merge(
    protocol_path: Path,
    checkpoint_dir: Path,
    output: Path,
) -> None:
    protocol, _, _, _, _ = _verified_inputs(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    results = {}
    for n_channels in protocol["evaluation_scope"]["n_channels"]:
        path = checkpoint_dir / f"n{int(n_channels)}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["status"] != "complete"
            or value["protocol_sha256"] != protocol_sha
        ):
            raise ValueError(f"incomplete R1 checkpoint {path}")
        results[str(n_channels)] = value
    wrong = sum(
        row["summary"]["wrong_codebook_decode_count"]
        for value in results.values()
        for group in value["outer_groups"].values()
        for row in group.get("variant_workpoints", [])
    )
    result = {
        "version": "1.0",
        "status": "r1_grouped_validation_complete",
        "protocol_sha256": protocol_sha,
        "results": results,
        "checks": {
            "all_n_complete": True,
            "wrong_codebook_actions_must_be_zero": wrong == 0.0,
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "R1 development validation on historically accessed 2022 data; "
            "not external Final evidence."
        ),
    }
    if output.exists():
        raise FileExistsError("refusing to overwrite R1 grouped result")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--n-channels", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["r1_grouped"]
    )
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else output.parent / "checkpoints"
    )
    if args.merge:
        _merge(protocol_path, checkpoint_dir, output)
    elif args.n_channels is not None:
        _run_n(
            int(args.n_channels),
            protocol_path,
            checkpoint_dir / f"n{int(args.n_channels)}.json",
        )
    else:
        raise ValueError("provide --n-channels or --merge")


if __name__ == "__main__":
    main()
