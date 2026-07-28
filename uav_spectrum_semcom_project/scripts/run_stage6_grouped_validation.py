#!/usr/bin/env python
"""Run frozen-candidate Stage-6 grouped development validation.

The 2022 fixed-site cache is development-only.  This runner rebuilds the
frozen task codebook from the excluded 2023 pilot, never fits on 2022, evaluates
each already-frozen controller on every site-by-date group, and checkpoints
after every outer group.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_matched_reliability_coarse_grid import (  # noqa: E402
    _find_compact_frame_bits,
    _session_random,
    _summarize,
    _trajectory_metrics,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    decode_context_install,
    encode_context_install,
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    build_grouped_sessions,
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
    simulate_semantic_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


DEFAULT_CANDIDATES = (
    PROJECT_DIR / "configs" / "stage6_grouped_validation_candidates_v1.json"
)


def _verify_hash(project_relative_path: str, expected: str, label: str) -> None:
    actual = sha256_file(PROJECT_DIR / project_relative_path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")


def _load_verified_inputs(candidate_path: Path) -> tuple[dict, dict, dict, dict]:
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    parent = candidates["parent_protocol"]
    _verify_hash(parent["path"], parent["sha256"], "parent protocol")
    protocol = json.loads(
        (PROJECT_DIR / parent["path"]).read_text(encoding="utf-8")
    )
    coarse = candidates["coarse_evidence"]
    _verify_hash(coarse["result"], coarse["result_sha256"], "coarse result")
    _verify_hash(
        coarse["analysis"], coarse["analysis_sha256"], "coarse analysis"
    )
    grouped = candidates["grouped_development_cache"]
    _verify_hash(grouped["cache"], grouped["cache_sha256"], "2022 cache")
    _verify_hash(grouped["result"], grouped["result_sha256"], "2022 cache audit")
    pilot = protocol["development_sources"]["coarse_grid_cache"]
    _verify_hash(pilot["path"], pilot["sha256"], "excluded 2023 pilot cache")
    freeze = candidates["freeze_rules"]
    if (
        freeze["may_refit_task_codebook_on_2022"]
        or freeze["may_refit_risk_ranker_on_2022"]
        or freeze["external_final_signal_values_may_be_loaded"]
        or not freeze["external_final_access_count_must_remain_zero"]
    ):
        raise ValueError("candidate freeze does not preserve the safety boundary")
    if (
        protocol["development_sources"]["external_final_archives_may_be_opened"]
        or protocol["development_sources"][
            "external_final_signal_values_may_be_loaded"
        ]
    ):
        raise ValueError("external Final access is forbidden")
    with np.load(PROJECT_DIR / grouped["cache"], allow_pickle=False) as handle:
        fixed_cache = {key: handle[key] for key in handle.files}
    with np.load(PROJECT_DIR / pilot["path"], allow_pickle=False) as handle:
        pilot_cache = {key: handle[key] for key in handle.files}
    return candidates, protocol, fixed_cache, pilot_cache


def _prepare_n(
    n_channels: int,
    protocol: dict,
    pilot_cache: dict,
    fixed_cache: dict,
) -> dict:
    task = protocol["task_grid"]
    demands = tuple(
        int(round(n_channels * float(value)))
        for value in task["demand_ratios"]
    )
    queries = tuple(SpectrumTaskQuery(value) for value in demands)
    pilot_states = [
        build_task_state(
            values,
            queries,
            epsilon_db=float(task["epsilon_db"]),
        )
        for values in pilot_cache[f"channel_power_n{n_channels}"]
    ]
    split = protocol["coarse_split"]
    train_indices, _, _, _ = grouped_train_evaluation_indices(
        pilot_cache[split["group_field"]],
        seed=int(split["seed"]),
        train_fraction=float(split["train_group_fraction"]),
    )
    codebook = fit_greedy_task_codebook(
        [pilot_states[int(index)] for index in train_indices]
    )
    sender = install_codebook(codebook, epoch=int(task["codebook_epoch"]))
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    states = [
        build_task_state(
            values,
            queries,
            epsilon_db=float(task["epsilon_db"]),
        )
        for values in fixed_cache[f"channel_power_n{n_channels}"]
    ]
    request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=sender.epoch,
            expected_update_epoch=0,
        ).size
    )
    response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=sender.epoch,
            current_update_epoch=0,
        ).size
    )
    return {
        "demands": demands,
        "queries": queries,
        "states": states,
        "sender": sender,
        "receiver": receiver,
        "install_packet": install_packet,
        "compact_frame_bits": _find_compact_frame_bits(
            pilot_states, train_indices, sender
        ),
        "reservation_bits": int(maximum_compact_update_bits(sender)),
        "ack_bits": int(cumulative_ack_payload_bits()),
        "heartbeat_request_bits": request_bits,
        "heartbeat_response_bits": response_bits,
        "exact_packet_bits": exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(task["task_header_bits"]),
        ),
        "codebook": {
            "codeword_count": len(codebook.codewords),
            "symbol_width_bits": int(codebook.symbol_width_bits),
            "context_install_bits": int(install_packet.size),
        },
    }


def _group_random(
    *,
    trajectory_count: int,
    scene_count: int,
    base_seed: int,
    n_position: int,
    group_position: int,
) -> np.ndarray:
    seed = int(base_seed) + int(n_position) * 1_000_000 + int(group_position) * 10_000
    return np.random.default_rng(seed).random(
        (trajectory_count, scene_count, RANDOM_STREAM_COUNT)
    )


def _run_semantic_workpoint(
    prepared: dict,
    fixed_cache: dict,
    sessions: tuple[np.ndarray, ...],
    common_random: np.ndarray,
    positions: dict[int, int],
    candidate: dict,
    deployment_mode: str,
    protocol: dict,
) -> dict:
    task = protocol["task_grid"]
    condition = protocol["primary_fault_condition"]
    trajectory_metrics = []
    no_protection = np.zeros(len(prepared["states"]), dtype=bool)
    for trajectory in range(common_random.shape[0]):
        session_runs = []
        for session in sessions:
            session_runs.append(
                simulate_semantic_trajectory(
                    prepared["states"],
                    fixed_cache["timestamps_local"],
                    session,
                    sender_session=prepared["sender"],
                    receiver_session=prepared["receiver"],
                    install_packet=prepared["install_packet"],
                    deployment_mode=deployment_mode,
                    condition=condition,
                    random_values=_session_random(
                        common_random[trajectory], session, positions
                    ),
                    epsilon_db=float(task["epsilon_db"]),
                    max_age_minutes=float(
                        candidate["maximum_state_age_minutes"]
                    ),
                    ack_frame_bits=prepared["ack_bits"],
                    outage_penalty_db=float(task["outage_penalty_db"]),
                    heartbeat_interval_scenes=int(
                        candidate["heartbeat_silence_scenes"]
                    ),
                    heartbeat_request_frame_bits=prepared[
                        "heartbeat_request_bits"
                    ],
                    heartbeat_response_frame_bits=prepared[
                        "heartbeat_response_bits"
                    ],
                    task_open_loop_attempts=int(
                        candidate["task_update_open_loop_attempts"]
                    ),
                    compact_codeword_frame_bits=prepared[
                        "compact_frame_bits"
                    ],
                    update_protection_candidates=no_protection,
                    maximum_update_reservations=0,
                    reservation_equivalent_bits=prepared["reservation_bits"],
                )
            )
        combined = combine_matched_trajectory_results(session_runs)
        trajectory_metrics.append(_trajectory_metrics(combined))
    return {
        "candidate_id": candidate["candidate_id"],
        "controller": {
            key: value
            for key, value in candidate.items()
            if key != "candidate_id"
        },
        "deployment_mode": deployment_mode,
        "session_count": len(sessions),
        "evaluated_scene_count": int(sum(len(value) for value in sessions)),
        "summary": _summarize(trajectory_metrics),
        "trajectory_metrics": trajectory_metrics,
    }


def _run_exact_workpoint(
    prepared: dict,
    fixed_cache: dict,
    sessions: tuple[np.ndarray, ...],
    common_random: np.ndarray,
    positions: dict[int, int],
    attempts: int,
    protocol: dict,
) -> dict:
    task = protocol["task_grid"]
    condition = protocol["primary_fault_condition"]
    trajectory_metrics = []
    for trajectory in range(common_random.shape[0]):
        session_runs = [
            simulate_exact_trajectory(
                prepared["states"],
                fixed_cache["timestamps_local"],
                session,
                random_values=_session_random(
                    common_random[trajectory], session, positions
                ),
                packet_loss_probability=float(
                    condition["task_loss_probability"]
                ),
                receiver_reset_probability=float(
                    condition["receiver_context_reset_probability"]
                ),
                task_open_loop_attempts=int(attempts),
                packet_bits=prepared["exact_packet_bits"],
                epsilon_db=float(task["epsilon_db"]),
                outage_penalty_db=float(task["outage_penalty_db"]),
            )
            for session in sessions
        ]
        combined = combine_matched_trajectory_results(session_runs)
        trajectory_metrics.append(_trajectory_metrics(combined))
    return {
        "task_packet_open_loop_attempts": int(attempts),
        "packet_bits": prepared["exact_packet_bits"],
        "session_count": len(sessions),
        "evaluated_scene_count": int(sum(len(value) for value in sessions)),
        "summary": _summarize(trajectory_metrics),
        "trajectory_metrics": trajectory_metrics,
    }


def _new_checkpoint(
    *,
    n_channels: int,
    candidate_sha: str,
    candidates: dict,
    protocol: dict,
    prepared: dict,
    group_ids: list[str],
) -> dict:
    return {
        "version": "1.0",
        "status": "in_progress",
        "candidate_protocol_sha256": candidate_sha,
        "parent_protocol_sha256": candidates["parent_protocol"]["sha256"],
        "grouped_cache_sha256": candidates["grouped_development_cache"][
            "cache_sha256"
        ],
        "n_channels": int(n_channels),
        "demands": list(prepared["demands"]),
        "codebook": prepared["codebook"],
        "configured_outer_group_ids": group_ids,
        "completed_outer_group_ids": [],
        "outer_groups": {},
        "trajectory_count": int(
            candidates["evaluation_conditions"]["link_trajectories"]
        ),
        "random_stream_count": int(
            candidates["evaluation_conditions"]["paired_random_stream_count"]
        ),
        "external_final_access_count": 0,
        "claim_boundary": (
            "Frozen-candidate grouped development validation on historically "
            "accessed 2022 fixed-site data; not external Final evidence."
        ),
    }


def _run_n(
    n_channels: int,
    candidates: dict,
    protocol: dict,
    fixed_cache: dict,
    pilot_cache: dict,
    candidate_path: Path,
    checkpoint_path: Path,
) -> None:
    configured_n = [int(value) for value in protocol["task_grid"]["n_channels"]]
    if n_channels not in configured_n:
        raise ValueError("N is outside the frozen task grid")
    candidate_sha = sha256_file(candidate_path)
    prepared = _prepare_n(n_channels, protocol, pilot_cache, fixed_cache)
    group_ids = sorted(set(fixed_cache["outer_group_ids"].astype(str)))
    if len(group_ids) != int(
        candidates["grouped_development_cache"]["outer_group_count"]
    ):
        raise ValueError("outer-group count does not match the freeze")
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint["candidate_protocol_sha256"] != candidate_sha
            or checkpoint["grouped_cache_sha256"]
            != candidates["grouped_development_cache"]["cache_sha256"]
            or checkpoint["configured_outer_group_ids"] != group_ids
            or int(checkpoint["n_channels"]) != n_channels
        ):
            raise ValueError("checkpoint does not match frozen inputs")
    else:
        checkpoint = _new_checkpoint(
            n_channels=n_channels,
            candidate_sha=candidate_sha,
            candidates=candidates,
            protocol=protocol,
            prepared=prepared,
            group_ids=group_ids,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(checkpoint_path, checkpoint)

    evaluation = candidates["evaluation_conditions"]
    trajectory_count = int(evaluation["link_trajectories"])
    if int(evaluation["paired_random_stream_count"]) != RANDOM_STREAM_COUNT:
        raise ValueError("paired random-stream layout changed")
    n_position = configured_n.index(n_channels)
    started = time.perf_counter()
    completed = set(checkpoint["completed_outer_group_ids"])
    for group_position, group_id in enumerate(group_ids):
        if group_id in completed:
            continue
        indices = np.flatnonzero(
            fixed_cache["outer_group_ids"].astype(str) == group_id
        ).astype(np.int64)
        if indices.size < 2:
            checkpoint["outer_groups"][group_id] = {
                "outer_group_id": group_id,
                "site_id": str(fixed_cache["site_ids"][indices[0]]),
                "scene_count": int(indices.size),
                "first_timestamp_local": str(
                    fixed_cache["timestamps_local"][indices[0]]
                ),
                "last_timestamp_local": str(
                    fixed_cache["timestamps_local"][indices[-1]]
                ),
                "status": "excluded_too_short_for_minimum_session",
                "exclusion_rule": "minimum_remainder_scenes_is_2",
                "semantic_workpoints": [],
                "exact_workpoints": [],
                "elapsed_seconds": 0.0,
            }
            checkpoint["completed_outer_group_ids"].append(group_id)
            checkpoint["elapsed_seconds_current_process"] = float(
                time.perf_counter() - started
            )
            if len(checkpoint["completed_outer_group_ids"]) == len(group_ids):
                checkpoint["status"] = "complete"
            atomic_write_json(checkpoint_path, checkpoint)
            print(
                f"N={n_channels} group={group_id} excluded_too_short "
                f"{len(checkpoint['completed_outer_group_ids'])}/{len(group_ids)}",
                flush=True,
            )
            continue
        site_values = sorted(set(fixed_cache["site_ids"][indices].astype(str)))
        if len(site_values) != 1:
            raise ValueError(f"group {group_id} crosses sites")
        positions = {
            int(index): position for position, index in enumerate(indices)
        }
        common_random = _group_random(
            trajectory_count=trajectory_count,
            scene_count=int(indices.size),
            base_seed=int(protocol["monte_carlo"]["seed"]),
            n_position=n_position,
            group_position=group_position,
        )
        session_cache = {
            int(scene_count): build_grouped_sessions(
                indices,
                fixed_cache["session_group_ids"],
                target_scene_count=int(scene_count),
                minimum_remainder_scenes=2,
            )
            for scene_count in evaluation["requested_session_scene_counts"]
        }
        semantic_rows = []
        exact_rows = []
        group_started = time.perf_counter()
        for scene_count, sessions in session_cache.items():
            for deployment_mode in evaluation["deployment_modes"]:
                for candidate in candidates["controller_candidates"][
                    str(n_channels)
                ]:
                    row = _run_semantic_workpoint(
                        prepared,
                        fixed_cache,
                        sessions,
                        common_random,
                        positions,
                        candidate,
                        deployment_mode,
                        protocol,
                    )
                    row["requested_session_scene_count"] = int(scene_count)
                    semantic_rows.append(row)
            for attempts in evaluation[
                "exact_task_packet_open_loop_attempts"
            ]:
                row = _run_exact_workpoint(
                    prepared,
                    fixed_cache,
                    sessions,
                    common_random,
                    positions,
                    int(attempts),
                    protocol,
                )
                row["requested_session_scene_count"] = int(scene_count)
                exact_rows.append(row)
        checkpoint["outer_groups"][group_id] = {
            "outer_group_id": group_id,
            "site_id": site_values[0],
            "scene_count": int(indices.size),
            "first_timestamp_local": str(
                fixed_cache["timestamps_local"][indices[0]]
            ),
            "last_timestamp_local": str(
                fixed_cache["timestamps_local"][indices[-1]]
            ),
            "status": "evaluated",
            "semantic_workpoints": semantic_rows,
            "exact_workpoints": exact_rows,
            "elapsed_seconds": float(time.perf_counter() - group_started),
        }
        checkpoint["completed_outer_group_ids"].append(group_id)
        checkpoint["elapsed_seconds_current_process"] = float(
            time.perf_counter() - started
        )
        if len(checkpoint["completed_outer_group_ids"]) == len(group_ids):
            checkpoint["status"] = "complete"
        atomic_write_json(checkpoint_path, checkpoint)
        print(
            f"N={n_channels} group={group_id} "
            f"{len(checkpoint['completed_outer_group_ids'])}/{len(group_ids)} "
            f"seconds={checkpoint['outer_groups'][group_id]['elapsed_seconds']:.1f}",
            flush=True,
        )


def _merge(
    candidates: dict,
    protocol: dict,
    candidate_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> None:
    candidate_sha = sha256_file(candidate_path)
    results = {}
    for n_channels in protocol["task_grid"]["n_channels"]:
        checkpoint_path = checkpoint_dir / f"n{int(n_channels)}.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing {checkpoint_path}")
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            value["status"] != "complete"
            or value["candidate_protocol_sha256"] != candidate_sha
            or int(value["external_final_access_count"]) != 0
        ):
            raise ValueError(f"incomplete or incompatible {checkpoint_path}")
        results[str(n_channels)] = value
    wrong_decode_count = sum(
        row["summary"]["wrong_codebook_decode_count"]
        for value in results.values()
        for group in value["outer_groups"].values()
        for row in group.get("semantic_workpoints", [])
    )
    excluded_group_ids = sorted(
        {
            group_id
            for value in results.values()
            for group_id, group in value["outer_groups"].items()
            if group.get("status", "evaluated") != "evaluated"
        }
    )
    result = {
        "version": "1.0",
        "status": "grouped_validation_complete",
        "candidate_protocol_sha256": candidate_sha,
        "parent_protocol_sha256": candidates["parent_protocol"]["sha256"],
        "source": {
            "cache": candidates["grouped_development_cache"]["cache"],
            "cache_sha256": candidates["grouped_development_cache"][
                "cache_sha256"
            ],
            "role": candidates["grouped_development_cache"]["role"],
            "scene_count": candidates["grouped_development_cache"][
                "scene_count"
            ],
            "outer_group_count": candidates["grouped_development_cache"][
                "outer_group_count"
            ],
            "evaluated_outer_group_count": (
                candidates["grouped_development_cache"]["outer_group_count"]
                - len(excluded_group_ids)
            ),
            "excluded_outer_group_ids": excluded_group_ids,
        },
        "results": results,
        "checks": {
            "all_n_complete": len(results)
            == len(protocol["task_grid"]["n_channels"]),
            "wrong_codebook_actions_must_be_zero": wrong_decode_count == 0.0,
            "candidate_parameters_were_not_refit_on_2022": True,
            "task_codebook_was_not_refit_on_2022": True,
            "too_short_groups_followed_frozen_minimum_remainder_rule": True,
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "environment": environment_snapshot(["numpy", "scikit-learn"]),
        "claim_boundary": (
            "Frozen-candidate grouped development validation on historically "
            "accessed 2022 fixed-site activity. It tests site/date stability "
            "but is not independent external Final evidence."
        ),
    }
    if output_path.exists():
        raise FileExistsError("refusing to overwrite grouped result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, result)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES
    )
    parser.add_argument("--n-channels", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = args.candidates.resolve()
    candidates, protocol, fixed_cache, pilot_cache = _load_verified_inputs(
        candidate_path
    )
    default_output = PROJECT_DIR / protocol["outputs"]["grouped_validation"]
    output_path = (
        args.output.resolve() if args.output is not None else default_output
    )
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else output_path.parent / "checkpoints"
    )
    if args.merge:
        if args.n_channels is not None:
            raise ValueError("--merge cannot be combined with --n-channels")
        _merge(
            candidates,
            protocol,
            candidate_path,
            checkpoint_dir,
            output_path,
        )
        return
    if args.n_channels is None:
        raise ValueError("--n-channels is required for resumable execution")
    _run_n(
        int(args.n_channels),
        candidates,
        protocol,
        fixed_cache,
        pilot_cache,
        candidate_path,
        checkpoint_dir / f"n{int(args.n_channels)}.json",
    )


if __name__ == "__main__":
    main()
