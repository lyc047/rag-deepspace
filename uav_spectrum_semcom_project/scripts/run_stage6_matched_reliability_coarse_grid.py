#!/usr/bin/env python
"""Run the preregistered S6.7b excluded-pilot coarse reliability grid."""

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
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
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
    encode_compact_update,
    encode_context_install,
    maximum_compact_update_bits,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_frozen_risk_ranker import (  # noqa: E402
    fit_frozen_risk_ranking,
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


DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "configs"
    / "stage6_matched_reliability_development_v1.json"
)


def _trajectory_metrics(value) -> dict[str, float]:
    scene_count = len(value.clean)
    effective = np.asarray(value.effective_regret_db, dtype=np.float64)
    available = np.asarray(value.available_regret_db, dtype=np.float64)
    bits = value.bit_breakdown
    return {
        "clean_rate": float(np.mean(value.clean)),
        "availability_rate": float(np.mean(value.available)),
        "actual_bits_per_scene": float(
            bits.actual_total_application_bits / scene_count
        ),
        "resource_equivalent_bits_per_scene": float(
            bits.resource_equivalent_bits / scene_count
        ),
        "forward_bits_per_scene": float(
            bits.actual_forward_bits / scene_count
        ),
        "feedback_bits_per_scene": float(
            bits.actual_feedback_bits / scene_count
        ),
        "initial_install_bits_per_scene": float(
            bits.initial_install_bits / scene_count
        ),
        "recovery_install_bits_per_scene": float(
            bits.recovery_install_bits / scene_count
        ),
        "task_frame_bits_per_scene": float(
            bits.task_frame_bits / scene_count
        ),
        "heartbeat_bits_per_scene": float(
            (
                bits.heartbeat_request_bits
                + bits.heartbeat_response_bits
            )
            / scene_count
        ),
        "unused_reserved_bits_per_scene": float(
            bits.unused_reserved_capacity_bits / scene_count
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": float(
            empirical_cvar_numpy(effective, 0.9)
        ),
        "conditional_mean_regret_db": (
            float(np.mean(available)) if available.size else 10.0
        ),
        "escape_bits_per_scene": float(
            bits.escape_exact_bits / scene_count
        ),
        "context_install_count": float(value.context_install_count),
        "compact_update_count": float(value.compact_update_count),
        "protected_update_count": float(value.protected_update_count),
        "heartbeat_probe_count": float(value.heartbeat_probe_count),
        "wrong_codebook_decode_count": float(
            value.wrong_codebook_decode_count
        ),
    }


def _summarize(metrics: list[dict[str, float]]) -> dict[str, float]:
    names = tuple(metrics[0])
    return {
        name: float(np.mean([row[name] for row in metrics]))
        for name in names
    }


def _session_random(
    full_random: np.ndarray,
    session: np.ndarray,
    positions: dict[int, int],
) -> np.ndarray:
    return full_random[
        np.asarray([positions[int(index)] for index in session]),
        :,
    ]


def _find_compact_frame_bits(states, train_indices, sender_session) -> int:
    for index in train_indices:
        packet, decision = encode_compact_update(
            states[int(index)],
            sender_session,
            node_id=1,
            update_epoch=1,
        )
        if not decision.uses_fallback:
            return int(packet.size)
    raise ValueError("codebook produced no compact training packet")


def _semantic_workpoints(protocol: dict) -> list[dict]:
    grid = protocol["semantic_working_point_grid"]
    rows = []
    for values in itertools.product(
        grid["heartbeat_silence_scenes"],
        grid["maximum_state_age_minutes"],
        grid["update_reservation_fraction"],
        grid["task_update_open_loop_attempts"],
        protocol["session_grid"]["deployment_modes"],
        protocol["session_grid"]["scene_counts"],
    ):
        heartbeat, age, reservation, attempts, deployment, scenes = values
        rows.append(
            {
                "heartbeat_silence_scenes": heartbeat,
                "maximum_state_age_minutes": float(age),
                "update_reservation_fraction": float(reservation),
                "task_update_open_loop_attempts": int(attempts),
                "deployment_mode": deployment,
                "requested_session_scene_count": int(scenes),
            }
        )
    return rows


def _exact_workpoints(protocol: dict) -> list[dict]:
    return [
        {
            "task_packet_open_loop_attempts": int(attempts),
            "requested_session_scene_count": int(scenes),
        }
        for attempts, scenes in itertools.product(
            protocol["exact_reference_grid"][
                "task_packet_open_loop_attempts"
            ],
            protocol["session_grid"]["scene_counts"],
        )
    ]


def _prepare_n(
    n_channels: int,
    protocol: dict,
    cache: dict,
    train_indices: np.ndarray,
    train_groups: tuple[str, ...],
    evaluation_groups: tuple[str, ...],
    selection_result: dict,
) -> dict:
    task = protocol["task_grid"]
    demands = tuple(
        int(round(n_channels * float(value)))
        for value in task["demand_ratios"]
    )
    queries = tuple(SpectrumTaskQuery(value) for value in demands)
    states = [
        build_task_state(
            values,
            queries,
            epsilon_db=float(task["epsilon_db"]),
        )
        for values in cache[f"channel_power_n{n_channels}"]
    ]
    codebook = fit_greedy_task_codebook(
        [states[int(index)] for index in train_indices]
    )
    sender = install_codebook(codebook, epoch=int(task["codebook_epoch"]))
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    fractions = protocol["semantic_working_point_grid"][
        "update_reservation_fraction"
    ]
    ranking = fit_frozen_risk_ranking(
        cache[f"channel_power_n{n_channels}"],
        states,
        cache["timestamps_local"],
        cache["cluster_ids"],
        codebook,
        train_groups=train_groups,
        evaluation_groups=evaluation_groups,
        selected_c=float(
            selection_result["results"][str(n_channels)]["selected_c"]
        ),
        reservation_fractions=fractions,
    )
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
            states, train_indices, sender
        ),
        "reservation_bits": int(maximum_compact_update_bits(sender)),
        "ack_bits": int(cumulative_ack_payload_bits()),
        "heartbeat_request_bits": request_bits,
        "heartbeat_response_bits": response_bits,
        "risk_ranking": ranking,
        "exact_packet_bits": exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(task["task_header_bits"]),
        ),
    }


def _run_n(
    n_channels: int,
    n_position: int,
    protocol: dict,
    cache: dict,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    train_groups: tuple[str, ...],
    evaluation_groups: tuple[str, ...],
    selection_result: dict,
) -> dict:
    prepared = _prepare_n(
        n_channels,
        protocol,
        cache,
        train_indices,
        train_groups,
        evaluation_groups,
        selection_result,
    )
    trajectory_count = int(protocol["monte_carlo"]["trajectories"])
    seed = int(protocol["monte_carlo"]["seed"]) + n_position * 100_000
    common_random = np.random.default_rng(seed).random(
        (
            trajectory_count,
            evaluation_indices.size,
            RANDOM_STREAM_COUNT,
        )
    )
    positions = {
        int(index): position
        for position, index in enumerate(evaluation_indices)
    }
    session_cache = {
        int(scene_count): build_grouped_sessions(
            evaluation_indices,
            cache["cluster_ids"],
            target_scene_count=int(scene_count),
            minimum_remainder_scenes=2,
        )
        for scene_count in protocol["session_grid"]["scene_counts"]
    }
    task = protocol["task_grid"]
    condition = protocol["primary_fault_condition"]
    semantic_results = []
    exact_results = []
    started = time.perf_counter()

    for workpoint_index, workpoint in enumerate(
        _semantic_workpoints(protocol)
    ):
        sessions = session_cache[
            workpoint["requested_session_scene_count"]
        ]
        fraction = workpoint["update_reservation_fraction"]
        candidates = prepared["risk_ranking"].candidates_by_fraction[
            fraction
        ]
        runs = []
        for trajectory in range(trajectory_count):
            session_runs = []
            for session in sessions:
                session_runs.append(
                    simulate_semantic_trajectory(
                        prepared["states"],
                        cache["timestamps_local"],
                        session,
                        sender_session=prepared["sender"],
                        receiver_session=prepared["receiver"],
                        install_packet=prepared["install_packet"],
                        deployment_mode=workpoint["deployment_mode"],
                        condition=condition,
                        random_values=_session_random(
                            common_random[trajectory],
                            session,
                            positions,
                        ),
                        epsilon_db=float(task["epsilon_db"]),
                        max_age_minutes=workpoint[
                            "maximum_state_age_minutes"
                        ],
                        ack_frame_bits=prepared["ack_bits"],
                        outage_penalty_db=float(
                            task["outage_penalty_db"]
                        ),
                        heartbeat_interval_scenes=workpoint[
                            "heartbeat_silence_scenes"
                        ],
                        heartbeat_request_frame_bits=prepared[
                            "heartbeat_request_bits"
                        ],
                        heartbeat_response_frame_bits=prepared[
                            "heartbeat_response_bits"
                        ],
                        task_open_loop_attempts=workpoint[
                            "task_update_open_loop_attempts"
                        ],
                        compact_codeword_frame_bits=prepared[
                            "compact_frame_bits"
                        ],
                        update_protection_candidates=candidates,
                        maximum_update_reservations=int(
                            math.ceil(fraction * len(session))
                        ),
                        reservation_equivalent_bits=prepared[
                            "reservation_bits"
                        ],
                    )
                )
            runs.append(combine_matched_trajectory_results(session_runs))
        trajectory_metrics = [_trajectory_metrics(value) for value in runs]
        semantic_results.append(
            {
                "workpoint_id": f"semantic_n{n_channels}_{workpoint_index:04d}",
                **workpoint,
                "session_count": len(sessions),
                "evaluated_scene_count": int(
                    sum(len(value) for value in sessions)
                ),
                "risk_threshold": prepared[
                    "risk_ranking"
                ].thresholds_by_fraction[fraction],
                "risk_candidate_count": int(np.sum(candidates)),
                "summary": _summarize(trajectory_metrics),
                "trajectory_metrics": trajectory_metrics,
            }
        )

    for workpoint_index, workpoint in enumerate(_exact_workpoints(protocol)):
        sessions = session_cache[
            workpoint["requested_session_scene_count"]
        ]
        runs = []
        for trajectory in range(trajectory_count):
            session_runs = [
                simulate_exact_trajectory(
                    prepared["states"],
                    cache["timestamps_local"],
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
                    task_open_loop_attempts=workpoint[
                        "task_packet_open_loop_attempts"
                    ],
                    packet_bits=prepared["exact_packet_bits"],
                    epsilon_db=float(task["epsilon_db"]),
                    outage_penalty_db=float(task["outage_penalty_db"]),
                )
                for session in sessions
            ]
            runs.append(combine_matched_trajectory_results(session_runs))
        trajectory_metrics = [_trajectory_metrics(value) for value in runs]
        exact_results.append(
            {
                "workpoint_id": f"exact_n{n_channels}_{workpoint_index:02d}",
                **workpoint,
                "session_count": len(sessions),
                "evaluated_scene_count": int(
                    sum(len(value) for value in sessions)
                ),
                "packet_bits": prepared["exact_packet_bits"],
                "summary": _summarize(trajectory_metrics),
                "trajectory_metrics": trajectory_metrics,
            }
        )

    return {
        "n_channels": n_channels,
        "demands": prepared["demands"],
        "codebook": {
            "codeword_count": len(prepared["sender"].codebook.codewords),
            "symbol_width_bits": (
                prepared["sender"].codebook.symbol_width_bits
            ),
            "context_install_bits": int(prepared["install_packet"].size),
            "compact_codeword_frame_bits": prepared["compact_frame_bits"],
            "maximum_compact_update_bits": prepared["reservation_bits"],
            "exact_query_bundle_bits": prepared["exact_packet_bits"],
        },
        "risk_ranker": {
            "selected_c": prepared["risk_ranking"].selected_c,
            "thresholds_by_fraction": {
                str(key): value
                for key, value in prepared[
                    "risk_ranking"
                ].thresholds_by_fraction.items()
            },
            "candidate_counts_by_fraction": {
                str(key): int(np.sum(value))
                for key, value in prepared[
                    "risk_ranking"
                ].candidates_by_fraction.items()
            },
        },
        "semantic_workpoints": semantic_results,
        "exact_workpoints": exact_results,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-channels", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["coarse_grid"]
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite coarse-grid result")
    if (
        protocol["development_sources"][
            "external_final_archives_may_be_opened"
        ]
        or protocol["development_sources"][
            "external_final_signal_values_may_be_loaded"
        ]
    ):
        raise ValueError("external Final access is forbidden")
    protocol_sha = sha256_file(protocol_path)
    freeze = protocol["upstream_freeze"]
    if sha256_file(PROJECT_DIR / freeze["manifest"]) != freeze[
        "manifest_sha256"
    ]:
        raise ValueError("S6-FC0 manifest changed")
    audit = protocol["governance_audit"]
    if sha256_file(PROJECT_DIR / audit["result"]) != audit["result_sha256"]:
        raise ValueError("S6.7a audit changed")
    ranker = protocol["frozen_risk_ranker"]
    selection_path = PROJECT_DIR / ranker["selection_result"]
    if sha256_file(selection_path) != ranker["selection_result_sha256"]:
        raise ValueError("frozen risk-ranker selection changed")
    selection_result = json.loads(
        selection_path.read_text(encoding="utf-8")
    )
    cache_entry = protocol["development_sources"]["coarse_grid_cache"]
    cache_path = PROJECT_DIR / cache_entry["path"]
    if sha256_file(cache_path) != cache_entry["sha256"]:
        raise ValueError("excluded-pilot cache changed")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["coarse_split"]
    (
        train_indices,
        evaluation_indices,
        train_groups,
        evaluation_groups,
    ) = grouped_train_evaluation_indices(
        cache[split["group_field"]],
        seed=int(split["seed"]),
        train_fraction=float(split["train_group_fraction"]),
    )
    configured_n = [int(value) for value in protocol["task_grid"]["n_channels"]]
    selected_n = (
        configured_n
        if not args.n_channels
        else [int(value) for value in args.n_channels]
    )
    if any(value not in configured_n for value in selected_n):
        raise ValueError("requested N is outside the preregistered grid")
    checkpoint_dir = output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for n_channels in selected_n:
        checkpoint = checkpoint_dir / f"n{n_channels}.json"
        if checkpoint.exists():
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            if value["protocol_sha256"] != protocol_sha:
                raise ValueError("checkpoint protocol hash mismatch")
            results[str(n_channels)] = value["result"]
            continue
        n_position = configured_n.index(n_channels)
        value = _run_n(
            n_channels,
            n_position,
            protocol,
            cache,
            train_indices,
            evaluation_indices,
            train_groups,
            evaluation_groups,
            selection_result,
        )
        checkpoint_value = {
            "protocol_sha256": protocol_sha,
            "n_channels": n_channels,
            "result": value,
        }
        atomic_write_json(checkpoint, checkpoint_value)
        results[str(n_channels)] = value
        print(checkpoint, flush=True)

    complete = selected_n == configured_n
    result = {
        "version": "1.0",
        "status": (
            "coarse_grid_complete"
            if complete
            else "coarse_grid_requested_subset_complete"
        ),
        "protocol_sha256": protocol_sha,
        "source": {
            "path": cache_entry["path"],
            "sha256": cache_entry["sha256"],
            "role": cache_entry["role"],
            "training_group_count": len(train_groups),
            "evaluation_group_count": len(evaluation_groups),
            "evaluation_scene_count": int(evaluation_indices.size),
        },
        "results": results,
        "checks": {
            "requested_n_complete": len(results) == len(selected_n),
            "wrong_codebook_actions_must_be_zero": all(
                all(
                    row["summary"].get(
                        "wrong_codebook_decode_count", 0.0
                    )
                    == 0.0
                    for row in value["semantic_workpoints"]
                )
                for value in results.values()
            ),
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": (
            "Development-only excluded-2023-pilot coarse grid. This may "
            "select candidates for grouped validation but is not external "
            "Final evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
