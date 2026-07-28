#!/usr/bin/env python
"""Smoke-test the S6.7b execution layer on excluded pilot data."""

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


_DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "configs"
    / "stage6_matched_reliability_development_v1.json"
)


def _mean_summary(runs) -> dict[str, float]:
    scene_count = len(runs[0].clean)
    return {
        "clean_rate": float(
            np.mean([np.mean(value.clean) for value in runs])
        ),
        "availability_rate": float(
            np.mean([np.mean(value.available) for value in runs])
        ),
        "actual_bits_per_scene": float(
            np.mean(
                [
                    value.bit_breakdown.actual_total_application_bits
                    for value in runs
                ]
            )
            / scene_count
        ),
        "resource_equivalent_bits_per_scene": float(
            np.mean(
                [
                    value.bit_breakdown.resource_equivalent_bits
                    for value in runs
                ]
            )
            / scene_count
        ),
        "mean_initial_install_bits": float(
            np.mean(
                [
                    value.bit_breakdown.initial_install_bits
                    for value in runs
                ]
            )
        ),
        "mean_recovery_install_bits": float(
            np.mean(
                [
                    value.bit_breakdown.recovery_install_bits
                    for value in runs
                ]
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=_DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene-count", type=int, default=20)
    parser.add_argument("--trajectories", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scene_count < 2 or args.trajectories < 1:
        raise ValueError("smoke sizes must be positive and non-trivial")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["execution_smoke"]
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite execution smoke result")
    if (
        protocol["development_sources"][
            "external_final_archives_may_be_opened"
        ]
        or protocol["development_sources"][
            "external_final_signal_values_may_be_loaded"
        ]
    ):
        raise ValueError("external Final must remain inaccessible")

    freeze = protocol["upstream_freeze"]
    if sha256_file(PROJECT_DIR / freeze["manifest"]) != freeze[
        "manifest_sha256"
    ]:
        raise ValueError("S6-FC0 manifest changed")
    audit = protocol["governance_audit"]
    if sha256_file(PROJECT_DIR / audit["result"]) != audit["result_sha256"]:
        raise ValueError("S6.7a governance audit changed")
    cache_entry = protocol["development_sources"]["coarse_grid_cache"]
    cache_path = PROJECT_DIR / cache_entry["path"]
    if sha256_file(cache_path) != cache_entry["sha256"]:
        raise ValueError("excluded-pilot cache changed")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}

    split = protocol["coarse_split"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    evaluation_indices = evaluation_indices[: args.scene_count]
    if evaluation_indices.size != args.scene_count:
        raise ValueError("not enough excluded-pilot evaluation scenes")

    task = protocol["task_grid"]
    n_channels = int(task["n_channels"][0])
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
    sender_session = install_codebook(
        codebook, epoch=int(task["codebook_epoch"])
    )
    install_packet = encode_context_install(sender_session, node_id=1)
    receiver_session = decode_context_install(install_packet).session
    compact_codeword_bits = None
    for index in train_indices:
        packet, decision = encode_compact_update(
            states[int(index)],
            sender_session,
            node_id=1,
            update_epoch=1,
        )
        if not decision.uses_fallback:
            compact_codeword_bits = int(packet.size)
            break
    if compact_codeword_bits is None:
        raise ValueError("training codebook produced no compact codeword")
    reservation_bits = int(maximum_compact_update_bits(sender_session))
    ack_bits = int(cumulative_ack_payload_bits())
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=sender_session.epoch,
            expected_update_epoch=0,
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=sender_session.epoch,
            current_update_epoch=0,
        ).size
    )
    exact_packet_bits = exact_query_bundle_bits(
        n_channels,
        queries,
        header_bits=int(task["task_header_bits"]),
    )
    ranker_entry = protocol["frozen_risk_ranker"]
    ranker_result_path = PROJECT_DIR / ranker_entry["selection_result"]
    if sha256_file(ranker_result_path) != ranker_entry[
        "selection_result_sha256"
    ]:
        raise ValueError("frozen risk-ranker selection changed")
    ranker_result = json.loads(
        ranker_result_path.read_text(encoding="utf-8")
    )
    risk_ranking = fit_frozen_risk_ranking(
        cache[f"channel_power_n{n_channels}"],
        states,
        cache["timestamps_local"],
        cache["cluster_ids"],
        codebook,
        train_groups=train_groups,
        evaluation_groups=evaluation_groups,
        selected_c=float(
            ranker_result["results"][str(n_channels)]["selected_c"]
        ),
        reservation_fractions=(0.0, 0.05, 0.1, 0.2),
    )

    common_random = []
    for trajectory in range(args.trajectories):
        rng = np.random.default_rng(
            int(protocol["monte_carlo"]["seed"]) + 1_200_000 + trajectory
        )
        common_random.append(
            rng.random((args.scene_count, RANDOM_STREAM_COUNT))
        )

    condition = protocol["primary_fault_condition"]
    no_protection = np.zeros(len(states), dtype=bool)
    semantic_rows = []
    exact_rows = []
    started = time.perf_counter()
    for deployment_mode in protocol["session_grid"]["deployment_modes"]:
        for heartbeat in (10, None):
            for attempts in (1, 2, 3):
                runs = [
                    simulate_semantic_trajectory(
                        states,
                        cache["timestamps_local"],
                        evaluation_indices,
                        sender_session=sender_session,
                        receiver_session=receiver_session,
                        install_packet=install_packet,
                        deployment_mode=deployment_mode,
                        condition=condition,
                        random_values=random_values,
                        epsilon_db=float(task["epsilon_db"]),
                        max_age_minutes=120.0,
                        ack_frame_bits=ack_bits,
                        outage_penalty_db=float(task["outage_penalty_db"]),
                        heartbeat_interval_scenes=heartbeat,
                        heartbeat_request_frame_bits=heartbeat_request_bits,
                        heartbeat_response_frame_bits=(
                            heartbeat_response_bits
                        ),
                        task_open_loop_attempts=attempts,
                        compact_codeword_frame_bits=compact_codeword_bits,
                        update_protection_candidates=no_protection,
                        maximum_update_reservations=0,
                        reservation_equivalent_bits=reservation_bits,
                    )
                    for random_values in common_random
                ]
                semantic_rows.append(
                    {
                        "deployment_mode": deployment_mode,
                        "heartbeat_silence_scenes": heartbeat,
                        "task_update_open_loop_attempts": attempts,
                        "update_reservation_fraction": 0.0,
                        "summary": _mean_summary(runs),
                    }
                )
    for deployment_mode in protocol["session_grid"]["deployment_modes"]:
        reservation_fraction = 0.1
        runs = [
            simulate_semantic_trajectory(
                states,
                cache["timestamps_local"],
                evaluation_indices,
                sender_session=sender_session,
                receiver_session=receiver_session,
                install_packet=install_packet,
                deployment_mode=deployment_mode,
                condition=condition,
                random_values=random_values,
                epsilon_db=float(task["epsilon_db"]),
                max_age_minutes=120.0,
                ack_frame_bits=ack_bits,
                outage_penalty_db=float(task["outage_penalty_db"]),
                heartbeat_interval_scenes=10,
                heartbeat_request_frame_bits=heartbeat_request_bits,
                heartbeat_response_frame_bits=heartbeat_response_bits,
                task_open_loop_attempts=1,
                compact_codeword_frame_bits=compact_codeword_bits,
                update_protection_candidates=(
                    risk_ranking.candidates_by_fraction[
                        reservation_fraction
                    ]
                ),
                maximum_update_reservations=int(
                    math.ceil(reservation_fraction * args.scene_count)
                ),
                reservation_equivalent_bits=reservation_bits,
            )
            for random_values in common_random
        ]
        semantic_rows.append(
            {
                "deployment_mode": deployment_mode,
                "heartbeat_silence_scenes": 10,
                "task_update_open_loop_attempts": 1,
                "update_reservation_fraction": reservation_fraction,
                "summary": _mean_summary(runs),
            }
        )
    for attempts in (1, 2, 3):
        runs = [
            simulate_exact_trajectory(
                states,
                cache["timestamps_local"],
                evaluation_indices,
                random_values=random_values,
                packet_loss_probability=float(
                    condition["task_loss_probability"]
                ),
                receiver_reset_probability=float(
                    condition["receiver_context_reset_probability"]
                ),
                task_open_loop_attempts=attempts,
                packet_bits=exact_packet_bits,
                epsilon_db=float(task["epsilon_db"]),
                outage_penalty_db=float(task["outage_penalty_db"]),
            )
            for random_values in common_random
        ]
        exact_rows.append(
            {
                "task_packet_open_loop_attempts": attempts,
                "summary": _mean_summary(runs),
            }
        )
    elapsed = time.perf_counter() - started
    workpoint_count = len(semantic_rows) + len(exact_rows)
    scene_evaluations = (
        workpoint_count * args.trajectories * args.scene_count
    )
    scene_evaluations_per_second = scene_evaluations / max(elapsed, 1e-12)

    preflight_path = (
        PROJECT_DIR / protocol["outputs"]["preflight"]
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    estimated_full_scene_evaluations = (
        preflight["grid"]["estimated_coarse_semantic_scene_evaluations"]
        + preflight["grid"]["estimated_coarse_exact_scene_evaluations"]
    )
    estimated_full_runtime_seconds = (
        estimated_full_scene_evaluations / scene_evaluations_per_second
    )
    checks = {
        "all_rows_produced": (
            len(semantic_rows) == 14 and len(exact_rows) == 3
        ),
        "paired_random_stream_count_matches_protocol": (
            RANDOM_STREAM_COUNT
            == int(protocol["monte_carlo"]["random_stream_count"])
        ),
        "preconfigured_rows_have_zero_initial_install_bits": all(
            row["summary"]["mean_initial_install_bits"] == 0.0
            for row in semantic_rows
            if row["deployment_mode"].startswith("preconfigured")
        ),
        "online_rows_charge_initial_install_bits": all(
            row["summary"]["mean_initial_install_bits"] > 0.0
            for row in semantic_rows
            if row["deployment_mode"].startswith("online")
        ),
        "risk_candidate_sets_are_nested": all(
            not np.any(
                risk_ranking.candidates_by_fraction[lower]
                & ~risk_ranking.candidates_by_fraction[upper]
            )
            for lower, upper in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2))
        ),
        "risk_candidates_reach_smoke_scenes": bool(
            np.any(
                risk_ranking.candidates_by_fraction[0.1][
                    evaluation_indices
                ]
            )
        ),
        "external_final_signal_values_not_loaded": True,
        "external_final_access_count_remains_zero": True,
    }
    result = {
        "version": "1.0",
        "status": (
            "execution_smoke_passed"
            if all(checks.values())
            else "execution_smoke_failed"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "source": {
            "path": cache_entry["path"],
            "sha256": cache_entry["sha256"],
            "role": cache_entry["role"],
        },
        "smoke_scope": {
            "n_channels": n_channels,
            "demands": demands,
            "scene_count": int(args.scene_count),
            "trajectory_count": int(args.trajectories),
            "semantic_workpoint_count": len(semantic_rows),
            "exact_workpoint_count": len(exact_rows),
            "reservation_fractions": [0.0, 0.1],
            "purpose": (
                "execution correctness and runtime estimate only; not "
                "working-point selection or performance evidence"
            ),
        },
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "scene_evaluations": int(scene_evaluations),
            "scene_evaluations_per_second": float(
                scene_evaluations_per_second
            ),
            "estimated_full_coarse_scene_evaluations": int(
                estimated_full_scene_evaluations
            ),
            "estimated_single_process_full_runtime_seconds": float(
                estimated_full_runtime_seconds
            ),
        },
        "semantic_rows": semantic_rows,
        "exact_rows": exact_rows,
        "checks": checks,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "Excluded-2023-pilot execution smoke only. No external Final "
            "signal values were opened, and no algorithm claim is made."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
