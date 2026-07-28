#!/usr/bin/env python
"""Dry-run the S6.7b bit ledger on excluded pilot data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_context_recovery_development import (  # noqa: E402
    simulate_trajectory,
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
from spectrum_semcom.stage6_bit_accounting import (  # noqa: E402
    reconstruct_frozen_trajectory_bits,
    validate_breakdown_identity,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=_DEFAULT_PROTOCOL,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene-count", type=int, default=20)
    parser.add_argument("--trajectories", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = (
        args.output.resolve()
        if args.output is not None
        else PROJECT_DIR / protocol["outputs"]["instrumentation_dry_run"]
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite dry-run result")
    if args.scene_count < 2 or args.trajectories < 1:
        raise ValueError("dry-run sizes must be positive and non-trivial")

    freeze = protocol["upstream_freeze"]
    freeze_path = PROJECT_DIR / freeze["manifest"]
    if sha256_file(freeze_path) != freeze["manifest_sha256"]:
        raise ValueError("S6-FC0 manifest changed")
    audit = protocol["governance_audit"]
    audit_path = PROJECT_DIR / audit["result"]
    if sha256_file(audit_path) != audit["result_sha256"]:
        raise ValueError("S6.7a governance audit changed")
    if audit["external_final_access_count"] != 0:
        raise ValueError("external Final access count is not zero")

    cache_entry = protocol["development_sources"]["coarse_grid_cache"]
    cache_path = PROJECT_DIR / cache_entry["path"]
    if sha256_file(cache_path) != cache_entry["sha256"]:
        raise ValueError("excluded-pilot cache changed")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}

    split = protocol["coarse_split"]
    train_indices, evaluation_indices, _, _ = (
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
        codebook,
        epoch=int(task["codebook_epoch"]),
    )
    install_packet = encode_context_install(sender_session, node_id=1)
    receiver_session = decode_context_install(install_packet).session
    first_update_bits, first_decision = encode_compact_update(
        states[int(evaluation_indices[0])],
        sender_session,
        node_id=1,
        update_epoch=1,
    )
    if first_decision.uses_fallback:
        raise ValueError("dry-run anchor unexpectedly requires escape")
    compact_codeword_bits = int(first_update_bits.size)
    reservation_bits = int(maximum_compact_update_bits(sender_session))
    ack_bits = int(cumulative_ack_payload_bits())
    request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=sender_session.epoch,
            expected_update_epoch=0,
        ).size
    )
    response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=sender_session.epoch,
            current_update_epoch=0,
        ).size
    )

    protection_candidates = np.zeros(len(states), dtype=bool)
    protection_candidates[evaluation_indices[::3]] = True
    maximum_reservations = int(math.ceil(0.1 * args.scene_count))
    condition = protocol["primary_fault_condition"]
    runs = []
    for trajectory_index in range(args.trajectories):
        rng = np.random.default_rng(
            int(protocol["monte_carlo"]["seed"])
            + 900_000
            + trajectory_index
        )
        random_values = rng.random((args.scene_count, 9))
        trajectory = simulate_trajectory(
            states,
            cache["timestamps_local"],
            evaluation_indices,
            sender_session=sender_session,
            receiver_session=receiver_session,
            install_packet=install_packet,
            method="belief_risk_recovery",
            condition=condition,
            random_values=random_values,
            epsilon_db=float(task["epsilon_db"]),
            max_age_minutes=120.0,
            ack_bits=ack_bits,
            outage_penalty_db=float(task["outage_penalty_db"]),
            heartbeat_interval_scenes=10,
            heartbeat_request_bits=request_bits,
            heartbeat_response_bits=response_bits,
            update_protection_candidates=protection_candidates,
            maximum_update_reservations=maximum_reservations,
            reservation_equivalent_bits=reservation_bits,
        )
        breakdown = reconstruct_frozen_trajectory_bits(
            trajectory,
            context_install_bits=int(install_packet.size),
            compact_codeword_update_bits=compact_codeword_bits,
            ack_frame_bits=ack_bits,
            heartbeat_request_frame_bits=request_bits,
            heartbeat_response_frame_bits=response_bits,
            reservation_equivalent_bits=reservation_bits,
            starts_with_online_install=True,
        )
        validate_breakdown_identity(breakdown)
        runs.append(
            {
                "trajectory_index": trajectory_index,
                "counts": {
                    "context_install_count": trajectory.context_install_count,
                    "compact_update_count": trajectory.compact_update_count,
                    "protected_update_count": trajectory.protected_update_count,
                    "heartbeat_probe_count": trajectory.heartbeat_probe_count,
                    "heartbeat_response_count": (
                        trajectory.heartbeat_response_count
                    ),
                    "ack_frame_count": trajectory.ack_frame_count,
                },
                "bit_breakdown": breakdown.to_dict(),
                "clean_rate": float(np.mean(trajectory.clean)),
                "availability_rate": float(
                    np.mean(trajectory.available)
                ),
            }
        )

    heartbeat_scene_count = 6
    heartbeat_state = states[int(evaluation_indices[0])]
    heartbeat_states = [heartbeat_state] * heartbeat_scene_count
    heartbeat_timestamps = cache["timestamps_local"][:heartbeat_scene_count]
    heartbeat_indices = np.arange(heartbeat_scene_count, dtype=np.int64)
    heartbeat_trajectory = simulate_trajectory(
        heartbeat_states,
        heartbeat_timestamps,
        heartbeat_indices,
        sender_session=sender_session,
        receiver_session=receiver_session,
        install_packet=install_packet,
        method="belief_risk_recovery",
        condition={
            "install_loss_probability": 0.0,
            "task_loss_probability": 0.0,
            "ack_loss_probability": 0.0,
            "delayed_duplicate_probability": 0.0,
            "receiver_context_reset_probability": 0.0,
            "include_reset_hypothesis_after_missing_ack": True,
        },
        random_values=np.ones((heartbeat_scene_count, 9)),
        epsilon_db=float(task["epsilon_db"]),
        max_age_minutes=1_000_000.0,
        ack_bits=ack_bits,
        outage_penalty_db=float(task["outage_penalty_db"]),
        heartbeat_interval_scenes=2,
        heartbeat_request_bits=request_bits,
        heartbeat_response_bits=response_bits,
    )
    heartbeat_breakdown = reconstruct_frozen_trajectory_bits(
        heartbeat_trajectory,
        context_install_bits=int(install_packet.size),
        compact_codeword_update_bits=compact_codeword_bits,
        ack_frame_bits=ack_bits,
        heartbeat_request_frame_bits=request_bits,
        heartbeat_response_frame_bits=response_bits,
        reservation_equivalent_bits=reservation_bits,
        starts_with_online_install=True,
    )
    validate_breakdown_identity(heartbeat_breakdown)
    heartbeat_probe = {
        "scene_count": heartbeat_scene_count,
        "heartbeat_interval_scenes": 2,
        "counts": {
            "heartbeat_probe_count": (
                heartbeat_trajectory.heartbeat_probe_count
            ),
            "heartbeat_response_count": (
                heartbeat_trajectory.heartbeat_response_count
            ),
            "ack_frame_count": heartbeat_trajectory.ack_frame_count,
        },
        "bit_breakdown": heartbeat_breakdown.to_dict(),
    }

    checks = {
        "all_bit_identities_pass": True,
        "all_escape_remainders_nonnegative": all(
            value["bit_breakdown"]["escape_exact_bits"] >= 0
            for value in runs
        ),
        "all_other_feedback_remainders_zero": all(
            value["bit_breakdown"]["other_feedback_bits"] == 0
            for value in runs
        )
        and heartbeat_breakdown.other_feedback_bits == 0,
        "heartbeat_request_and_response_observed": (
            heartbeat_trajectory.heartbeat_probe_count > 0
            and heartbeat_trajectory.heartbeat_response_count > 0
            and heartbeat_breakdown.heartbeat_request_bits > 0
            and heartbeat_breakdown.heartbeat_response_bits > 0
        ),
        "external_final_signal_values_not_loaded": True,
        "external_final_access_count_remains_zero": True,
    }
    result = {
        "version": "1.0",
        "status": (
            "instrumentation_dry_run_passed"
            if all(checks.values())
            else "instrumentation_dry_run_failed"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "source": {
            "cache": cache_entry["path"],
            "cache_sha256": cache_entry["sha256"],
            "role": cache_entry["role"],
        },
        "dry_run": {
            "n_channels": n_channels,
            "demands": demands,
            "scene_count": int(args.scene_count),
            "trajectory_count": int(args.trajectories),
            "context_install_bits": int(install_packet.size),
            "compact_codeword_update_bits": compact_codeword_bits,
            "maximum_compact_update_bits": reservation_bits,
            "ack_bits": ack_bits,
            "heartbeat_request_bits": request_bits,
            "heartbeat_response_bits": response_bits,
            "runs": runs,
            "heartbeat_accounting_probe": heartbeat_probe,
        },
        "checks": checks,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "Instrumentation identity dry-run on permanently excluded 2023 "
            "pilot data. It is not a rate-risk result or external Final."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
