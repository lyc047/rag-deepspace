#!/usr/bin/env python
"""Run controlled Stage-6 context, data, and ACK fault injections."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
    decode_cumulative_ack,
    encode_cumulative_ack,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    decode_compact_update,
    decode_context_install,
    encode_compact_update,
    encode_context_install,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    decode_context_probe_request,
    decode_context_probe_response,
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_context_recovery import (  # noqa: E402
    belief_is_uncertain,
    belief_maximum_age_minutes,
    belief_requires_context_install,
    belief_worst_case_regret_db,
    compact_update_attempt,
    condition_belief_on_cumulative_ack,
    context_install_attempt,
    initial_context_belief,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


@dataclass
class ReceiverState:
    context_installed: bool = False
    codebook_epoch: int | None = None
    actions: tuple[int, ...] | None = None
    update_epoch: int | None = None
    success_timestamp: str | None = None


@dataclass
class TrajectoryResult:
    total_application_bits: int
    forward_application_bits: int
    ack_application_bits: int
    reserved_capacity_bits: int
    context_install_count: int
    compact_update_count: int
    update_reservation_count: int
    protected_update_count: int
    ack_frame_count: int
    stale_ack_rejection_count: int
    rejected_compact_count: int
    wrong_codebook_decode_count: int
    maximum_belief_size: int
    heartbeat_probe_count: int
    heartbeat_response_count: int
    heartbeat_failure_count: int
    available: list[bool]
    clean: list[bool]
    effective_regret_db: list[float]
    available_regret_db: list[float]


def _age_minutes(previous: str | None, current: str) -> float:
    if previous is None:
        return float("inf")
    return (
        datetime.fromisoformat(current) - datetime.fromisoformat(previous)
    ).total_seconds() / 60.0


def simulate_trajectory(
    states,
    timestamps: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    sender_session,
    receiver_session,
    install_packet: np.ndarray,
    method: str,
    condition: dict,
    random_values: np.ndarray,
    epsilon_db: float,
    max_age_minutes: float,
    ack_bits: int,
    outage_penalty_db: float,
    heartbeat_interval_scenes: int | None = None,
    heartbeat_request_bits: int = 0,
    heartbeat_response_bits: int = 0,
    heartbeat_intervals_by_state: np.ndarray | None = None,
    update_protection_candidates: np.ndarray | None = None,
    maximum_update_reservations: int = 0,
    reservation_equivalent_bits: int = 0,
) -> TrajectoryResult:
    adaptive_heartbeat = heartbeat_intervals_by_state is not None
    protected_updates = update_protection_candidates is not None
    if adaptive_heartbeat:
        heartbeat_intervals_by_state = np.asarray(
            heartbeat_intervals_by_state, dtype=np.int64
        )
        if (
            heartbeat_intervals_by_state.shape != (len(states),)
            or np.any(heartbeat_intervals_by_state < 1)
        ):
            raise ValueError("invalid adaptive heartbeat intervals")
    if protected_updates:
        update_protection_candidates = np.asarray(
            update_protection_candidates, dtype=bool
        )
        if (
            update_protection_candidates.shape != (len(states),)
            or maximum_update_reservations < 0
            or reservation_equivalent_bits < 1
            or random_values.shape[1] < 9
        ):
            raise ValueError("invalid predictive update protection")
    if heartbeat_interval_scenes is not None or adaptive_heartbeat:
        if (
            heartbeat_interval_scenes is not None
            and heartbeat_interval_scenes < 1
        ):
            raise ValueError("heartbeat interval must be positive")
        if random_values.shape[1] < 8:
            raise ValueError("heartbeat simulation requires eight random streams")
    receiver = ReceiverState()
    belief = initial_context_belief()
    naive_context_assumed = False
    naive_actions: tuple[int, ...] | None = None
    naive_timestamp: str | None = None
    confirmed_update_epoch: int | None = None
    last_sent_epoch = 0
    last_delivered_ack_epoch: int | None = None
    sent_records: dict[int, tuple[tuple[int, ...], str]] = {}

    forward_bits = 0
    feedback_bits = 0
    reserved_capacity_bits = 0
    install_count = 0
    update_count = 0
    update_reservation_count = 0
    protected_update_count = 0
    ack_count = 0
    stale_rejections = 0
    rejected_compact = 0
    wrong_decode = 0
    heartbeat_probe_count = 0
    heartbeat_response_count = 0
    heartbeat_failure_count = 0
    scenes_since_successful_feedback = 0
    maximum_belief_size = len(belief)
    available_rows = []
    clean_rows = []
    effective_regret = []
    available_regret = []

    fault_enabled = method != "ideal_feedback"
    install_loss = (
        float(condition["install_loss_probability"]) if fault_enabled else 0.0
    )
    task_loss = (
        float(condition["task_loss_probability"]) if fault_enabled else 0.0
    )
    ack_loss = (
        float(condition["ack_loss_probability"]) if fault_enabled else 0.0
    )
    duplicate_probability = (
        float(condition["delayed_duplicate_probability"])
        if fault_enabled
        else 0.0
    )
    reset_probability = (
        float(condition["receiver_context_reset_probability"])
        if fault_enabled
        else 0.0
    )
    include_reset = bool(
        condition["include_reset_hypothesis_after_missing_ack"]
    )

    for local_position, index in enumerate(evaluation_indices):
        task_state = states[int(index)]
        timestamp = str(timestamps[int(index)])
        random = random_values[local_position]
        feedback_received_this_scene = False
        if random[0] < reset_probability:
            receiver = ReceiverState()

        active_heartbeat_interval = (
            int(heartbeat_intervals_by_state[int(index)])
            if adaptive_heartbeat
            else heartbeat_interval_scenes
        )
        if (
            active_heartbeat_interval is not None
            and method not in {"naive_assumption", "ideal_feedback"}
            and scenes_since_successful_feedback
            >= active_heartbeat_interval
            and not belief_requires_context_install(
                belief, codebook_epoch=sender_session.epoch
            )
        ):
            request = encode_context_probe_request(
                node_id=1,
                codebook_epoch=sender_session.epoch,
                expected_update_epoch=last_sent_epoch,
            )
            if int(request.size) != int(heartbeat_request_bits):
                raise ValueError("heartbeat request accounting mismatch")
            forward_bits += int(request.size)
            heartbeat_probe_count += 1
            response_delivered = False
            if random[6] >= task_loss:
                decoded_request = decode_context_probe_request(request)
                if (
                    decoded_request.node_id == 1
                    and decoded_request.codebook_epoch
                    == receiver.codebook_epoch
                    and receiver.context_installed
                    and receiver.actions is not None
                    and receiver.update_epoch is not None
                ):
                    response = encode_context_probe_response(
                        node_id=decoded_request.node_id,
                        codebook_epoch=int(receiver.codebook_epoch),
                        current_update_epoch=int(receiver.update_epoch),
                    )
                    if int(response.size) != int(heartbeat_response_bits):
                        raise ValueError(
                            "heartbeat response accounting mismatch"
                        )
                    feedback_bits += int(response.size)
                    heartbeat_response_count += 1
                    response_delivered = random[7] >= ack_loss
                    if response_delivered:
                        decoded_response = decode_context_probe_response(
                            response
                        )
                        if (
                            decoded_response.node_id != 1
                            or decoded_response.codebook_epoch
                            != sender_session.epoch
                        ):
                            raise ValueError(
                                "heartbeat response context mismatch"
                            )
                        belief, confirmed_update_epoch, accepted = (
                            condition_belief_on_cumulative_ack(
                                belief,
                                ack_epoch=(
                                    decoded_response.current_update_epoch
                                ),
                                confirmed_epoch=confirmed_update_epoch,
                                guard_stale_epochs=False,
                            )
                        )
                        if not accepted:
                            raise ValueError(
                                "heartbeat confirmed an impossible belief"
                            )
                        feedback_received_this_scene = True
            if not response_delivered:
                belief = frozenset(
                    set(belief) | set(initial_context_belief())
                )
                heartbeat_failure_count += 1

        update_reserved = (
            protected_updates
            and bool(update_protection_candidates[int(index)])
            and update_reservation_count
            < int(maximum_update_reservations)
        )
        if update_reserved:
            update_reservation_count += 1
            reserved_capacity_bits += int(reservation_equivalent_bits)

        if method == "naive_assumption":
            needs_install = not naive_context_assumed
        elif method == "ideal_feedback":
            needs_install = not receiver.context_installed
        else:
            needs_install = belief_requires_context_install(
                belief, codebook_epoch=sender_session.epoch
            )

        if needs_install:
            forward_bits += int(install_packet.size)
            install_count += 1
            install_delivered = random[1] >= install_loss
            install_ack_delivered = False
            if install_delivered:
                decoded_install = decode_context_install(install_packet)
                same_context = (
                    receiver.context_installed
                    and receiver.codebook_epoch
                    == decoded_install.session.epoch
                )
                receiver.context_installed = True
                receiver.codebook_epoch = decoded_install.session.epoch
                if not same_context:
                    receiver.actions = None
                    receiver.update_epoch = None
                    receiver.success_timestamp = None
                feedback_bits += ack_bits
                ack_count += 1
                install_ack_delivered = random[2] >= ack_loss
                feedback_received_this_scene = (
                    feedback_received_this_scene or install_ack_delivered
                )
            if method == "naive_assumption":
                naive_context_assumed = True
            else:
                belief = context_install_attempt(
                    belief,
                    codebook_epoch=sender_session.epoch,
                    ack_received=install_ack_delivered,
                )

        if method == "naive_assumption":
            assumed_regret = (
                float("inf")
                if naive_actions is None
                else task_state.max_regret_db(naive_actions)
            )
            assumed_age = _age_minutes(naive_timestamp, timestamp)
            should_update = (
                naive_actions is None
                or assumed_regret > epsilon_db + 1e-12
                or assumed_age > max_age_minutes
            )
        elif method == "ideal_feedback":
            actual_regret = (
                float("inf")
                if receiver.actions is None
                else task_state.max_regret_db(receiver.actions)
            )
            actual_age = _age_minutes(receiver.success_timestamp, timestamp)
            should_update = (
                receiver.actions is None
                or actual_regret > epsilon_db + 1e-12
                or actual_age > max_age_minutes
            )
        else:
            worst_regret = belief_worst_case_regret_db(
                task_state,
                belief,
                codebook_epoch=sender_session.epoch,
            )
            worst_age = belief_maximum_age_minutes(
                belief,
                timestamp_local=timestamp,
                codebook_epoch=sender_session.epoch,
            )
            should_update = (
                worst_regret > epsilon_db + 1e-12
                or worst_age > max_age_minutes
            )
            if method == "immediate_uncertainty_recovery":
                should_update = should_update or belief_is_uncertain(belief)

        current_ack_delivered = False
        current_ack_epoch: int | None = None
        if should_update:
            last_sent_epoch = (last_sent_epoch + 1) % 256
            update_bits, decision = encode_compact_update(
                task_state,
                sender_session,
                node_id=1,
                update_epoch=last_sent_epoch,
            )
            sent_records[last_sent_epoch] = (
                decision.decoder_actions,
                timestamp,
            )
            forward_bits += int(update_bits.size)
            update_count += 1
            update_delivered = random[3] >= task_loss
            if update_reserved:
                forward_bits += int(update_bits.size)
                protected_update_count += 1
                update_delivered = (
                    update_delivered or random[8] >= task_loss
                )
            update_accepted = False
            if update_delivered:
                if (
                    receiver.context_installed
                    and receiver.codebook_epoch == sender_session.epoch
                ):
                    decoded = decode_compact_update(
                        update_bits, receiver_session
                    )
                    if decoded.decoder_actions != decision.decoder_actions:
                        wrong_decode += 1
                    else:
                        receiver.actions = decoded.decoder_actions
                        receiver.update_epoch = decoded.update_epoch
                        receiver.success_timestamp = timestamp
                        update_accepted = True
                else:
                    rejected_compact += 1
            if update_accepted:
                ack_frame = encode_cumulative_ack(
                    node_id=1,
                    epoch=last_sent_epoch,
                )
                ack = decode_cumulative_ack(ack_frame)
                feedback_bits += ack_bits
                ack_count += 1
                current_ack_epoch = ack.epoch
                current_ack_delivered = random[4] >= ack_loss
                feedback_received_this_scene = (
                    feedback_received_this_scene or current_ack_delivered
                )

            if method == "naive_assumption":
                naive_actions = decision.decoder_actions
                naive_timestamp = timestamp
            else:
                belief = compact_update_attempt(
                    belief,
                    codebook_epoch=sender_session.epoch,
                    update_epoch=last_sent_epoch,
                    decoder_actions=decision.decoder_actions,
                    success_timestamp=timestamp,
                    ack_received=False,
                    include_context_loss_on_no_ack=(
                        include_reset and not current_ack_delivered
                    ),
                )
                if current_ack_delivered and current_ack_epoch is not None:
                    belief, confirmed_update_epoch, accepted = (
                        condition_belief_on_cumulative_ack(
                            belief,
                            ack_epoch=current_ack_epoch,
                            confirmed_epoch=confirmed_update_epoch,
                            guard_stale_epochs=True,
                        )
                    )
                    if not accepted:
                        stale_rejections += 1

        duplicate_epoch = last_delivered_ack_epoch
        if (
            duplicate_epoch is not None
            and random[5] < duplicate_probability
        ):
            feedback_bits += ack_bits
            ack_count += 1
            if method == "naive_assumption":
                if duplicate_epoch in sent_records:
                    naive_actions, naive_timestamp = sent_records[
                        duplicate_epoch
                    ]
            else:
                belief, confirmed_update_epoch, accepted = (
                    condition_belief_on_cumulative_ack(
                        belief,
                        ack_epoch=duplicate_epoch,
                        confirmed_epoch=confirmed_update_epoch,
                        guard_stale_epochs=True,
                    )
                )
                if not accepted:
                    stale_rejections += 1
        if current_ack_delivered and current_ack_epoch is not None:
            last_delivered_ack_epoch = current_ack_epoch

        if feedback_received_this_scene:
            scenes_since_successful_feedback = 0
        else:
            scenes_since_successful_feedback += 1
        maximum_belief_size = max(maximum_belief_size, len(belief))
        available = receiver.actions is not None
        available_rows.append(available)
        if available:
            regret = task_state.max_regret_db(receiver.actions)
            available_regret.append(regret)
            effective_regret.append(regret)
            clean_rows.append(regret <= epsilon_db + 1e-12)
        else:
            effective_regret.append(outage_penalty_db)
            clean_rows.append(False)

    return TrajectoryResult(
        total_application_bits=forward_bits + feedback_bits,
        forward_application_bits=forward_bits,
        ack_application_bits=feedback_bits,
        reserved_capacity_bits=reserved_capacity_bits,
        context_install_count=install_count,
        compact_update_count=update_count,
        update_reservation_count=update_reservation_count,
        protected_update_count=protected_update_count,
        ack_frame_count=ack_count,
        stale_ack_rejection_count=stale_rejections,
        rejected_compact_count=rejected_compact,
        wrong_codebook_decode_count=wrong_decode,
        maximum_belief_size=maximum_belief_size,
        heartbeat_probe_count=heartbeat_probe_count,
        heartbeat_response_count=heartbeat_response_count,
        heartbeat_failure_count=heartbeat_failure_count,
        available=available_rows,
        clean=clean_rows,
        effective_regret_db=effective_regret,
        available_regret_db=available_regret,
    )


def summarize_trajectories(
    trajectories: list[TrajectoryResult],
    *,
    scene_count: int,
) -> dict:
    effective = np.asarray(
        [
            value
            for trajectory in trajectories
            for value in trajectory.effective_regret_db
        ],
        dtype=np.float64,
    )
    available_regret = np.asarray(
        [
            value
            for trajectory in trajectories
            for value in trajectory.available_regret_db
        ],
        dtype=np.float64,
    )
    total_scene_count = len(trajectories) * scene_count
    return {
        "trajectory_count": len(trajectories),
        "scene_evaluations": total_scene_count,
        "mean_application_bits_per_scene": float(
            np.mean(
                [value.total_application_bits for value in trajectories]
            )
            / scene_count
        ),
        "mean_forward_bits_per_scene": float(
            np.mean(
                [value.forward_application_bits for value in trajectories]
            )
            / scene_count
        ),
        "mean_ack_bits_per_scene": float(
            np.mean([value.ack_application_bits for value in trajectories])
            / scene_count
        ),
        "mean_reserved_capacity_bits_per_scene": float(
            np.mean(
                [value.reserved_capacity_bits for value in trajectories]
            )
            / scene_count
        ),
        "mean_context_install_count": float(
            np.mean([value.context_install_count for value in trajectories])
        ),
        "mean_compact_update_count": float(
            np.mean([value.compact_update_count for value in trajectories])
        ),
        "mean_update_reservation_count": float(
            np.mean(
                [value.update_reservation_count for value in trajectories]
            )
        ),
        "mean_protected_update_count": float(
            np.mean(
                [value.protected_update_count for value in trajectories]
            )
        ),
        "mean_ack_frame_count": float(
            np.mean([value.ack_frame_count for value in trajectories])
        ),
        "mean_stale_ack_rejection_count": float(
            np.mean(
                [value.stale_ack_rejection_count for value in trajectories]
            )
        ),
        "mean_rejected_compact_count": float(
            np.mean([value.rejected_compact_count for value in trajectories])
        ),
        "wrong_codebook_decode_count": int(
            sum(value.wrong_codebook_decode_count for value in trajectories)
        ),
        "maximum_belief_size": int(
            max(value.maximum_belief_size for value in trajectories)
        ),
        "mean_heartbeat_probe_count": float(
            np.mean([value.heartbeat_probe_count for value in trajectories])
        ),
        "mean_heartbeat_response_count": float(
            np.mean(
                [value.heartbeat_response_count for value in trajectories]
            )
        ),
        "mean_heartbeat_failure_count": float(
            np.mean(
                [value.heartbeat_failure_count for value in trajectories]
            )
        ),
        "availability_rate": float(
            sum(sum(value.available) for value in trajectories)
            / total_scene_count
        ),
        "clean_rate": float(
            sum(sum(value.clean) for value in trajectories)
            / total_scene_count
        ),
        "risk_violation_rate": float(
            1.0
            - sum(sum(value.clean) for value in trajectories)
            / total_scene_count
        ),
        "conditional_mean_regret_db": (
            float(np.mean(available_regret))
            if available_regret.size
            else None
        ),
        "conditional_maximum_regret_db": (
            float(np.max(available_regret))
            if available_regret.size
            else None
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": empirical_cvar_numpy(effective, 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_context_recovery_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/context_recovery_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "context_recovery_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 recovery result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        not governance["faults_are_controlled_injections_not_measurements"]
        or governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("Stage-6 recovery governance is invalid")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("Stage-6 recovery cache hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train, evaluation, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    trajectories = int(protocol["monte_carlo"]["trajectories"])
    base_seed = int(protocol["monte_carlo"]["seed"])
    ack_bits = cumulative_ack_payload_bits()
    if ack_bits != int(grid["ack_application_bits"]):
        raise ValueError("ACK bit declaration does not match real codec")

    all_results = {}
    for n_position, n_value in enumerate(grid["n_channels"]):
        n_channels = int(n_value)
        demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
        queries = tuple(SpectrumTaskQuery(demand) for demand in demands)
        states = [
            build_task_state(
                values,
                queries,
                epsilon_db=float(grid["epsilon_db"]),
            )
            for values in cache[f"channel_power_n{n_channels}"]
        ]
        codebook = fit_greedy_task_codebook(
            [states[int(index)] for index in train]
        )
        sender_session = install_codebook(
            codebook, epoch=int(grid["codebook_epoch"])
        )
        install_packet = encode_context_install(sender_session, node_id=1)
        receiver_session = decode_context_install(install_packet).session
        condition_results = {}
        for condition_position, condition in enumerate(
            protocol["fault_conditions"]
        ):
            rng = np.random.default_rng(
                base_seed
                + n_position * 100_000
                + condition_position * 10_000
            )
            common_random = rng.random(
                (trajectories, len(evaluation), 6),
                dtype=np.float64,
            )
            method_results = {}
            for method in protocol["methods"]:
                runs = [
                    simulate_trajectory(
                        states,
                        cache["timestamps_local"],
                        evaluation,
                        sender_session=sender_session,
                        receiver_session=receiver_session,
                        install_packet=install_packet,
                        method=method,
                        condition=condition,
                        random_values=common_random[trajectory],
                        epsilon_db=float(grid["epsilon_db"]),
                        max_age_minutes=float(
                            grid["max_state_age_minutes"]
                        ),
                        ack_bits=ack_bits,
                        outage_penalty_db=float(grid["outage_penalty_db"]),
                    )
                    for trajectory in range(trajectories)
                ]
                method_results[method] = summarize_trajectories(
                    runs, scene_count=len(evaluation)
                )
            immediate = method_results["immediate_uncertainty_recovery"]
            belief = method_results["belief_risk_recovery"]
            comparison = {
                "belief_actual_bit_reduction_pct_vs_immediate": float(
                    100.0
                    * (
                        immediate["mean_application_bits_per_scene"]
                        - belief["mean_application_bits_per_scene"]
                    )
                    / immediate["mean_application_bits_per_scene"]
                ),
                "belief_clean_rate_difference_vs_immediate": float(
                    belief["clean_rate"] - immediate["clean_rate"]
                ),
                "belief_effective_cvar_difference_db_vs_immediate": float(
                    belief["effective_cvar_0_9_regret_db"]
                    - immediate["effective_cvar_0_9_regret_db"]
                ),
            }
            condition_results[condition["name"]] = {
                "faults": condition,
                "methods": method_results,
                "belief_vs_immediate": comparison,
            }
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "context_install_bits": int(install_packet.size),
            "conditions": condition_results,
        }

    gates = protocol["primary_gates"]
    checks = {
        "wrong_codebook_decode_count_equals_zero": all(
            method["wrong_codebook_decode_count"] == 0
            for n_result in all_results.values()
            for condition in n_result["conditions"].values()
            for method in condition["methods"].values()
        ),
        "clean_condition_clean_rate_equals_one": all(
            abs(method["clean_rate"] - 1.0) <= 1e-12
            for n_result in all_results.values()
            for method in n_result["conditions"]["clean"]["methods"].values()
        ),
        "belief_never_uses_more_bits_than_immediate": all(
            condition["belief_vs_immediate"][
                "belief_actual_bit_reduction_pct_vs_immediate"
            ]
            >= -1e-12
            for n_result in all_results.values()
            for condition in n_result["conditions"].values()
        ),
    }
    if (
        checks["wrong_codebook_decode_count_equals_zero"]
        != bool(gates["wrong_codebook_decode_count_must_equal_zero"])
        or checks["clean_condition_clean_rate_equals_one"]
        != bool(gates["clean_condition_clean_rate_must_equal_one"])
        or checks["belief_never_uses_more_bits_than_immediate"]
        != bool(gates["belief_risk_must_not_use_more_bits_than_immediate_recovery"])
    ):
        raise AssertionError(f"Stage-6 recovery gate failed: {checks}")
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks["final_access_state_unchanged"] = access_before == access_after
    checks["final_access_count_remains_zero"] = (
        access_after["access_count"] == 0
        and not access_after["final_signal_values_accessed"]
    )
    if not all(checks.values()):
        raise AssertionError(f"Stage-6 recovery governance failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_context_recovery_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_sha256": source["cache_sha256"],
            "train_scene_count": int(train.size),
            "evaluation_scene_count": int(evaluation.size),
            "train_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
        },
        "results": all_results,
        "checks": checks,
        "governance": {
            "development_only": True,
            "faults_are_controlled_injections_not_measurements": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "checks": checks,
                "belief_vs_immediate": {
                    n: {
                        condition_name: {
                            "bit_reduction_pct": condition[
                                "belief_vs_immediate"
                            ][
                                "belief_actual_bit_reduction_pct_vs_immediate"
                            ],
                            "clean_rate_difference": condition[
                                "belief_vs_immediate"
                            ][
                                "belief_clean_rate_difference_vs_immediate"
                            ],
                            "cvar_difference_db": condition[
                                "belief_vs_immediate"
                            ][
                                "belief_effective_cvar_difference_db_vs_immediate"
                            ],
                        }
                        for condition_name, condition in value[
                            "conditions"
                        ].items()
                    }
                    for n, value in all_results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
