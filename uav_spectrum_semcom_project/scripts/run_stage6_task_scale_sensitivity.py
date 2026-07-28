#!/usr/bin/env python
"""Run checkpointed Stage-6 epsilon and task-scale sensitivity experiments."""

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

from run_stage6_grouped_validation import _group_random  # noqa: E402
from run_stage6_matched_reliability_coarse_grid import (  # noqa: E402
    _find_compact_frame_bits,
    _session_random,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
)
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
)
from spectrum_semcom.stage6_reliability_protocol_r1 import (  # noqa: E402
    combine_r1_trajectory_results,
    simulate_r1_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6_task_scale_sensitivity import (  # noqa: E402
    TaskScaleGridPoint,
    codeword_usage_metrics,
    registered_grid_points,
    validate_complete_query_lattice,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_task_scale_sensitivity_v1.json"
)
DEFAULT_CHECKPOINT_DIR = (
    PROJECT_DIR
    / "results/stage6/task_scale_sensitivity_grouped_v1/checkpoints"
)
DEFAULT_RESULT = (
    PROJECT_DIR / "results/stage6/task_scale_sensitivity_grouped_v1/result.json"
)
DEFAULT_REPRESENTATION = (
    PROJECT_DIR / "results/stage6/task_scale_representation_v1/result.json"
)
DEFAULT_PREFLIGHT = (
    PROJECT_DIR
    / "results/stage6/task_scale_sensitivity_preflight_v1/preflight.json"
)


def _verified_inputs(protocol_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for label, entry in protocol["frozen_inputs"].items():
        actual = sha256_file(PROJECT_DIR / entry["path"])
        if actual != entry["sha256"]:
            raise ValueError(f"{label} hash mismatch: {actual}")
    governance = protocol["governance"]
    if (
        governance["external_final_signal_values_may_be_loaded"]
        or not governance["external_final_access_count_must_remain_zero"]
        or governance["2022_may_be_used_to_refit_codebook"]
        or governance["2022_may_be_used_to_retune_controller"]
    ):
        raise ValueError("sensitivity governance boundary changed")
    validate_complete_query_lattice(protocol)
    points = registered_grid_points(protocol)
    expected = (
        len(protocol["task_grid"]["n_channels"])
        * len(protocol["task_grid"]["epsilon_db"])
        * len(protocol["task_grid"]["query_sets"])
    )
    if len(points) != expected:
        raise ValueError("registered sensitivity grid is incomplete")
    r1_analysis_entry = protocol["frozen_inputs"]["r1_grouped_analysis"]
    r1_analysis = json.loads(
        (PROJECT_DIR / r1_analysis_entry["path"]).read_text(encoding="utf-8")
    )
    if (
        r1_analysis["decision_summary"]["protocol_decision"]
        != "retain_task_semantic_codebook_as_primary_protocol_contribution"
        or not r1_analysis["checks"]["external_final_access_count_remains_zero"]
    ):
        raise ValueError("R1 did not authorize sensitivity analysis")
    parent_entry = protocol["frozen_inputs"]["matched_reliability_parent"]
    parent = json.loads(
        (PROJECT_DIR / parent_entry["path"]).read_text(encoding="utf-8")
    )
    candidates_entry = protocol["frozen_inputs"]["grouped_candidates"]
    candidates = json.loads(
        (PROJECT_DIR / candidates_entry["path"]).read_text(encoding="utf-8")
    )
    fixed_entry = protocol["frozen_inputs"]["fixed_site_cache"]
    with np.load(PROJECT_DIR / fixed_entry["path"], allow_pickle=False) as handle:
        fixed = {key: handle[key] for key in handle.files}
    pilot_entry = protocol["frozen_inputs"]["excluded_pilot_cache"]
    with np.load(PROJECT_DIR / pilot_entry["path"], allow_pickle=False) as handle:
        pilot = {key: handle[key] for key in handle.files}
    return protocol, parent, candidates, fixed, pilot


def _trajectory_metric(value) -> dict[str, float]:
    scenes = len(value.clean)
    if scenes < 1:
        raise ValueError("empty trajectory result")
    bits = value.bit_breakdown
    effective = np.asarray(value.effective_regret_db, dtype=np.float64)
    available = np.asarray(value.available_regret_db, dtype=np.float64)
    result = {
        "clean_rate": float(np.mean(value.clean)),
        "availability_rate": float(np.mean(value.available)),
        "actual_bits_per_scene": float(
            bits.actual_total_application_bits / scenes
        ),
        "resource_equivalent_bits_per_scene": float(
            bits.resource_equivalent_bits / scenes
        ),
        "forward_bits_per_scene": float(bits.actual_forward_bits / scenes),
        "feedback_bits_per_scene": float(bits.actual_feedback_bits / scenes),
        "task_frame_bits_per_scene": float(bits.task_frame_bits / scenes),
        "heartbeat_bits_per_scene": float(
            (bits.heartbeat_request_bits + bits.heartbeat_response_bits)
            / scenes
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": float(
            empirical_cvar_numpy(effective, 0.9)
        ),
        "conditional_mean_regret_db": (
            float(np.mean(available)) if available.size else 10.0
        ),
        "conditional_cvar_0_9_regret_db": (
            float(empirical_cvar_numpy(available, 0.9))
            if available.size
            else 10.0
        ),
        "wrong_codebook_decode_count": float(
            getattr(value, "wrong_codebook_decode_count", 0)
        ),
    }
    for field in (
        "compact_refresh_count",
        "exact_refresh_count",
        "actual_action_reset_count",
        "actual_codebook_reset_count",
        "missing_update_ack_uncertainty_count",
        "failed_heartbeat_uncertainty_count",
        "maximum_belief_size",
    ):
        result[field] = float(getattr(value, field, 0))
    return result


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        field: float(np.mean([row[field] for row in rows]))
        for field in rows[0]
    }


def _exact_trajectory_metric(value) -> dict[str, float]:
    """Keep only exact-reference fields used by the paired analysis."""

    scenes = len(value.clean)
    effective = np.asarray(value.effective_regret_db, dtype=np.float64)
    return {
        "clean_rate": float(np.mean(value.clean)),
        "availability_rate": float(np.mean(value.available)),
        "actual_bits_per_scene": float(
            value.bit_breakdown.actual_total_application_bits / scenes
        ),
        "effective_mean_regret_db": float(np.mean(effective)),
        "effective_cvar_0_9_regret_db": float(
            empirical_cvar_numpy(effective, 0.9)
        ),
    }


def _decision_summary(decisions, codebook) -> dict:
    symbols = [int(value.symbol_id) for value in decisions]
    usage = codeword_usage_metrics(
        symbols,
        codeword_count=len(codebook.codewords),
        escape_symbol=codebook.escape_symbol,
    )
    regrets = np.asarray(
        [float(value.max_regret_db) for value in decisions], dtype=np.float64
    )
    return usage | {
        "mean_max_query_regret_db": float(np.mean(regrets)),
        "cvar_0_9_max_query_regret_db": float(
            empirical_cvar_numpy(regrets, 0.9)
        ),
        "maximum_max_query_regret_db": float(np.max(regrets)),
    }


def _prepare_grid_point(
    point: TaskScaleGridPoint,
    protocol: dict,
    parent: dict,
    pilot: dict,
    fixed: dict,
) -> dict:
    queries = tuple(SpectrumTaskQuery(value) for value in point.demands)
    pilot_states = [
        build_task_state(values, queries, epsilon_db=point.epsilon_db)
        for values in pilot[f"channel_power_n{point.n_channels}"]
    ]
    split = protocol["codebook_fit_split"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            pilot[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    training_states = [pilot_states[int(index)] for index in train_indices]
    codebook = fit_greedy_task_codebook(training_states)
    fixed_states = [
        build_task_state(values, queries, epsilon_db=point.epsilon_db)
        for values in fixed[f"channel_power_n{point.n_channels}"]
    ]
    pilot_decisions = [
        encode_task_state(codebook, pilot_states[int(index)])
        for index in evaluation_indices
    ]
    fixed_decisions = [
        encode_task_state(codebook, state) for state in fixed_states
    ]
    training_action_tuples = {state.optimal_actions for state in training_states}
    new_action_rate = float(
        np.mean(
            [
                state.optimal_actions not in training_action_tuples
                for state in fixed_states
            ]
        )
    )
    representation = {
        "grid_point_id": point.grid_point_id,
        "n_channels": point.n_channels,
        "epsilon_db": point.epsilon_db,
        "query_set_id": point.query_set_id,
        "demand_ratios": list(point.demand_ratios),
        "demands": list(point.demands),
        "query_count": len(point.demands),
        "pilot_train_scene_count": len(train_indices),
        "pilot_evaluation_scene_count": len(evaluation_indices),
        "pilot_train_group_count": len(train_groups),
        "pilot_evaluation_group_count": len(evaluation_groups),
        "training_exact_action_tuple_count": int(
            codebook.exact_action_tuple_count
        ),
        "codeword_count": len(codebook.codewords),
        "symbol_width_bits": int(codebook.symbol_width_bits),
        "training_coverage_rate": float(
            codebook.training_covered_count / codebook.training_scene_count
        ),
        "pilot_evaluation": _decision_summary(pilot_decisions, codebook),
        "fixed_site_2022": _decision_summary(fixed_decisions, codebook),
        "new_exact_action_tuple_rate_on_2022": new_action_rate,
        "decoder_actions": [
            list(value.decoder_actions) for value in codebook.codewords
        ],
    }
    deployable = 1 <= len(codebook.codewords) <= 255
    if not deployable:
        return {
            "point": point,
            "queries": queries,
            "states": fixed_states,
            "representation": representation
            | {
                "deployable_in_current_context_codec": False,
                "non_deployable_reason": "codeword_count_outside_1_to_255",
                "context_install_bits": None,
                "compact_frame_bits": None,
                "exact_action_frame_bits": int(
                    exact_query_bundle_bits(
                        point.n_channels,
                        queries,
                        header_bits=int(protocol["task_grid"]["task_header_bits"]),
                    )
                ),
            },
            "deployable": False,
        }
    sender = install_codebook(
        codebook, epoch=int(protocol["task_grid"]["codebook_epoch"])
    )
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
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
    compact_bits = _find_compact_frame_bits(
        pilot_states, train_indices, sender
    )
    exact_bits = exact_query_bundle_bits(
        point.n_channels,
        queries,
        header_bits=int(protocol["task_grid"]["task_header_bits"]),
    )
    representation |= {
        "deployable_in_current_context_codec": True,
        "non_deployable_reason": None,
        "context_install_bits": int(install_packet.size),
        "compact_frame_bits": int(compact_bits),
        "exact_action_frame_bits": int(exact_bits),
        "codebook_manifest_sha256": sender.manifest_sha256,
    }
    return {
        "point": point,
        "queries": queries,
        "states": fixed_states,
        "representation": representation,
        "deployable": True,
        "sender": sender,
        "receiver": receiver,
        "install_packet": install_packet,
        "compact_frame_bits": int(compact_bits),
        "exact_packet_bits": int(exact_bits),
        "ack_bits": int(cumulative_ack_payload_bits()),
        "heartbeat_request_bits": request_bits,
        "heartbeat_response_bits": response_bits,
    }


def _run_group(
    prepared: dict,
    group_id: str,
    group_position: int,
    n_position: int,
    protocol: dict,
    parent: dict,
    candidates: dict,
    fixed: dict,
) -> dict:
    point = prepared["point"]
    indices = np.flatnonzero(
        fixed["outer_group_ids"].astype(str) == group_id
    ).astype(np.int64)
    site_id = str(fixed["site_ids"][indices[0]])
    minimum = int(protocol["system_evaluation"]["minimum_remainder_scenes"])
    if indices.size < minimum:
        return {
            "outer_group_id": group_id,
            "site_id": site_id,
            "scene_count": int(indices.size),
            "status": "excluded_too_short_for_minimum_session",
        }
    positions = {int(index): position for position, index in enumerate(indices)}
    trajectory_count = int(protocol["system_evaluation"]["link_trajectories"])
    common_random = _group_random(
        trajectory_count=trajectory_count,
        scene_count=int(indices.size),
        base_seed=int(protocol["system_evaluation"]["random_seed"]),
        n_position=n_position,
        group_position=group_position,
    )
    sessions = build_grouped_sessions(
        indices,
        fixed["session_group_ids"],
        target_scene_count=int(
            protocol["frozen_architecture"]["session_scene_count"]
        ),
        minimum_remainder_scenes=minimum,
    )
    controller = candidates["controller_candidates"][str(point.n_channels)][0]
    condition = parent["primary_fault_condition"]
    semantic_metrics = []
    exact_metrics = {
        int(attempts): []
        for attempts in protocol["system_evaluation"][
            "exact_reference_open_loop_attempts"
        ]
    }
    for trajectory in range(trajectory_count):
        semantic_runs = []
        exact_runs = {attempts: [] for attempts in exact_metrics}
        for session in sessions:
            random_values = _session_random(
                common_random[trajectory], session, positions
            )
            semantic_runs.append(
                simulate_r1_trajectory(
                    prepared["states"],
                    fixed["timestamps_local"],
                    session,
                    variant=protocol["frozen_architecture"]["semantic_variant"],
                    sender_session=prepared["sender"],
                    receiver_session=prepared["receiver"],
                    install_packet=prepared["install_packet"],
                    deployment_mode=protocol["frozen_architecture"][
                        "deployment_mode"
                    ],
                    condition=condition,
                    random_values=random_values,
                    epsilon_db=point.epsilon_db,
                    max_age_minutes=float(
                        controller["maximum_state_age_minutes"]
                    ),
                    ack_frame_bits=prepared["ack_bits"],
                    outage_penalty_db=float(
                        protocol["task_grid"]["outage_penalty_db"]
                    ),
                    heartbeat_interval_scenes=int(
                        controller["heartbeat_silence_scenes"]
                    ),
                    heartbeat_request_frame_bits=prepared[
                        "heartbeat_request_bits"
                    ],
                    heartbeat_response_frame_bits=prepared[
                        "heartbeat_response_bits"
                    ],
                    task_open_loop_attempts=int(
                        controller["task_update_open_loop_attempts"]
                    ),
                    compact_codeword_frame_bits=prepared[
                        "compact_frame_bits"
                    ],
                    exact_action_frame_bits=prepared["exact_packet_bits"],
                )
            )
            for attempts in exact_runs:
                exact_runs[attempts].append(
                    simulate_exact_trajectory(
                        prepared["states"],
                        fixed["timestamps_local"],
                        session,
                        random_values=random_values,
                        packet_loss_probability=float(
                            condition["task_loss_probability"]
                        ),
                        receiver_reset_probability=float(
                            condition["receiver_context_reset_probability"]
                        ),
                        task_open_loop_attempts=attempts,
                        packet_bits=prepared["exact_packet_bits"],
                        epsilon_db=point.epsilon_db,
                        outage_penalty_db=float(
                            protocol["task_grid"]["outage_penalty_db"]
                        ),
                    )
                )
        semantic_metrics.append(
            _trajectory_metric(combine_r1_trajectory_results(semantic_runs))
        )
        for attempts, values in exact_runs.items():
            exact_metrics[attempts].append(
                _exact_trajectory_metric(
                    combine_matched_trajectory_results(values)
                )
            )
    evaluated_scenes = int(sum(len(value) for value in sessions))
    return {
        "outer_group_id": group_id,
        "site_id": site_id,
        "scene_count": int(indices.size),
        "status": "evaluated",
        "session_count": len(sessions),
        "evaluated_scene_count": evaluated_scenes,
        "semantic_workpoint": {
            "variant_id": protocol["frozen_architecture"]["semantic_variant"],
            "deployment_mode": protocol["frozen_architecture"][
                "deployment_mode"
            ],
            "summary": _summary(semantic_metrics),
            "trajectory_metrics": semantic_metrics,
        },
        "exact_workpoints": [
            {
                "task_packet_open_loop_attempts": attempts,
                "packet_bits": prepared["exact_packet_bits"],
                "summary": _summary(rows),
                "trajectory_metrics": rows,
            }
            for attempts, rows in sorted(exact_metrics.items())
        ],
    }


def run_preflight(protocol_path: Path, output: Path) -> None:
    protocol, _, _, fixed, pilot = _verified_inputs(protocol_path)
    points = registered_grid_points(protocol)
    value = {
        "version": "1.0",
        "status": "pass",
        "protocol_sha256": sha256_file(protocol_path),
        "registered_grid_point_count": len(points),
        "grid_point_count_by_n": {
            str(n): sum(value.n_channels == int(n) for value in points)
            for n in protocol["task_grid"]["n_channels"]
        },
        "fixed_site_scene_count": int(fixed["outer_group_ids"].size),
        "fixed_site_outer_group_count": int(
            np.unique(fixed["outer_group_ids"].astype(str)).size
        ),
        "pilot_scene_count": int(pilot["cluster_ids"].size),
        "checks": {
            "complete_query_lattice": True,
            "r1_gate_passed": True,
            "frozen_input_hashes_match": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
        },
        "claim_boundary": "Preflight only; no sensitivity result was inspected.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value)


def run_n(n_channels: int, protocol_path: Path, checkpoint_dir: Path) -> None:
    protocol, parent, candidates, fixed, pilot = _verified_inputs(protocol_path)
    points = [
        value
        for value in registered_grid_points(protocol)
        if value.n_channels == int(n_channels)
    ]
    if not points:
        raise ValueError("N is outside the registered sensitivity grid")
    protocol_sha = sha256_file(protocol_path)
    group_ids = sorted(set(fixed["outer_group_ids"].astype(str)))
    n_position = [
        int(value) for value in protocol["task_grid"]["n_channels"]
    ].index(int(n_channels))
    n_dir = checkpoint_dir / f"n{int(n_channels)}"
    n_dir.mkdir(parents=True, exist_ok=True)
    for point_position, point in enumerate(points):
        path = n_dir / f"{point.grid_point_id}.json"
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if (
                checkpoint["protocol_sha256"] != protocol_sha
                or checkpoint["grid_point_id"] != point.grid_point_id
            ):
                raise ValueError(f"checkpoint mismatch: {path}")
            if checkpoint["status"] == "complete":
                continue
        else:
            checkpoint = {
                "version": "1.0",
                "status": "preparing",
                "protocol_sha256": protocol_sha,
                "grid_point_id": point.grid_point_id,
                "n_channels": point.n_channels,
                "epsilon_db": point.epsilon_db,
                "query_set_id": point.query_set_id,
                "configured_outer_group_ids": group_ids,
                "completed_outer_group_ids": [],
                "outer_groups": {},
                "external_final_access_count": 0,
            }
            atomic_write_json(path, checkpoint)
        prepared = _prepare_grid_point(point, protocol, parent, pilot, fixed)
        checkpoint["representation"] = prepared["representation"]
        if not prepared["deployable"]:
            checkpoint["status"] = "complete"
            checkpoint["system_status"] = "not_deployable_in_current_codec"
            atomic_write_json(path, checkpoint)
            print(
                f"S6.7c N={n_channels} point={point.grid_point_id} "
                "non-deployable",
                flush=True,
            )
            continue
        checkpoint["status"] = "in_progress"
        checkpoint["system_status"] = "evaluating"
        completed = set(checkpoint["completed_outer_group_ids"])
        started = time.perf_counter()
        for group_position, group_id in enumerate(group_ids):
            if group_id in completed:
                continue
            checkpoint["outer_groups"][group_id] = _run_group(
                prepared,
                group_id,
                group_position,
                n_position,
                protocol,
                parent,
                candidates,
                fixed,
            )
            checkpoint["completed_outer_group_ids"].append(group_id)
            checkpoint["elapsed_seconds_current_process"] = float(
                time.perf_counter() - started
            )
            atomic_write_json(path, checkpoint)
            print(
                f"S6.7c N={n_channels} point={point_position + 1}/{len(points)} "
                f"group={len(checkpoint['completed_outer_group_ids'])}/{len(group_ids)}",
                flush=True,
            )
        checkpoint["status"] = "complete"
        checkpoint["system_status"] = "evaluated"
        atomic_write_json(path, checkpoint)


def merge(
    protocol_path: Path,
    checkpoint_dir: Path,
    output: Path,
    representation_output: Path,
) -> None:
    protocol, _, _, _, _ = _verified_inputs(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    results: dict[str, dict] = {}
    representation = []
    wrong_actions = 0.0
    for point in registered_grid_points(protocol):
        path = (
            checkpoint_dir
            / f"n{point.n_channels}"
            / f"{point.grid_point_id}.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["status"] != "complete"
            or value["protocol_sha256"] != protocol_sha
        ):
            raise ValueError(f"incomplete sensitivity checkpoint: {path}")
        results.setdefault(str(point.n_channels), {})[point.grid_point_id] = value
        representation.append(value["representation"])
        for group in value["outer_groups"].values():
            if group.get("status") == "evaluated":
                wrong_actions += float(
                    group["semantic_workpoint"]["summary"][
                        "wrong_codebook_decode_count"
                    ]
                )
    result = {
        "version": "1.0",
        "status": "task_scale_sensitivity_grouped_complete",
        "protocol_sha256": protocol_sha,
        "results": results,
        "checks": {
            "registered_grid_point_count": len(representation),
            "all_grid_points_retained": len(representation)
            == len(registered_grid_points(protocol)),
            "wrong_codebook_actions_must_be_zero": wrong_actions == 0.0,
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
        },
        "environment": environment_snapshot(PROJECT_DIR),
        "claim_boundary": protocol["claim_boundary"],
    }
    representation_result = {
        "version": "1.0",
        "status": "task_scale_representation_complete",
        "protocol_sha256": protocol_sha,
        "grid_points": representation,
        "checks": result["checks"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    representation_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    atomic_write_json(representation_output, representation_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--run-n", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--representation-output", type=Path, default=DEFAULT_REPRESENTATION
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = sum((args.preflight, args.run_n is not None, args.merge))
    if selected != 1:
        raise SystemExit("select exactly one of --preflight, --run-n, or --merge")
    if args.preflight:
        run_preflight(args.protocol, args.preflight_output)
    elif args.run_n is not None:
        run_n(args.run_n, args.protocol, args.checkpoint_dir)
    else:
        merge(
            args.protocol,
            args.checkpoint_dir,
            args.output,
            args.representation_output,
        )


if __name__ == "__main__":
    main()
