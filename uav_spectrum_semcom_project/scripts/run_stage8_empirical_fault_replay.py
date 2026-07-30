#!/usr/bin/env python
"""Replay fixed heartbeat intervals under the frozen Stage-8 fault model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage6r_activation_protocol_dry_run import (  # noqa: E402
    _compact_bits,
    _mean_ci,
    _metric_arrays,
)
from run_stage6r_few_shot_adaptation import _load_combined  # noqa: E402
from run_stage6r_silent_heartbeat_scan import _make_session  # noqa: E402
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
    role_members,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
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
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6r_activation_reliability import (  # noqa: E402
    simulate_activation_semantic_trajectory,
)
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    hierarchical_mean_ci,
    synthetic_six_hour_timestamps,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)


DEFAULT_CONFIG = ROOT / "configs/stage8_empirical_fault_replay_v1.json"
DEFAULT_MODEL = ROOT / "results/stage8/stage8_empirical_fault_model_v1/result.json"
DEFAULT_OUTPUT = ROOT / "results/stage8/stage8_empirical_fault_replay_v1/result.json"


def heartbeat_key(value: int | None) -> str:
    return "none" if value is None else str(int(value))


def paired_group_summary(
    candidate_rows: list[dict[str, np.ndarray]],
    baseline_rows: list[dict[str, np.ndarray]],
    scene_counts: list[int],
    *,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    weights = np.asarray(scene_counts, dtype=np.float64)
    clean_difference_pp = np.stack(
        [
            100.0 * (candidate["clean_rate"] - baseline["clean_rate"])
            for candidate, baseline in zip(candidate_rows, baseline_rows)
        ]
    )
    availability_difference_pp = np.stack(
        [
            100.0
            * (candidate["availability_rate"] - baseline["availability_rate"])
            for candidate, baseline in zip(candidate_rows, baseline_rows)
        ]
    )
    bit_saving_percent = np.stack(
        [
            100.0
            * (
                baseline["total_bits_per_scene"]
                - candidate["total_bits_per_scene"]
            )
            / baseline["total_bits_per_scene"]
            for candidate, baseline in zip(candidate_rows, baseline_rows)
        ]
    )
    settings = config["monte_carlo"]
    return {
        "clean_difference_percentage_points": hierarchical_mean_ci(
            clean_difference_pp,
            seed=seed,
            replicates=int(settings["bootstrap_replicates"]),
            confidence=float(settings["confidence_level"]),
            site_weights=weights,
        ),
        "availability_difference_percentage_points": hierarchical_mean_ci(
            availability_difference_pp,
            seed=seed + 1009,
            replicates=int(settings["bootstrap_replicates"]),
            confidence=float(settings["confidence_level"]),
            site_weights=weights,
        ),
        "relative_total_bit_saving_percentage": hierarchical_mean_ci(
            bit_saving_percent,
            seed=seed + 2018,
            replicates=int(settings["bootstrap_replicates"]),
            confidence=float(settings["confidence_level"]),
            site_weights=weights,
        ),
    }


def run(config_path: Path, model_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite Stage-8 replay")
    config = read_json(config_path)
    model = read_json(model_path)
    if model["status"] != "EMPIRICAL_FAULT_MODEL_FROZEN_BEFORE_CONFIRMATION_ACCESS":
        raise ValueError("fault model is not frozen")
    paths = {name: ROOT / value for name, value in config["inputs"].items()}
    split = read_json(paths["stage8_split"])
    if model["confirmation_access_at_freeze"] != {
        "aadm_flights_opened": 0,
        "alfa_raw_flights_opened": 0,
    }:
        raise ValueError("confirmation access occurred before model freeze")
    if split["signal_access"]["confirmation_value_access_count"] != 0:
        raise ValueError("split manifest reports confirmation access")

    adaptation = read_json(paths["adaptation_config"])
    temporal = read_json(paths["temporal_gate_result"])
    scan_config = read_json(paths["matched_scan_config"])
    activation_protocol = read_json(paths["activation_protocol"])
    electro_protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    if access["roles"]["pilot"]["access_count"] != 1:
        raise ValueError("ElectroSense pilot development values are unavailable")
    if access["roles"]["confirmation_lockbox"]["access_count"] != 1:
        raise ValueError("historical Stage-6 confirmation access ledger changed")
    if access["roles"]["reserve"]["access_count"] != 0:
        raise ValueError("Stage-6 reserve values were accessed")
    if electro_protocol["data_adapter"]["normalization"] != "none":
        raise ValueError("ElectroSense adapter changed")

    heartbeat_values = [
        int(config["intervention"]["baseline_heartbeat_scenes"]),
        *config["intervention"]["candidate_heartbeat_scenes"],
    ]
    unique_heartbeats: list[int | None] = []
    for value in heartbeat_values:
        normalized = None if value is None else int(value)
        if normalized not in unique_heartbeats:
            unique_heartbeats.append(normalized)
    baseline_heartbeat = int(config["intervention"]["baseline_heartbeat_scenes"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(config["monte_carlo"]["random_seed"])
    fault_seed = int(model["task_link_model"]["random_seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    task_rates = np.asarray(
        [
            row["loss_probability_lower_bound"]
            for row in model["task_link_model"]["flight_rates"]
        ],
        dtype=np.float64,
    )
    condition_base = dict(scan_config["conditions"]["primary_fault"])
    epsilon = float(config["frozen_task"]["epsilon_db"])
    codebook_epoch = int(config["frozen_task"]["codebook_epoch"])
    catalog_epoch = int(config["frozen_task"]["catalog_epoch"])
    maximum_activation_attempts = int(
        config["frozen_task"]["maximum_activation_attempts"]
    )
    if int(
        activation_protocol["fail_closed"][
            "maximum_activation_attempts_before_full_install_fallback"
        ]
    ) != maximum_activation_attempts:
        raise ValueError("activation fail-closed limit changed")
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

    pilot_metadata = role_members(registry, "pilot")
    pilot_arrays = load_npy_members(
        Path(electro_protocol["inputs"]["archive"]),
        [row["member"] for row in pilot_metadata],
    )
    started = time.perf_counter()
    n_results: dict[str, Any] = {}
    for n_position, n_channels in enumerate(
        map(int, config["frozen_task"]["n_channels"])
    ):
        queries = task_queries(n_channels, config["frozen_task"]["demand_ratios"])
        development_power, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == int(config["frozen_task"]["primary_k"])
        }
        sessions = []
        for campaign in config["development_spectrum_sessions"][
            "aerpaw_activities"
        ]:
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            sessions.append(
                _make_session(
                    session_id=campaign,
                    domain="aerpaw",
                    power=development_power[indices],
                    timestamps=timestamps[indices],
                    queries=queries,
                    epsilon=epsilon,
                    resolution={
                        "decision": temporal_rows[campaign]["decision"],
                        "calibration_scenes": temporal_rows[campaign][
                            "calibration_scenes"
                        ],
                        "selected_actions": temporal_rows[campaign][
                            "selected_actions"
                        ],
                    },
                )
            )
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        pilot_resolution = {
            row["site"]: row for row in frozen_n["pilot_adapter_gate"]
        }
        for metadata in pilot_metadata:
            power = aggregate_frequency_bins(
                pilot_arrays[metadata["member"]], n_channels
            )
            sessions.append(
                _make_session(
                    session_id=metadata["site"],
                    domain="electrosense_pilot",
                    power=power,
                    timestamps=synthetic_six_hour_timestamps(power.shape[0]),
                    queries=queries,
                    epsilon=epsilon,
                    resolution=pilot_resolution[metadata["site"]],
                )
            )

        semantic_wp = frozen_n["workpoints"]["semantic"]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(scan_config["task"]["exact_action_header_bits"]),
        )
        per_heartbeat = {
            heartbeat_key(value): {
                "arrays": [],
                "scene_counts": [],
                "domains": [],
                "session_summaries": {},
            }
            for value in unique_heartbeats
        }
        sampled_rate_rows = {}
        for session_position, session in enumerate(sessions):
            states = session["states"]
            timestamps_local = session["timestamps"]
            indices = np.arange(len(states), dtype=np.int64)
            random_values = np.random.default_rng(
                base_seed + n_position * 1_000_000 + session_position * 10_000
            ).random((trajectories, len(states), RANDOM_STREAM_COUNT))
            rate_indices = np.random.default_rng(
                fault_seed + session_position * 10_000
            ).integers(0, task_rates.size, size=trajectories)
            sampled_rates = task_rates[rate_indices]
            sampled_rate_rows[session["session_id"]] = {
                "mean": float(np.mean(sampled_rates)),
                "minimum": float(np.min(sampled_rates)),
                "maximum": float(np.max(sampled_rates)),
                "sampled_flight_rate_indices": rate_indices.tolist(),
            }

            unresolved = session["decision"] == "UNRESOLVED_EXACT_FALLBACK"
            if not unresolved:
                calibration_count = int(session["calibration_scenes"])
                calibration_indices = indices[:calibration_count]
                future_indices = indices[calibration_count:]
                codebook = codebook_from_selected_actions(
                    states[:calibration_count], session["selected_actions"]
                )
                sender = install_codebook(codebook, epoch=codebook_epoch)
                full_packet = encode_context_install(sender, node_id=1)
                receiver = decode_context_install(full_packet).session
                compact_bits = _compact_bits(states, indices, sender)
                if str(session["decision"]).startswith("BANK:"):
                    source = str(session["decision"]).removeprefix("BANK:")
                    bank_id = int(frozen_n["bank_ids"][source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)], catalog_epoch=catalog_epoch
                    )
                else:
                    bank_id = None
                    catalog = None

            calibration_runs = []
            exact_full_runs = []
            for trajectory in range(trajectories):
                condition = dict(condition_base)
                condition["task_loss_probability"] = float(
                    sampled_rates[trajectory]
                )
                if unresolved:
                    exact_full_runs.append(
                        simulate_exact_trajectory(
                            states,
                            timestamps_local,
                            indices,
                            random_values=random_values[trajectory],
                            packet_loss_probability=float(
                                condition["task_loss_probability"]
                            ),
                            receiver_reset_probability=float(
                                condition["receiver_context_reset_probability"]
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp["calibration_open_loop_attempts"]
                            ),
                            packet_bits=exact_packet_bits,
                            epsilon_db=epsilon,
                            outage_penalty_db=float(
                                scan_config["task"]["outage_penalty_db"]
                            ),
                        )
                    )
                else:
                    calibration_runs.append(
                        simulate_exact_trajectory(
                            states,
                            timestamps_local,
                            calibration_indices,
                            random_values=random_values[
                                trajectory, :calibration_count
                            ],
                            packet_loss_probability=float(
                                condition["task_loss_probability"]
                            ),
                            receiver_reset_probability=float(
                                condition["receiver_context_reset_probability"]
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp["calibration_open_loop_attempts"]
                            ),
                            packet_bits=exact_packet_bits,
                            epsilon_db=epsilon,
                            outage_penalty_db=float(
                                scan_config["task"]["outage_penalty_db"]
                            ),
                        )
                    )

            for heartbeat_position, heartbeat in enumerate(unique_heartbeats):
                if unresolved:
                    runs = exact_full_runs
                else:
                    runs = []
                    for trajectory in range(trajectories):
                        condition = dict(condition_base)
                        condition["task_loss_probability"] = float(
                            sampled_rates[trajectory]
                        )
                        suffix = simulate_activation_semantic_trajectory(
                            states,
                            timestamps_local,
                            future_indices,
                            sender_session=sender,
                            receiver_session=receiver,
                            full_install_packet=full_packet,
                            sender_catalog=catalog,
                            receiver_catalog=catalog,
                            bank_id=bank_id,
                            catalog_node_id=1,
                            maximum_activation_attempts=maximum_activation_attempts,
                            condition=condition,
                            random_values=random_values[
                                trajectory, calibration_count:
                            ],
                            epsilon_db=epsilon,
                            max_age_minutes=float(
                                semantic_wp["maximum_state_age_minutes"]
                            ),
                            ack_frame_bits=ack_bits,
                            outage_penalty_db=float(
                                scan_config["task"]["outage_penalty_db"]
                            ),
                            heartbeat_interval_scenes=heartbeat,
                            heartbeat_request_frame_bits=heartbeat_request_bits,
                            heartbeat_response_frame_bits=heartbeat_response_bits,
                            task_open_loop_attempts=int(
                                semantic_wp["task_update_open_loop_attempts"]
                            ),
                            compact_codeword_frame_bits=compact_bits,
                        )
                        runs.append(
                            combine_matched_trajectory_results(
                                [calibration_runs[trajectory], suffix.matched]
                            )
                        )
                arrays = _metric_arrays(runs)
                bucket = per_heartbeat[heartbeat_key(heartbeat)]
                bucket["arrays"].append(arrays)
                bucket["scene_counts"].append(len(states))
                bucket["domains"].append(session["domain"])
                bucket["session_summaries"][session["session_id"]] = {
                    "domain": session["domain"],
                    "decision": session["decision"],
                    "scene_count": len(states),
                    "metrics": {
                        name: _mean_ci(
                            values,
                            seed=bootstrap_seed
                            + n_position * 1_000_000
                            + heartbeat_position * 100_000
                            + session_position * 100
                            + metric_position,
                            replicates=int(
                                config["monte_carlo"]["bootstrap_replicates"]
                            ),
                            confidence=float(
                                config["monte_carlo"]["confidence_level"]
                            ),
                        )
                        for metric_position, (name, values) in enumerate(
                            arrays.items()
                        )
                    },
                }
            print(
                f"Stage8 replay N={n_channels} session={session['session_id']} complete",
                flush=True,
            )

        baseline = per_heartbeat[heartbeat_key(baseline_heartbeat)]
        candidate_results = {}
        for candidate_position, candidate in enumerate(
            config["intervention"]["candidate_heartbeat_scenes"]
        ):
            candidate = None if candidate is None else int(candidate)
            bucket = per_heartbeat[heartbeat_key(candidate)]
            groups = {}
            for group_position, (group_name, selected) in enumerate(
                {
                    "combined": list(range(len(sessions))),
                    "aerpaw": [
                        index
                        for index, domain in enumerate(bucket["domains"])
                        if domain == "aerpaw"
                    ],
                    "pilot": [
                        index
                        for index, domain in enumerate(bucket["domains"])
                        if domain == "electrosense_pilot"
                    ],
                }.items()
            ):
                groups[group_name] = paired_group_summary(
                    [bucket["arrays"][index] for index in selected],
                    [baseline["arrays"][index] for index in selected],
                    [bucket["scene_counts"][index] for index in selected],
                    seed=bootstrap_seed
                    + n_position * 1_000_000
                    + candidate_position * 100_000
                    + group_position * 10_000,
                    config=config,
                )
            combined = groups["combined"]
            gate = config["selection_gate"]
            wrong_decode_zero = all(
                float(np.max(arrays["wrong_codebook_decode_count"])) == 0.0
                for arrays in bucket["arrays"]
            )
            checks = {
                "clean_difference_ci_lower_pass": (
                    combined["clean_difference_percentage_points"]["ci_lower"]
                    >= float(
                        gate[
                            "clean_difference_ci_lower_percentage_points_minimum"
                        ]
                    )
                ),
                "relative_total_bit_saving_ci_lower_pass": (
                    combined["relative_total_bit_saving_percentage"]["ci_lower"]
                    > float(
                        gate[
                            "relative_total_bit_saving_ci_lower_strictly_greater_than"
                        ]
                    )
                ),
                "wrong_codebook_decode_count_zero": wrong_decode_zero,
            }
            checks["n_passed"] = all(checks.values())
            candidate_results[heartbeat_key(candidate)] = {
                "heartbeat_silence_scenes": candidate,
                "groups": groups,
                "checks": checks,
                "session_summaries": bucket["session_summaries"],
            }
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "baseline_heartbeat_scenes": baseline_heartbeat,
            "sampled_task_loss_rates": sampled_rate_rows,
            "candidate_results": candidate_results,
        }

    selection_rows = {}
    selected = "NO_CANDIDATE"
    for candidate in config["intervention"]["selection_order"]:
        normalized = None if candidate is None else int(candidate)
        key = heartbeat_key(normalized)
        passing_n = [
            n
            for n in config["frozen_task"]["n_channels"]
            if n_results[str(n)]["candidate_results"][key]["checks"]["n_passed"]
        ]
        selection_rows[key] = {
            "heartbeat_silence_scenes": normalized,
            "passing_n_values": passing_n,
            "passing_n_count": len(passing_n),
            "development_gate_passed": (
                len(passing_n)
                >= int(config["selection_gate"]["minimum_passing_n_values"])
            ),
        }
        if (
            selected == "NO_CANDIDATE"
            and selection_rows[key]["development_gate_passed"]
        ):
            selected = normalized

    result = {
        "version": "1.0",
        "status": "STAGE8_EMPIRICAL_FAULT_DEVELOPMENT_REPLAY_COMPLETE",
        "verification_status": "DEVELOPMENT_ONLY",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(model_path),
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "access_log": {
            "aadm_development_model_used": True,
            "aadm_confirmation_values_opened": 0,
            "alfa_confirmation_values_opened": 0,
            "stage6_main_final_rerun": False,
            "stage6_confirmation_lockbox_rerun": False,
            "stage7_closed_test_opened": False,
            "stage7_reserve_opened": False,
        },
        "fault_model": model["task_link_model"],
        "fixed_stress_parameters": model["fixed_stress_parameters"],
        "n_results": n_results,
        "selection": {
            "baseline_heartbeat_scenes": baseline_heartbeat,
            "candidate_rows": selection_rows,
            "selected_development_candidate": selected,
            "new_candidate_frozen": False,
            "reason": (
                "A development selection, if any, still requires one-time "
                "AADM confirmation replay before it can become a Stage-8 candidate."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.model, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection": result["selection"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
