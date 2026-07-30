#!/usr/bin/env python
"""Run the registered S6R.3g fixed silent-heartbeat development scan."""

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

from run_stage6r_activation_protocol_dry_run import (  # noqa: E402
    _compact_bits,
    _mean_ci,
    _metric_arrays,
    _paired_savings,
)
from run_stage6r_few_shot_adaptation import _load_combined  # noqa: E402
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
    role_members,
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
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
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


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_silent_heartbeat_scan_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/silent_heartbeat_scan_v1/result.json"
)


def _summary(arrays: dict, *, seed: int, config: dict) -> dict:
    return {
        name: _mean_ci(
            values,
            seed=seed + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
        )
        for position, (name, values) in enumerate(arrays.items())
    }


def _hierarchical_group(
    semantic_rows: list[dict[str, np.ndarray]],
    exact_rows: list[dict[str, np.ndarray]],
    scene_counts: list[int],
    *,
    seed: int,
    config: dict,
) -> dict:
    weights = np.asarray(scene_counts, dtype=np.float64)
    semantic = {
        name: hierarchical_mean_ci(
            np.stack([row[name] for row in semantic_rows]),
            seed=seed + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
            site_weights=weights,
        )
        for position, name in enumerate(semantic_rows[0])
    }
    exact = {
        name: hierarchical_mean_ci(
            np.stack([row[name] for row in exact_rows]),
            seed=seed + 100_000 + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
            site_weights=weights,
        )
        for position, name in enumerate(exact_rows[0])
    }
    savings = hierarchical_mean_ci(
        np.stack(
            [
                _paired_savings(semantic_row, exact_row)
                for semantic_row, exact_row in zip(
                    semantic_rows, exact_rows
                )
            ]
        ),
        seed=seed + 200_000,
        replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
        confidence=float(config["monte_carlo"]["confidence_level"]),
        site_weights=weights,
    )
    return {
        "semantic": semantic,
        "exact": exact,
        "paired_savings_percentage": savings,
    }


def _make_session(
    *,
    session_id: str,
    domain: str,
    power: np.ndarray,
    timestamps: np.ndarray,
    queries: tuple,
    epsilon: float,
    resolution: dict,
) -> dict:
    states = [
        build_task_state(row, queries, epsilon_db=epsilon)
        for row in power
    ]
    return {
        "session_id": session_id,
        "domain": domain,
        "states": states,
        "timestamps": np.asarray(timestamps).astype(str),
        "decision": resolution["decision"],
        "calibration_scenes": int(resolution["calibration_scenes"]),
        "selected_actions": resolution["selected_actions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite heartbeat scan result")
    config = read_json(args.config)
    paths = {
        name: PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    adaptation = read_json(paths["adaptation_config"])
    temporal = read_json(paths["temporal_gate_result"])
    scan_config = read_json(paths["matched_scan_config"])
    scan = read_json(paths["matched_scan_result"])
    activation_protocol = read_json(paths["activation_protocol"])
    electro_protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    if access["roles"]["pilot"]["access_count"] != 1:
        raise ValueError("Pilot development role is unavailable")
    if access["roles"]["stage6_final"]["access_count"] != 1:
        raise ValueError("Final failure evidence must already be frozen")
    if any(
        access["roles"][role]["access_count"] != 0
        for role in ("confirmation_lockbox", "reserve")
    ):
        raise ValueError("lockbox or reserve values were accessed")
    if config["frozen_task"]["n_channels"] != [16, 32]:
        raise ValueError("this registered scan is restricted to N=16/32")
    if config["frozen_task"]["demand_ratios"] != adaptation["task"][
        "demand_ratios"
    ]:
        raise ValueError("task demand ratios changed")
    if abs(
        float(config["frozen_task"]["epsilon_db"])
        - float(scan_config["task"]["epsilon_db"])
    ) > 1e-12:
        raise ValueError("epsilon changed")
    if int(
        activation_protocol["fail_closed"][
            "maximum_activation_attempts_before_full_install_fallback"
        ]
    ) != int(config["frozen_task"]["maximum_activation_attempts"]):
        raise ValueError("activation attempt limit changed")
    if electro_protocol["data_adapter"]["normalization"] != "none":
        raise ValueError("ElectroSense adapter changed")

    pilot_metadata = role_members(registry, "pilot")
    archive = Path(electro_protocol["inputs"]["archive"])
    pilot_arrays = load_npy_members(
        archive, [row["member"] for row in pilot_metadata]
    )
    condition = scan_config["conditions"]["primary_fault"]
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(config["monte_carlo"]["random_seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    epsilon = float(config["frozen_task"]["epsilon_db"])
    codebook_epoch = int(config["frozen_task"]["codebook_epoch"])
    catalog_epoch = int(config["frozen_task"]["catalog_epoch"])
    maximum_activation_attempts = int(
        config["frozen_task"]["maximum_activation_attempts"]
    )
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
    heartbeat_candidates = list(
        map(int, config["intervention"]["heartbeat_candidates_scenes"])
    )
    started = time.perf_counter()
    n_results = {}
    for n_position, n_channels in enumerate(
        map(int, config["frozen_task"]["n_channels"])
    ):
        queries = task_queries(
            n_channels, config["frozen_task"]["demand_ratios"]
        )
        development_power, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == int(config["frozen_task"]["primary_k"])
        }
        sessions = []
        for campaign in config["development_data"]["aerpaw_activities"]:
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[
                np.argsort(timestamps[indices], kind="stable")
            ]
            row = temporal_rows[campaign]
            sessions.append(
                _make_session(
                    session_id=campaign,
                    domain="aerpaw",
                    power=development_power[indices],
                    timestamps=timestamps[indices],
                    queries=queries,
                    epsilon=epsilon,
                    resolution={
                        "decision": row["decision"],
                        "calibration_scenes": row["calibration_scenes"],
                        "selected_actions": row["selected_actions"],
                    },
                )
            )
        pilot_resolution = {
            row["site"]: row
            for row in freeze["n_artifacts"][str(n_channels)][
                "pilot_adapter_gate"
            ]
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
                    timestamps=synthetic_six_hour_timestamps(
                        power.shape[0]
                    ),
                    queries=queries,
                    epsilon=epsilon,
                    resolution=pilot_resolution[metadata["site"]],
                )
            )
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        workpoints = frozen_n["workpoints"]
        semantic_wp = workpoints["semantic"]
        exact_wp = workpoints["exact"]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(scan_config["task"]["exact_action_header_bits"]),
        )
        per_candidate = {
            heartbeat: {
                "semantic_arrays": [],
                "exact_arrays": [],
                "scene_counts": [],
                "domains": [],
                "session_results": {},
            }
            for heartbeat in heartbeat_candidates
        }
        for session_position, session in enumerate(sessions):
            states = session["states"]
            timestamps_local = session["timestamps"]
            indices = np.arange(len(states), dtype=np.int64)
            random = np.random.default_rng(
                base_seed
                + n_position * 1_000_000
                + session_position * 10_000
            ).random((trajectories, len(states), RANDOM_STREAM_COUNT))
            exact_runs = [
                simulate_exact_trajectory(
                    states,
                    timestamps_local,
                    indices,
                    random_values=random[trajectory],
                    packet_loss_probability=float(
                        condition["task_loss_probability"]
                    ),
                    receiver_reset_probability=float(
                        condition["receiver_context_reset_probability"]
                    ),
                    task_open_loop_attempts=int(
                        exact_wp["task_packet_open_loop_attempts"]
                    ),
                    packet_bits=exact_packet_bits,
                    epsilon_db=epsilon,
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                )
                for trajectory in range(trajectories)
            ]
            exact_arrays = _metric_arrays(exact_runs)
            unresolved = (
                session["decision"] == "UNRESOLVED_EXACT_FALLBACK"
            )
            if not unresolved:
                calibration_count = int(session["calibration_scenes"])
                calibration_indices = indices[:calibration_count]
                future_indices = indices[calibration_count:]
                codebook = codebook_from_selected_actions(
                    states[:calibration_count],
                    session["selected_actions"],
                )
                sender = install_codebook(codebook, epoch=codebook_epoch)
                full_packet = encode_context_install(sender, node_id=1)
                receiver = decode_context_install(full_packet).session
                compact_bits = _compact_bits(states, indices, sender)
                is_bank = str(session["decision"]).startswith("BANK:")
                if is_bank:
                    source = str(session["decision"]).removeprefix(
                        "BANK:"
                    )
                    bank_id = int(frozen_n["bank_ids"][source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)], catalog_epoch=catalog_epoch
                    )
                else:
                    bank_id = None
                    catalog = None
            for heartbeat_position, heartbeat in enumerate(
                heartbeat_candidates
            ):
                if unresolved:
                    semantic_runs = exact_runs
                else:
                    semantic_runs = []
                    for trajectory in range(trajectories):
                        calibration = simulate_exact_trajectory(
                            states,
                            timestamps_local,
                            calibration_indices,
                            random_values=random[
                                trajectory, :calibration_count
                            ],
                            packet_loss_probability=float(
                                condition["task_loss_probability"]
                            ),
                            receiver_reset_probability=float(
                                condition[
                                    "receiver_context_reset_probability"
                                ]
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp[
                                    "calibration_open_loop_attempts"
                                ]
                            ),
                            packet_bits=exact_packet_bits,
                            epsilon_db=epsilon,
                            outage_penalty_db=float(
                                scan_config["task"]["outage_penalty_db"]
                            ),
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
                            maximum_activation_attempts=(
                                maximum_activation_attempts
                            ),
                            condition=condition,
                            random_values=random[
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
                            heartbeat_request_frame_bits=(
                                heartbeat_request_bits
                            ),
                            heartbeat_response_frame_bits=(
                                heartbeat_response_bits
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp[
                                    "task_update_open_loop_attempts"
                                ]
                            ),
                            compact_codeword_frame_bits=compact_bits,
                        )
                        semantic_runs.append(
                            combine_matched_trajectory_results(
                                [calibration, suffix.matched]
                            )
                        )
                semantic_arrays = _metric_arrays(semantic_runs)
                bucket = per_candidate[heartbeat]
                bucket["semantic_arrays"].append(semantic_arrays)
                bucket["exact_arrays"].append(exact_arrays)
                bucket["scene_counts"].append(len(states))
                bucket["domains"].append(session["domain"])
                bucket["session_results"][session["session_id"]] = {
                    "domain": session["domain"],
                    "decision": session["decision"],
                    "scene_count": len(states),
                    "semantic": _summary(
                        semantic_arrays,
                        seed=bootstrap_seed
                        + n_position * 1_000_000
                        + heartbeat_position * 100_000
                        + session_position * 100,
                        config=config,
                    ),
                    "paired_savings_percentage": _mean_ci(
                        _paired_savings(
                            semantic_arrays, exact_arrays
                        ),
                        seed=bootstrap_seed
                        + n_position * 1_000_000
                        + heartbeat_position * 100_000
                        + session_position * 100
                        + 71,
                        replicates=int(
                            config["monte_carlo"][
                                "bootstrap_replicates"
                            ]
                        ),
                        confidence=float(
                            config["monte_carlo"]["confidence_level"]
                        ),
                    ),
                }
            print(
                f"Heartbeat scan N={n_channels} "
                f"session={session['session_id']} complete",
                flush=True,
            )
        candidate_results = {}
        for heartbeat_position, heartbeat in enumerate(
            heartbeat_candidates
        ):
            bucket = per_candidate[heartbeat]
            domain_indices = {
                domain: [
                    index
                    for index, value in enumerate(bucket["domains"])
                    if value == domain
                ]
                for domain in ("aerpaw", "electrosense_pilot")
            }
            groups = {}
            for group_name, selected in {
                "combined": list(range(len(sessions))),
                "aerpaw": domain_indices["aerpaw"],
                "pilot": domain_indices["electrosense_pilot"],
            }.items():
                groups[group_name] = _hierarchical_group(
                    [
                        bucket["semantic_arrays"][index]
                        for index in selected
                    ],
                    [
                        bucket["exact_arrays"][index]
                        for index in selected
                    ],
                    [bucket["scene_counts"][index] for index in selected],
                    seed=bootstrap_seed
                    + n_position * 1_000_000
                    + heartbeat_position * 100_000
                    + {
                        "combined": 7001,
                        "aerpaw": 8001,
                        "pilot": 9001,
                    }[group_name],
                    config=config,
                )
            gate = config["selection_gate"]
            checks = {
                "combined_clean_lower_pass": (
                    groups["combined"]["semantic"]["clean_rate"][
                        "ci_lower"
                    ]
                    >= float(
                        gate[
                            "combined_scene_weighted_clean_ci_lower_minimum"
                        ]
                    )
                ),
                "pilot_clean_lower_pass": (
                    groups["pilot"]["semantic"]["clean_rate"]["ci_lower"]
                    >= float(
                        gate[
                            "pilot_scene_weighted_clean_ci_lower_minimum"
                        ]
                    )
                ),
                "combined_savings_lower_pass": (
                    groups["combined"]["paired_savings_percentage"][
                        "ci_lower"
                    ]
                    > float(
                        gate[
                            "combined_scene_weighted_savings_ci_lower_minimum"
                        ]
                    )
                ),
                "pilot_savings_lower_pass": (
                    groups["pilot"]["paired_savings_percentage"][
                        "ci_lower"
                    ]
                    > float(
                        gate[
                            "pilot_scene_weighted_savings_ci_lower_minimum"
                        ]
                    )
                ),
                "wrong_codebook_decode_count_zero": all(
                    float(
                        np.max(
                            arrays["wrong_codebook_decode_count"]
                        )
                    )
                    == 0.0
                    for arrays in bucket["semantic_arrays"]
                ),
            }
            checks["candidate_passed"] = all(checks.values())
            candidate_results[str(heartbeat)] = {
                "heartbeat_silence_scenes": heartbeat,
                "groups": groups,
                "session_results": bucket["session_results"],
                "checks": checks,
            }
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "candidate_results": candidate_results,
        }

    selected = None
    for heartbeat in heartbeat_candidates:
        if all(
            n_results[str(n)]["candidate_results"][str(heartbeat)][
                "checks"
            ]["candidate_passed"]
            for n in config["selection_gate"]["required_n_values"]
        ):
            selected = heartbeat
            break
    result = {
        "version": "1.0",
        "status": "stage6r_silent_heartbeat_scan_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "access_counts_after_run": {
            role: read_json(paths["electrosense_access_state"])["roles"][
                role
            ]["access_count"]
            for role in (
                "pilot",
                "stage6_final",
                "confirmation_lockbox",
                "reserve",
            )
        },
        "n_results": n_results,
        "selection": {
            "selected_heartbeat_silence_scenes": selected,
            "selection_succeeded": selected is not None,
            "next_action": (
                "freeze_fixed_heartbeat_candidate"
                if selected is not None
                else config["selection_gate"]["if_none_pass"]
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(
        {
            "output": str(args.output),
            "selection": result["selection"],
            "access_counts_after_run": result["access_counts_after_run"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
