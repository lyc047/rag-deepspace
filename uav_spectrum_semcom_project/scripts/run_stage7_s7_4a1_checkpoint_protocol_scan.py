#!/usr/bin/env python
"""Run the S7.4A-1 full-protocol checkpoint interval scan."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_activation_protocol_dry_run import _compact_bits  # noqa: E402
from run_stage7_s7_3b_full_protocol_oracle import (  # noqa: E402
    _hierarchical,
    _trajectory_metrics,
)
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
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
from spectrum_semcom.stage6r_activation_reliability import (  # noqa: E402
    ActivationMatchedTrajectoryResult,
    simulate_activation_semantic_trajectory,
)
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    resolve_context_and_actions,
    synthetic_six_hour_timestamps,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage7_s7_4a1_checkpoint_protocol_scan_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_4a1_checkpoint_protocol_scan_v1/result.json"
)


def _metrics(
    matched,
    activated: ActivationMatchedTrajectoryResult | None,
) -> dict[str, float]:
    output = _trajectory_metrics(matched)
    scenes = len(matched.clean)
    output.update(
        {
            "checkpoint_count_per_scene": (
                float(activated.checkpoint_count / scenes)
                if activated is not None
                else 0.0
            ),
            "checkpoint_ack_count_per_scene": (
                float(activated.checkpoint_ack_count / scenes)
                if activated is not None
                else 0.0
            ),
            "checkpoint_rejection_count_per_scene": (
                float(activated.checkpoint_rejection_count / scenes)
                if activated is not None
                else 0.0
            ),
            "checkpoint_bits_per_scene": (
                float(activated.checkpoint_bits / scenes)
                if activated is not None
                else 0.0
            ),
            "checkpoint_escape_bits_per_scene": (
                float(activated.checkpoint_escape_bits / scenes)
                if activated is not None
                else 0.0
            ),
        }
    )
    return output


def _arrays(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def _policy_summary(
    site_arrays: dict[str, list[dict[str, np.ndarray]]],
    scene_counts: list[int],
    *,
    config: dict,
    seed: int,
) -> dict:
    weights = np.asarray(scene_counts, dtype=np.float64)
    output = {}
    for policy_position, (policy, rows) in enumerate(site_arrays.items()):
        output[policy] = {}
        for metric_position, metric in enumerate(rows[0]):
            output[policy][metric] = _hierarchical(
                [row[metric] for row in rows],
                seed=(
                    seed
                    + policy_position * 100_000
                    + metric_position * 1009
                ),
                config=config,
                site_weights=weights,
            )
    return output


def _paired_comparisons(
    site_arrays: dict[str, list[dict[str, np.ndarray]]],
    scene_counts: list[int],
    *,
    config: dict,
    seed: int,
) -> dict:
    reference = str(config["system_gate"]["reference_policy"])
    baseline = site_arrays[reference]
    weights = np.asarray(scene_counts, dtype=np.float64)
    output = {}
    for position, (policy, rows) in enumerate(site_arrays.items()):
        if policy == reference:
            continue
        clean = [
            100.0 * (candidate["clean_rate"] - fixed["clean_rate"])
            for fixed, candidate in zip(baseline, rows)
        ]
        availability = [
            100.0
            * (
                candidate["availability_rate"]
                - fixed["availability_rate"]
            )
            for fixed, candidate in zip(baseline, rows)
        ]
        bit_reduction = [
            100.0
            * (
                fixed["total_bits_per_scene"]
                - candidate["total_bits_per_scene"]
            )
            / fixed["total_bits_per_scene"]
            for fixed, candidate in zip(baseline, rows)
        ]
        output[policy] = {
            "paired_clean_gain_percentage_points": _hierarchical(
                clean,
                seed=seed + position * 10_000 + 1,
                config=config,
                site_weights=weights,
            ),
            "paired_availability_gain_percentage_points": _hierarchical(
                availability,
                seed=seed + position * 10_000 + 2,
                config=config,
                site_weights=weights,
            ),
            "paired_total_bit_reduction_percentage": _hierarchical(
                bit_reduction,
                seed=seed + position * 10_000 + 3,
                config=config,
                site_weights=weights,
            ),
        }
    return output


def _policy_checks(
    comparisons: dict,
    wrong_decode: dict[str, int],
    *,
    config: dict,
) -> dict:
    gate = config["system_gate"]
    checks = {}
    for policy in (
        name for name in config["policies"] if name.startswith("checkpoint")
    ):
        row = comparisons[policy]
        clean = row["paired_clean_gain_percentage_points"]
        bits = row["paired_total_bit_reduction_percentage"]
        path_a = (
            float(bits["ci_lower"])
            >= float(gate["path_a"]["minimum_total_bit_reduction_percent"])
            and float(clean["ci_lower"])
            >= -float(
                gate["path_a"]["maximum_clean_loss_percentage_points"]
            )
        )
        path_b = (
            float(clean["ci_lower"])
            >= float(gate["path_b"]["minimum_clean_gain_percentage_points"])
            and float(bits["ci_lower"])
            >= -float(
                gate["path_b"]["maximum_total_bit_increase_percent"]
            )
        )
        safe = (
            int(wrong_decode[policy])
            == int(gate["wrong_codebook_actions_must_equal"])
        )
        checks[policy] = {
            "path_a_passed": bool(path_a),
            "path_b_passed": bool(path_b),
            "wrong_codebook_gate_passed": bool(safe),
            "system_gate_passed": bool((path_a or path_b) and safe),
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help=(
            "Smoke-only bootstrap override; the registered full run always "
            "uses the configuration value."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.4A-1 result")
    config = read_json(args.config)
    paths = {
        key: PROJECT_DIR / value
        for key, value in config["inputs"].items()
    }
    split = read_json(paths["development_split"])["split"]
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    scan = read_json(paths["matched_scan_config"])
    activation = read_json(paths["activation_protocol"])
    source = read_json(paths["s7_4a0_result"])
    validation_sites = list(split["validation"])
    internal_sites = list(split["internal_development_test"])
    if (
        not source["decision"]["feasibility_gate_passed"]
        or len(split["train"])
        != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
        or int(access["roles"]["reserve"]["access_count"]) != 0
    ):
        raise ValueError("S7.4A-1 governance or trigger mismatch")
    smoke = (
        args.max_sites is not None
        or args.trajectories is not None
        or args.bootstrap_replicates is not None
    )
    if args.bootstrap_replicates is not None:
        if int(args.bootstrap_replicates) < 1:
            raise ValueError("bootstrap-replicates must be positive")
        if args.max_sites is None and args.trajectories is None:
            raise ValueError(
                "bootstrap override is allowed only with a smoke subset"
            )
        config["monte_carlo"]["bootstrap_replicates"] = int(
            args.bootstrap_replicates
        )
    if args.max_sites is not None:
        if int(args.max_sites) < 1:
            raise ValueError("max-sites must be positive")
        validation_sites = validation_sites[: int(args.max_sites)]
    trajectories = (
        int(args.trajectories)
        if args.trajectories is not None
        else int(config["monte_carlo"]["trajectories"])
    )
    if trajectories < 1:
        raise ValueError("trajectories must be positive")
    metadata = registry["selected_site_members"]
    loaded = load_npy_members(
        Path(protocol["inputs"]["archive"]),
        [metadata[site]["member"] for site in validation_sites],
    )
    condition = scan["conditions"]["primary_fault"]
    base_seed = int(config["monte_carlo"]["random_seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    epsilon = float(config["task"]["epsilon_db"])
    codebook_epoch = int(config["reliability"]["codebook_epoch"])
    catalog_epoch = int(config["reliability"]["catalog_epoch"])
    maximum_activation_attempts = int(
        config["reliability"]["maximum_activation_attempts"]
    )
    if maximum_activation_attempts != int(
        activation["fail_closed"][
            "maximum_activation_attempts_before_full_install_fallback"
        ]
    ):
        raise ValueError("activation fallback rule changed")
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
    policies = config["policies"]
    started = time.perf_counter()
    n_results = {}
    policy_passing_n_count = {
        policy: 0
        for policy in policies
        if policy.startswith("checkpoint")
    }
    for n_position, n_channels in enumerate(
        map(int, config["task"]["n_channels"])
    ):
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        queries = task_queries(
            n_channels, config["task"]["demand_ratios"]
        )
        semantic_wp = frozen_n["workpoints"]["semantic"]
        exact_wp = frozen_n["workpoints"]["exact"]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(scan["task"]["exact_action_header_bits"]),
        )
        prototypes = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in frozen_n["prototypes"].items()
        }
        site_arrays = {policy: [] for policy in policies}
        scene_counts = []
        site_output = {}
        wrong_decode = {policy: 0 for policy in policies}
        for site_position, site in enumerate(validation_sites):
            power = aggregate_frequency_bins(
                loaded[metadata[site]["member"]], n_channels
            )
            timestamps = synthetic_six_hour_timestamps(power.shape[0])
            resolution = resolve_context_and_actions(
                power=power,
                queries=queries,
                epsilon_db=epsilon,
                prototypes=prototypes,
                novelty_threshold=float(frozen_n["novelty_threshold"]),
                bank_actions=frozen_n["bank_actions"],
                gate=protocol["context_gate"],
                primary_k=int(config["task"]["primary_k"]),
            )
            states = resolution.pop("states")
            indices = np.arange(len(states), dtype=np.int64)
            random = np.random.default_rng(
                base_seed
                + n_position * 1_000_000
                + site_position * 10_000
            ).random((trajectories, len(states), RANDOM_STREAM_COUNT))
            exact_runs = [
                simulate_exact_trajectory(
                    states,
                    timestamps,
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
                        scan["task"]["outage_penalty_db"]
                    ),
                )
                for trajectory in range(trajectories)
            ]
            if not resolution["resolved"]:
                policy_metrics = {
                    policy: [_metrics(run, None) for run in exact_runs]
                    for policy in policies
                }
            else:
                calibration_count = int(resolution["calibration_scenes"])
                calibration_indices = indices[:calibration_count]
                future_indices = indices[calibration_count:]
                codebook = codebook_from_selected_actions(
                    states[:calibration_count],
                    resolution["selected_actions"],
                )
                sender = install_codebook(codebook, epoch=codebook_epoch)
                full_packet = encode_context_install(sender, node_id=1)
                receiver = decode_context_install(full_packet).session
                compact_bits = _compact_bits(states, indices, sender)
                decision_name = str(resolution["decision"])
                if decision_name.startswith("BANK:"):
                    bank_source = decision_name.removeprefix("BANK:")
                    bank_id = int(frozen_n["bank_ids"][bank_source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)],
                        catalog_epoch=catalog_epoch,
                    )
                else:
                    bank_id = None
                    catalog = None
                policy_metrics = {policy: [] for policy in policies}
                for trajectory in range(trajectories):
                    calibration_result = simulate_exact_trajectory(
                        states,
                        timestamps,
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
                            semantic_wp["calibration_open_loop_attempts"]
                        ),
                        packet_bits=exact_packet_bits,
                        epsilon_db=epsilon,
                        outage_penalty_db=float(
                            scan["task"]["outage_penalty_db"]
                        ),
                    )
                    for policy, settings in policies.items():
                        heartbeat = settings["heartbeat_interval_scenes"]
                        checkpoint = settings[
                            "checkpoint_interval_scenes"
                        ]
                        activated = simulate_activation_semantic_trajectory(
                            states,
                            timestamps,
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
                                semantic_wp[
                                    "maximum_state_age_minutes"
                                ]
                            ),
                            ack_frame_bits=ack_bits,
                            outage_penalty_db=float(
                                scan["task"]["outage_penalty_db"]
                            ),
                            heartbeat_interval_scenes=(
                                None if heartbeat is None else int(heartbeat)
                            ),
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
                            checkpoint_interval_scenes=(
                                None
                                if checkpoint is None
                                else int(checkpoint)
                            ),
                        )
                        combined = combine_matched_trajectory_results(
                            [calibration_result, activated.matched]
                        )
                        policy_metrics[policy].append(
                            _metrics(combined, activated)
                        )
                        wrong_decode[policy] += int(
                            combined.wrong_codebook_decode_count
                        )
            for policy, rows in policy_metrics.items():
                site_arrays[policy].append(_arrays(rows))
            scene_counts.append(len(states))
            site_output[site] = {
                "resolved": bool(resolution["resolved"]),
                "decision": resolution["decision"],
                "scene_count": len(states),
                "calibration_scenes": int(
                    resolution["calibration_scenes"]
                ),
            }
            print(
                f"S7.4A-1 N={n_channels} site={site} complete",
                flush=True,
            )
        summary = _policy_summary(
            site_arrays,
            scene_counts,
            config=config,
            seed=bootstrap_seed + n_position * 1_000_000,
        )
        comparisons = _paired_comparisons(
            site_arrays,
            scene_counts,
            config=config,
            seed=bootstrap_seed + n_position * 1_000_000 + 500_000,
        )
        checks = _policy_checks(
            comparisons,
            wrong_decode,
            config=config,
        )
        for policy, row in checks.items():
            policy_passing_n_count[policy] += int(
                row["system_gate_passed"]
            )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "site_results": site_output,
            "policy_metrics": summary,
            "paired_comparisons_vs_fixed10": comparisons,
            "wrong_codebook_decode_total": wrong_decode,
            "checkpoint_policy_checks": checks,
        }
        print(
            "S7.4A-1 "
            f"N={n_channels} passing="
            f"{[key for key, value in checks.items() if value['system_gate_passed']]}",
            flush=True,
        )
    required = int(config["system_gate"]["minimum_passing_n_count"])
    best_policy = max(
        policy_passing_n_count,
        key=lambda name: (policy_passing_n_count[name], name),
    )
    passed = (
        policy_passing_n_count[best_policy] >= required and not smoke
    )
    next_action = (
        config["system_gate"]["if_gate_passes"]
        if passed
        else config["system_gate"]["if_gate_fails"]
    )
    result = {
        "version": "1.0",
        "status": (
            "stage7_s7_4a1_checkpoint_smoke_complete"
            if smoke
            else "stage7_s7_4a1_checkpoint_protocol_scan_complete"
        ),
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            key: sha256_file(path)
            for key, path in paths.items()
            if path.is_file()
        },
        "loaded_sites": {
            "train": [],
            "validation": validation_sites,
            "internal_development_test": [],
            "reserve": [],
        },
        "trajectories": trajectories,
        "bootstrap_replicates": int(
            config["monte_carlo"]["bootstrap_replicates"]
        ),
        "smoke": smoke,
        "n_results": n_results,
        "decision": {
            "policy_passing_n_count": policy_passing_n_count,
            "best_common_policy": best_policy,
            "required_n_count": required,
            "system_gate_passed": passed,
            "next_action": next_action,
        },
        "governance_checks": {
            "internal_development_test_not_loaded": not bool(
                set(validation_sites) & set(internal_sites)
            ),
            "reserve_remained_unread": True,
            "all_registered_policies_reported": True,
            "full_packet_ack_reset_recovery_accounting_used": True,
            "external_final_claim_forbidden": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
