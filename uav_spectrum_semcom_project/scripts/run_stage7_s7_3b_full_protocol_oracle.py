#!/usr/bin/env python
"""Run the S7.3b full-protocol forced-refresh oracle diagnostic."""

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
from spectrum_semcom.stage6_bit_accounting import (  # noqa: E402
    validate_breakdown_identity,
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
    resolve_context_and_actions,
    synthetic_six_hour_timestamps,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)
from spectrum_semcom.stage7_event_value import (  # noqa: E402
    build_task_event_value_labels,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage7_s7_3b_full_protocol_oracle_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_3b_full_protocol_oracle_v1/result.json"
)


def _maximum_true_run(values: tuple[bool, ...]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _cvar(values: tuple[float, ...], alpha: float = 0.9) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(np.ceil((1.0 - float(alpha)) * array.size)))
    return float(np.mean(array[-count:]))


def _trajectory_metrics(result) -> dict[str, float]:
    validate_breakdown_identity(result.bit_breakdown)
    scenes = len(result.clean)
    bits = result.bit_breakdown
    return {
        "clean_rate": float(np.mean(result.clean)),
        "availability_rate": float(np.mean(result.available)),
        "mean_effective_regret_db": float(
            np.mean(result.effective_regret_db)
        ),
        "cvar_0_9_effective_regret_db": _cvar(
            result.effective_regret_db
        ),
        "longest_dirty_run_scenes": float(
            _maximum_true_run(tuple(not value for value in result.clean))
        ),
        "total_bits_per_scene": float(
            bits.actual_total_application_bits / scenes
        ),
        "forward_bits_per_scene": float(bits.actual_forward_bits / scenes),
        "ack_bits_per_scene": float(bits.ack_bits / scenes),
        "heartbeat_bits_per_scene": float(
            (bits.heartbeat_request_bits + bits.heartbeat_response_bits)
            / scenes
        ),
        "recovery_bits_per_scene": float(
            bits.recovery_install_bits / scenes
        ),
        "escape_bits_per_scene": float(bits.escape_exact_bits / scenes),
        "wrong_codebook_decode_count": float(
            result.wrong_codebook_decode_count
        ),
    }


def _arrays(results: list) -> dict[str, np.ndarray]:
    rows = [_trajectory_metrics(result) for result in results]
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def _hierarchical(
    rows: list[np.ndarray],
    *,
    seed: int,
    config: dict,
    site_weights: np.ndarray,
) -> dict:
    return hierarchical_mean_ci(
        np.stack(rows),
        seed=seed,
        replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
        confidence=float(config["monte_carlo"]["confidence_level"]),
        site_weights=site_weights,
    )


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
    reference = str(config["diagnostic_gate"]["reference_policy"])
    baseline = site_arrays[reference]
    weights = np.asarray(scene_counts, dtype=np.float64)
    comparisons = {}
    for policy_position, (policy, rows) in enumerate(site_arrays.items()):
        if policy == reference:
            continue
        clean_gain = [
            100.0 * (candidate["clean_rate"] - fixed["clean_rate"])
            for fixed, candidate in zip(baseline, rows)
        ]
        availability_gain = [
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
        comparisons[policy] = {
            "paired_clean_gain_percentage_points": _hierarchical(
                clean_gain,
                seed=seed + policy_position * 10_000 + 1,
                config=config,
                site_weights=weights,
            ),
            "paired_availability_gain_percentage_points": _hierarchical(
                availability_gain,
                seed=seed + policy_position * 10_000 + 2,
                config=config,
                site_weights=weights,
            ),
            "paired_total_bit_reduction_percentage": _hierarchical(
                bit_reduction,
                seed=seed + policy_position * 10_000 + 3,
                config=config,
                site_weights=weights,
            ),
        }
    return comparisons


def _oracle_mask(
    states: list,
    site: str,
    codebook,
    *,
    config: dict,
    incremental_bits: int,
) -> tuple[np.ndarray, dict]:
    labels = build_task_event_value_labels(
        states,
        [site] * len(states),
        codebook,
        horizon=int(config["task"]["oracle_label_horizon_scenes"]),
        persistent_run_lengths=(2, 3),
        refresh_offset=int(
            config["task"]["oracle_refresh_offset_scenes"]
        ),
        incremental_bits=int(incremental_bits),
        clean_scene_value_bits=(32, 64, 128, 256),
    )
    mask = np.zeros(len(states), dtype=bool)
    offset = int(config["task"]["oracle_refresh_offset_scenes"])
    source = labels["source_indices"][
        labels["forced_refresh_beneficial"].astype(bool)
    ]
    mask[source + offset] = True
    return mask, {
        "eligible_source_scenes": int(labels["source_indices"].size),
        "beneficial_source_scenes": int(source.size),
        "trigger_count": int(np.sum(mask)),
        "trigger_prevalence": float(np.mean(mask)),
    }


def _gate(
    comparisons: dict,
    wrong_decode_total: dict[str, int],
    *,
    config: dict,
) -> dict:
    rules = config["diagnostic_gate"]
    oracle_policies = [
        name
        for name, row in config["policies"].items()
        if bool(row["forced_refresh_oracle"])
    ]
    policy_checks = {}
    any_policy_passed = False
    for policy in oracle_policies:
        row = comparisons[policy]
        clean = row["paired_clean_gain_percentage_points"]
        bits = row["paired_total_bit_reduction_percentage"]
        path_a = (
            float(bits["ci_lower"])
            >= float(
                rules["path_a"]["minimum_total_bit_reduction_percent"]
            )
            and float(clean["ci_lower"])
            >= -float(
                rules["path_a"]["maximum_clean_loss_percentage_points"]
            )
        )
        path_b = (
            float(clean["ci_lower"])
            >= float(
                rules["path_b"]["minimum_clean_gain_percentage_points"]
            )
            and float(bits["ci_lower"])
            >= -float(
                rules["path_b"]["maximum_total_bit_increase_percent"]
            )
        )
        safe = (
            wrong_decode_total[policy]
            == int(rules["wrong_codebook_actions_must_equal"])
        )
        passed = bool((path_a or path_b) and safe)
        any_policy_passed = any_policy_passed or passed
        policy_checks[policy] = {
            "path_a_passed": bool(path_a),
            "path_b_passed": bool(path_b),
            "wrong_codebook_gate_passed": bool(safe),
            "diagnostic_gate_passed": passed,
        }
    return {
        "oracle_policy_checks": policy_checks,
        "n_gate_passed": bool(any_policy_passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Smoke test only; does not satisfy the registered gate.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=None,
        help="Smoke test only; registered result uses the configured count.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.3b result")
    config = read_json(args.config)
    paths = {
        name: PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    scan = read_json(paths["matched_scan_config"])
    activation = read_json(paths["activation_protocol"])
    split = read_json(paths["development_split"])["split"]
    source = read_json(paths["s7_3a_result"])
    if (
        source["decision"]["label_viability_gate_passed"]
        or int(access["roles"]["reserve"]["access_count"]) != 0
        or int(access["roles"]["pilot"]["access_count"]) != 1
        or int(access["roles"]["stage6_final"]["access_count"]) != 1
        or int(access["roles"]["confirmation_lockbox"]["access_count"]) != 1
    ):
        raise ValueError("S7.3b governance or trigger mismatch")
    validation_sites = list(split["validation"])
    internal_sites = list(split["internal_development_test"])
    if (
        len(split["train"])
        != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
    ):
        raise ValueError("development split changed")
    smoke = args.max_sites is not None or args.trajectories is not None
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
    members = [metadata[site]["member"] for site in validation_sites]
    archive = Path(protocol["inputs"]["archive"])
    loaded = load_npy_members(archive, members)
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
    passing_n_count = 0
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
        wrong_decode_total = {policy: 0 for policy in policies}
        forced_request_total = {policy: 0 for policy in policies}
        forced_redundant_total = {policy: 0 for policy in policies}
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
            label_audit = {
                "eligible_source_scenes": 0,
                "beneficial_source_scenes": 0,
                "trigger_count": 0,
                "trigger_prevalence": 0.0,
            }
            if not resolution["resolved"]:
                policy_runs = {
                    policy: exact_runs for policy in policies
                }
            else:
                calibration_count = int(resolution["calibration_scenes"])
                calibration_indices = indices[:calibration_count]
                future_indices = indices[calibration_count:]
                future_states = [
                    states[int(index)] for index in future_indices
                ]
                codebook = codebook_from_selected_actions(
                    states[:calibration_count],
                    resolution["selected_actions"],
                )
                sender = install_codebook(codebook, epoch=codebook_epoch)
                full_packet = encode_context_install(sender, node_id=1)
                receiver = decode_context_install(full_packet).session
                compact_bits = _compact_bits(states, indices, sender)
                incremental_bits = (
                    compact_bits
                    * int(semantic_wp["task_update_open_loop_attempts"])
                    + ack_bits
                )
                oracle_mask, label_audit = _oracle_mask(
                    future_states,
                    site,
                    codebook,
                    config=config,
                    incremental_bits=incremental_bits,
                )
                if str(resolution["decision"]).startswith("BANK:"):
                    bank_source = str(resolution["decision"]).removeprefix(
                        "BANK:"
                    )
                    bank_id = int(frozen_n["bank_ids"][bank_source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)],
                        catalog_epoch=catalog_epoch,
                    )
                else:
                    bank_id = None
                    catalog = None
                policy_runs = {policy: [] for policy in policies}
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
                            heartbeat_interval_scenes=int(
                                settings["heartbeat_interval_scenes"]
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
                            forced_update_trigger_mask=(
                                oracle_mask
                                if bool(
                                    settings["forced_refresh_oracle"]
                                )
                                else None
                            ),
                        )
                        combined = combine_matched_trajectory_results(
                            [calibration_result, activated.matched]
                        )
                        policy_runs[policy].append(combined)
                        wrong_decode_total[policy] += int(
                            combined.wrong_codebook_decode_count
                        )
                        forced_request_total[policy] += int(
                            activated.forced_update_request_count
                        )
                        forced_redundant_total[policy] += int(
                            activated.forced_update_redundant_count
                        )
            for policy, runs in policy_runs.items():
                arrays = _arrays(runs)
                site_arrays[policy].append(arrays)
                if not resolution["resolved"]:
                    wrong_decode_total[policy] += int(
                        sum(
                            run.wrong_codebook_decode_count
                            for run in runs
                        )
                    )
            scene_counts.append(len(states))
            site_output[site] = {
                "resolved": bool(resolution["resolved"]),
                "decision": resolution["decision"],
                "scene_count": len(states),
                "calibration_scenes": int(
                    resolution["calibration_scenes"]
                ),
                "oracle_label": label_audit,
            }
            print(
                f"S7.3b N={n_channels} site={site} complete",
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
        gate = _gate(
            comparisons,
            wrong_decode_total,
            config=config,
        )
        passing_n_count += int(gate["n_gate_passed"])
        trigger_diagnostics = {}
        for policy in policies:
            request_count = int(forced_request_total[policy])
            redundant_count = int(forced_redundant_total[policy])
            trigger_diagnostics[policy] = {
                "request_count": request_count,
                "redundant_with_analytic_trigger_count": redundant_count,
                "redundant_fraction": (
                    float(redundant_count / request_count)
                    if request_count
                    else None
                ),
            }
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "site_results": site_output,
            "policy_metrics": summary,
            "paired_comparisons_vs_fixed10": comparisons,
            "trigger_diagnostics": trigger_diagnostics,
            "wrong_codebook_decode_total": wrong_decode_total,
            "gate": gate,
        }
        print(
            f"S7.3b N={n_channels} pass={gate['n_gate_passed']}",
            flush=True,
        )
    required = int(config["diagnostic_gate"]["minimum_passing_n_count"])
    passed = passing_n_count >= required and not smoke
    next_action = (
        config["diagnostic_gate"]["if_gate_passes"]
        if passed
        else config["diagnostic_gate"]["if_gate_fails"]
    )
    result = {
        "version": "1.0",
        "status": (
            "stage7_s7_3b_smoke_complete"
            if smoke
            else "stage7_s7_3b_full_protocol_oracle_complete"
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
        "smoke": smoke,
        "n_results": n_results,
        "decision": {
            "passing_n_count": passing_n_count,
            "required_n_count": required,
            "diagnostic_gate_passed": passed,
            "next_action": next_action,
        },
        "governance_checks": {
            "s7_3a_gate_failure_retained": True,
            "internal_development_test_not_loaded": not bool(
                set(validation_sites) & set(internal_sites)
            ),
            "reserve_remained_unread": True,
            "full_packet_ack_reset_recovery_accounting_used": True,
            "oracle_is_noncausal": True,
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
