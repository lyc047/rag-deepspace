#!/usr/bin/env python
"""Run Stage-7 S7.0 label audit and perfect-reset oracle headroom study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_activation_protocol_dry_run import _compact_bits  # noqa: E402
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
    RESET_STREAM,
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    encode_task_state,
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
from spectrum_semcom.stage7_multihorizon_risk import (  # noqa: E402
    build_multihorizon_risk_labels,
    perfect_reset_oracle_trigger_mask,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_s7_0_oracle_headroom_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/s7_0_oracle_headroom_v1/result.json"
)


def _development_split(sites: list[str], config: dict) -> dict[str, list[str]]:
    salt = str(config["data"]["deterministic_split_salt"])
    ordered = sorted(
        sites,
        key=lambda site: hashlib.sha256(
            f"{salt}|{site}".encode("utf-8")
        ).hexdigest(),
    )
    counts = config["data"]["split_counts"]
    train_end = int(counts["train"])
    validation_end = train_end + int(counts["validation"])
    split = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "internal_development_test": ordered[validation_end:],
    }
    if (
        sum(map(len, split.values())) != len(sites)
        or len(set(sum(split.values(), []))) != len(sites)
        or len(split["internal_development_test"])
        != int(counts["internal_development_test"])
    ):
        raise ValueError("development split counts do not match sites")
    return split


def _trajectory_metrics(result) -> dict[str, float]:
    validate_breakdown_identity(result.bit_breakdown)
    scenes = len(result.clean)
    breakdown = result.bit_breakdown
    return {
        "clean_rate": float(np.mean(result.clean)),
        "availability_rate": float(np.mean(result.available)),
        "total_bits_per_scene": float(
            breakdown.actual_total_application_bits / scenes
        ),
        "heartbeat_bits_per_scene": float(
            (
                breakdown.heartbeat_request_bits
                + breakdown.heartbeat_response_bits
            )
            / scenes
        ),
        "recovery_bits_per_scene": float(
            breakdown.recovery_install_bits / scenes
        ),
        "wrong_codebook_decode_count": float(
            result.wrong_codebook_decode_count
        ),
    }


def _arrays(results: list) -> dict[str, np.ndarray]:
    rows = [_trajectory_metrics(result) for result in results]
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in rows[0]
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


def _summarize_comparison(
    fixed_rows: list[dict[str, np.ndarray]],
    oracle_rows: list[dict[str, np.ndarray]],
    scene_counts: list[int],
    *,
    seed: int,
    config: dict,
) -> dict:
    weights = np.asarray(scene_counts, dtype=np.float64)
    metrics = {}
    for position, name in enumerate(fixed_rows[0]):
        metrics[name] = {
            "fixed10": _hierarchical(
                [row[name] for row in fixed_rows],
                seed=seed + position * 1009,
                config=config,
                site_weights=weights,
            ),
            "oracle": _hierarchical(
                [row[name] for row in oracle_rows],
                seed=seed + 100_000 + position * 1009,
                config=config,
                site_weights=weights,
            ),
        }
    clean_gain = [
        100.0 * (oracle["clean_rate"] - fixed["clean_rate"])
        for fixed, oracle in zip(fixed_rows, oracle_rows)
    ]
    availability_gain = [
        100.0
        * (oracle["availability_rate"] - fixed["availability_rate"])
        for fixed, oracle in zip(fixed_rows, oracle_rows)
    ]
    bit_reduction = [
        100.0
        * (
            fixed["total_bits_per_scene"]
            - oracle["total_bits_per_scene"]
        )
        / fixed["total_bits_per_scene"]
        for fixed, oracle in zip(fixed_rows, oracle_rows)
    ]
    return {
        "metrics": metrics,
        "paired_clean_gain_percentage_points": _hierarchical(
            clean_gain,
            seed=seed + 210_001,
            config=config,
            site_weights=weights,
        ),
        "paired_availability_gain_percentage_points": _hierarchical(
            availability_gain,
            seed=seed + 210_002,
            config=config,
            site_weights=weights,
        ),
        "paired_total_bit_reduction_percentage": _hierarchical(
            bit_reduction,
            seed=seed + 210_003,
            config=config,
            site_weights=weights,
        ),
    }


def _empty_label_counts(horizons: tuple[int, ...]) -> dict:
    return {
        str(horizon): {
            "task_observed": 0,
            "task_positive": 0,
            "context_observed": 0,
            "context_positive": 0,
            "any_positive": 0,
        }
        for horizon in horizons
    }


def _accumulate_labels(
    counts: dict,
    labels: dict,
    *,
    include_task: bool,
) -> None:
    for position, horizon in enumerate(labels["horizons"]):
        observed = np.asarray(labels["observed"][:, position], dtype=bool)
        row = counts[str(horizon)]
        if include_task:
            row["task_observed"] += int(np.sum(observed))
            row["task_positive"] += int(
                np.sum(labels["task_failure"][observed, position])
            )
        else:
            row["context_observed"] += int(np.sum(observed))
            row["context_positive"] += int(
                np.sum(labels["context_failure"][observed, position])
            )
            row["any_positive"] += int(
                np.sum(labels["any_failure"][observed, position])
            )


def _finalize_label_counts(counts: dict, trajectories: int) -> dict:
    output = {}
    for horizon, row in counts.items():
        task_observed = int(row["task_observed"])
        context_observed = int(row["context_observed"])
        expected_context_observed = task_observed * int(trajectories)
        if context_observed != expected_context_observed:
            raise ValueError("context label denominator is inconsistent")
        output[horizon] = {
            **row,
            "task_positive_rate": (
                float(row["task_positive"] / task_observed)
                if task_observed
                else None
            ),
            "context_positive_rate": (
                float(row["context_positive"] / context_observed)
                if context_observed
                else None
            ),
            "any_positive_rate": (
                float(row["any_positive"] / context_observed)
                if context_observed
                else None
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Development smoke only; full registered result requires all sites.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-7 S7.0 result")
    config = read_json(args.config)
    paths = {
        name: PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    governance = read_json(paths["governance"])
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    scan_config = read_json(paths["matched_scan_config"])
    activation_protocol = read_json(paths["activation_protocol"])
    roles = list(config["data"]["development_roles"])
    for role in roles:
        if int(access["roles"][role]["access_count"]) != 1:
            raise ValueError(f"development role {role} was not previously consumed")
    if int(access["roles"]["reserve"]["access_count"]) != 0:
        raise ValueError("reserve values must remain unread")
    metadata_rows = [
        row for role in roles for row in role_members(registry, role)
    ]
    expected_sites = int(config["data"]["expected_development_site_count"])
    if len(metadata_rows) != expected_sites:
        raise ValueError("development site count changed")
    site_split = _development_split(
        [row["site"] for row in metadata_rows], config
    )
    smoke = args.max_sites is not None
    if smoke:
        if int(args.max_sites) < 1:
            raise ValueError("max-sites must be positive")
        metadata_rows = metadata_rows[: int(args.max_sites)]
    archive = Path(protocol["inputs"]["archive"])
    if not archive.is_file():
        raise FileNotFoundError(archive)
    loaded = load_npy_members(
        archive, [row["member"] for row in metadata_rows]
    )

    condition = scan_config["conditions"]["primary_fault"]
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(config["monte_carlo"]["random_seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    epsilon = float(config["task"]["epsilon_db"])
    horizons = tuple(map(int, config["task"]["prediction_horizons_scenes"]))
    codebook_epoch = int(config["reliability"]["codebook_epoch"])
    catalog_epoch = int(config["reliability"]["catalog_epoch"])
    maximum_activation_attempts = int(
        config["reliability"]["maximum_activation_attempts"]
    )
    if maximum_activation_attempts != int(
        activation_protocol["fail_closed"][
            "maximum_activation_attempts_before_full_install_fallback"
        ]
    ):
        raise ValueError("activation attempt limit changed")
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
    started = time.perf_counter()
    n_results = {}
    for n_position, n_channels in enumerate(
        map(int, config["task"]["n_channels"])
    ):
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        queries = task_queries(n_channels, config["task"]["demand_ratios"])
        semantic_wp = frozen_n["workpoints"]["semantic"]
        exact_wp = frozen_n["workpoints"]["exact"]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(scan_config["task"]["exact_action_header_bits"]),
        )
        prototypes = {
            source: np.asarray(values, dtype=np.float64)
            for source, values in frozen_n["prototypes"].items()
        }
        fixed_site_arrays = []
        oracle_site_arrays = []
        scene_counts = []
        label_counts = _empty_label_counts(horizons)
        site_outputs = {}
        resolved_site_count = 0
        for site_position, metadata in enumerate(metadata_rows):
            site = metadata["site"]
            power = aggregate_frequency_bins(
                loaded[metadata["member"]], n_channels
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
                        scan_config["task"]["outage_penalty_db"]
                    ),
                )
                for trajectory in range(trajectories)
            ]
            if not resolution["resolved"]:
                fixed_runs = exact_runs
                oracle_runs = exact_runs
                site_label_output = {"included": False, "reason": "exact_fallback"}
            else:
                resolved_site_count += 1
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
                is_bank = str(resolution["decision"]).startswith("BANK:")
                if is_bank:
                    source = str(resolution["decision"]).removeprefix("BANK:")
                    bank_id = int(frozen_n["bank_ids"][source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)], catalog_epoch=catalog_epoch
                    )
                else:
                    bank_id = None
                    catalog = None
                fixed_runs = []
                oracle_runs = []
                future_states = [states[int(index)] for index in future_indices]
                current_actions = [
                    encode_task_state(codebook, state).decoder_actions
                    for state in future_states
                ]
                task_only = build_multihorizon_risk_labels(
                    future_states,
                    current_actions,
                    [site] * len(future_states),
                    horizons=horizons,
                )
                _accumulate_labels(
                    label_counts, task_only, include_task=True
                )
                site_task_rates = {}
                for position, horizon in enumerate(horizons):
                    observed = task_only["observed"][:, position]
                    site_task_rates[str(horizon)] = float(
                        np.mean(task_only["task_failure"][observed, position])
                    )
                context_positive = {str(value): 0 for value in horizons}
                any_positive = {str(value): 0 for value in horizons}
                context_observed = {str(value): 0 for value in horizons}
                for trajectory in range(trajectories):
                    calibration = simulate_exact_trajectory(
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
                    future_random = random[trajectory, calibration_count:]
                    oracle_mask = perfect_reset_oracle_trigger_mask(
                        future_random,
                        reset_probability=float(
                            condition["receiver_context_reset_probability"]
                        ),
                        reset_stream_index=RESET_STREAM,
                    )
                    fixed_suffix = simulate_activation_semantic_trajectory(
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
                        maximum_activation_attempts=maximum_activation_attempts,
                        condition=condition,
                        random_values=future_random,
                        epsilon_db=epsilon,
                        max_age_minutes=float(
                            semantic_wp["maximum_state_age_minutes"]
                        ),
                        ack_frame_bits=ack_bits,
                        outage_penalty_db=float(
                            scan_config["task"]["outage_penalty_db"]
                        ),
                        heartbeat_interval_scenes=int(
                            config["reliability"][
                                "fixed_baseline_heartbeat_scenes"
                            ]
                        ),
                        heartbeat_request_frame_bits=heartbeat_request_bits,
                        heartbeat_response_frame_bits=heartbeat_response_bits,
                        task_open_loop_attempts=int(
                            semantic_wp["task_update_open_loop_attempts"]
                        ),
                        compact_codeword_frame_bits=compact_bits,
                    )
                    oracle_suffix = simulate_activation_semantic_trajectory(
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
                        maximum_activation_attempts=maximum_activation_attempts,
                        condition=condition,
                        random_values=future_random,
                        epsilon_db=epsilon,
                        max_age_minutes=float(
                            semantic_wp["maximum_state_age_minutes"]
                        ),
                        ack_frame_bits=ack_bits,
                        outage_penalty_db=float(
                            scan_config["task"]["outage_penalty_db"]
                        ),
                        heartbeat_interval_scenes=None,
                        heartbeat_request_frame_bits=heartbeat_request_bits,
                        heartbeat_response_frame_bits=heartbeat_response_bits,
                        task_open_loop_attempts=int(
                            semantic_wp["task_update_open_loop_attempts"]
                        ),
                        compact_codeword_frame_bits=compact_bits,
                        heartbeat_trigger_mask=oracle_mask,
                    )
                    fixed_runs.append(
                        combine_matched_trajectory_results(
                            [calibration, fixed_suffix.matched]
                        )
                    )
                    oracle_runs.append(
                        combine_matched_trajectory_results(
                            [calibration, oracle_suffix.matched]
                        )
                    )
                    combined_labels = build_multihorizon_risk_labels(
                        future_states,
                        current_actions,
                        [site] * len(future_states),
                        horizons=horizons,
                        context_failure_events=oracle_mask,
                    )
                    _accumulate_labels(
                        label_counts,
                        combined_labels,
                        include_task=False,
                    )
                    for position, horizon in enumerate(horizons):
                        observed = combined_labels["observed"][:, position]
                        key = str(horizon)
                        context_observed[key] += int(np.sum(observed))
                        context_positive[key] += int(
                            np.sum(
                                combined_labels["context_failure"][
                                    observed, position
                                ]
                            )
                        )
                        any_positive[key] += int(
                            np.sum(
                                combined_labels["any_failure"][
                                    observed, position
                                ]
                            )
                        )
                site_label_output = {
                    "included": True,
                    "task_positive_rate": site_task_rates,
                    "context_positive_rate": {
                        key: float(context_positive[key] / context_observed[key])
                        for key in context_observed
                    },
                    "any_positive_rate": {
                        key: float(any_positive[key] / context_observed[key])
                        for key in context_observed
                    },
                }
            fixed_arrays = _arrays(fixed_runs)
            oracle_arrays = _arrays(oracle_runs)
            fixed_site_arrays.append(fixed_arrays)
            oracle_site_arrays.append(oracle_arrays)
            scene_counts.append(len(states))
            site_outputs[site] = {
                "source_role": next(
                    role
                    for role in roles
                    if site in registry["split"]["roles"][role]
                ),
                "scene_count": len(states),
                "decision": resolution["decision"],
                "resolved": bool(resolution["resolved"]),
                "calibration_scenes": int(resolution["calibration_scenes"]),
                "labels": site_label_output,
                "fixed10_mean_clean_rate": float(
                    np.mean(fixed_arrays["clean_rate"])
                ),
                "oracle_mean_clean_rate": float(
                    np.mean(oracle_arrays["clean_rate"])
                ),
                "fixed10_mean_total_bits_per_scene": float(
                    np.mean(fixed_arrays["total_bits_per_scene"])
                ),
                "oracle_mean_total_bits_per_scene": float(
                    np.mean(oracle_arrays["total_bits_per_scene"])
                ),
            }
            print(
                f"S7.0 N={n_channels} site={site} complete",
                flush=True,
            )
        comparison = _summarize_comparison(
            fixed_site_arrays,
            oracle_site_arrays,
            scene_counts,
            seed=bootstrap_seed + n_position * 1_000_000,
            config=config,
        )
        gate = config["oracle_headroom_gate"]
        bit = comparison["paired_total_bit_reduction_percentage"]
        clean = comparison["paired_clean_gain_percentage_points"]
        wrong_oracle = comparison["metrics"][
            "wrong_codebook_decode_count"
        ]["oracle"]["mean"]
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "evaluated_site_count": len(metadata_rows),
            "resolved_semantic_site_count": resolved_site_count,
            "unresolved_exact_fallback_site_count": (
                len(metadata_rows) - resolved_site_count
            ),
            "label_audit": _finalize_label_counts(
                label_counts, trajectories
            ),
            "comparison": comparison,
            "site_results": site_outputs,
            "gate": {
                "bit_headroom_pass": bool(
                    bit["ci_lower"]
                    >= float(gate["minimum_total_bit_reduction_percent"])
                    and clean["ci_lower"]
                    >= -float(
                        gate["maximum_clean_loss_percentage_points"]
                    )
                ),
                "clean_headroom_pass": bool(
                    clean["ci_lower"]
                    >= float(
                        gate["or_minimum_clean_gain_percentage_points"]
                    )
                ),
                "wrong_codebook_actions_zero": bool(
                    abs(wrong_oracle) < 1e-12
                ),
            },
        }
    passing_n = sum(
        (
            row["gate"]["bit_headroom_pass"]
            or row["gate"]["clean_headroom_pass"]
        )
        and row["gate"]["wrong_codebook_actions_zero"]
        for row in n_results.values()
    )
    required = int(
        config["oracle_headroom_gate"]["minimum_passing_n_count"]
    )
    result = {
        "version": "1.0",
        "status": (
            "stage7_s7_0_smoke_complete"
            if smoke
            else "stage7_s7_0_oracle_headroom_complete"
        ),
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        },
        "archive_path": str(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "development_roles": roles,
        "development_site_count": len(metadata_rows),
        "reserve_access_count_before_and_after": int(
            access["roles"]["reserve"]["access_count"]
        ),
        "development_split": site_split,
        "n_results": n_results,
        "decision": {
            "passing_n_count": int(passing_n),
            "required_passing_n_count": required,
            "oracle_headroom_gate_passed": bool(
                not smoke and passing_n >= required
            ),
            "next_action": (
                config["oracle_headroom_gate"]["if_gate_passes"]
                if not smoke and passing_n >= required
                else (
                    "run_full_registered_s7_0"
                    if smoke
                    else config["oracle_headroom_gate"]["if_gate_fails"]
                )
            ),
        },
        "governance_checks": {
            "stage7_governance_status": governance["status"],
            "all_development_roles_previously_consumed": True,
            "reserve_remained_unread": (
                int(access["roles"]["reserve"]["access_count"]) == 0
            ),
            "oracle_is_noncausal_upper_bound": True,
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
