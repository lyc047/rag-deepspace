#!/usr/bin/env python
"""Run Stage-6R fault Shapley attribution and activation counterfactual."""

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
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_few_shot_adaptation import _load_combined, _queries  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
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
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    simulate_calibrated_semantic_trajectory,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_recovery_fault_attribution_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/recovery_fault_attribution_v1/result.json"
)


def _mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, float]:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    positions = rng.integers(
        0, rows.size, size=(int(replicates), rows.size)
    )
    boot = np.mean(rows[positions], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(rows)),
        "ci_lower": float(np.quantile(boot, alpha)),
        "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
    }


def _aggregate_metric_arrays(results: list) -> dict[str, np.ndarray]:
    output = {
        "recovery_install_bits_per_scene": [],
        "total_bits_per_scene": [],
        "task_failure_probability": [],
        "unavailability_probability": [],
    }
    for result in results:
        scenes = len(result.clean)
        output["recovery_install_bits_per_scene"].append(
            result.bit_breakdown.recovery_install_bits / scenes
        )
        output["total_bits_per_scene"].append(
            result.bit_breakdown.actual_total_application_bits / scenes
        )
        output["task_failure_probability"].append(
            1.0 - float(np.mean(result.clean))
        )
        output["unavailability_probability"].append(
            1.0 - float(np.mean(result.available))
        )
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in output.items()
    }


def _coalition_key(names: tuple[str, ...]) -> str:
    return "NONE" if not names else "+".join(names)


def _shapley(
    coalition_values: dict[frozenset[str], float],
    factors: tuple[str, ...],
) -> dict[str, float]:
    count = len(factors)
    denominator = math.factorial(count)
    output = {}
    universe = set(factors)
    for factor in factors:
        value = 0.0
        others = sorted(universe - {factor})
        for size in range(len(others) + 1):
            weight = (
                math.factorial(size)
                * math.factorial(count - size - 1)
                / denominator
            )
            for subset_tuple in itertools.combinations(others, size):
                subset = frozenset(subset_tuple)
                value += weight * (
                    coalition_values[subset | {factor}]
                    - coalition_values[subset]
                )
        output[factor] = float(value)
    return output


def _adjusted_install_bits(
    result,
    *,
    decision: str,
    frame_bits: int,
    scenario: str,
) -> int:
    bits = result.bit_breakdown
    current = (
        bits.initial_install_bits + bits.recovery_install_bits
    )
    install_count = int(result.context_install_count)
    if decision.startswith("BANK:"):
        replacement = install_count * int(frame_bits)
    elif scenario == "cached_all":
        recovery_count = max(0, install_count - 1)
        replacement = (
            bits.initial_install_bits + recovery_count * int(frame_bits)
        )
    else:
        replacement = current
    return int(
        bits.actual_total_application_bits - current + replacement
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fault attribution")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        name: PROJECT_DIR / config["inputs"][name]
        for name in config["inputs"]
    }
    adaptation = json.loads(
        paths["adaptation_config"].read_text(encoding="utf-8")
    )
    temporal = json.loads(
        paths["temporal_gate_result"].read_text(encoding="utf-8")
    )
    scan_config = json.loads(
        paths["matched_scan_config"].read_text(encoding="utf-8")
    )
    scan = json.loads(
        paths["matched_scan_result"].read_text(encoding="utf-8")
    )
    access = json.loads(
        paths["external_final_access_state"].read_text(encoding="utf-8")
    )
    factor_config = config["fault_factors"]
    factors = tuple(factor_config)
    factor_coalitions = [
        frozenset(
            factors[position]
            for position, enabled in enumerate(bits)
            if enabled
        )
        for bits in itertools.product((False, True), repeat=len(factors))
    ]
    target = float(config["frozen_selection"]["matched_clean_target"])
    primary_k = int(config["frozen_selection"]["primary_k"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(scan_config["monte_carlo"]["seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    replicates = int(config["monte_carlo"]["bootstrap_replicates"])
    confidence = float(config["monte_carlo"]["confidence_level"])
    codebook_epoch = int(scan_config["task"]["codebook_epoch"])
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
    n_results = {}
    started = time.perf_counter()
    registered_n = list(map(int, scan_config["task"]["n_channels"]))

    for n_position, n_channels in enumerate(registered_n):
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(
                row,
                queries,
                epsilon_db=float(scan_config["task"]["epsilon_db"]),
            )
            for row in powers
        ]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(
                scan_config["task"]["exact_action_header_bits"]
            ),
        )
        scan_n = scan["conditions"]["primary_fault"][str(n_channels)]
        matched = next(
            row
            for row in scan_n["matched_clean"]
            if row["common_eligible"]
            and abs(float(row["target_clean_rate"]) - target) < 1e-12
        )
        semantic_workpoint = next(
            row
            for row in scan_n["semantic_workpoints"]
            if row["workpoint_id"] == matched["semantic_workpoint_id"]
        )
        exact_workpoint = next(
            row
            for row in scan_n["exact_workpoints"]
            if row["workpoint_id"] == matched["exact_workpoint_id"]
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == primary_k
        }
        activity_inputs = {}
        for campaign_position, campaign in enumerate(
            adaptation["splitting"]["campaign_ids"]
        ):
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            row = temporal_rows[campaign]
            activity_inputs[campaign] = {
                "indices": indices,
                "calibration_count": int(row["calibration_scenes"]),
                "selected_actions": row["selected_actions"],
                "decision": row["decision"],
                "random": np.random.default_rng(
                    base_seed
                    + n_position * 1_000_000
                    + campaign_position * 10_000
                ).random(
                    (trajectories, indices.size, RANDOM_STREAM_COUNT)
                ),
            }

        coalition_outputs = {}
        primary_activity_results = None
        for coalition_position, coalition in enumerate(factor_coalitions):
            condition = {
                values["condition_key"]: float(
                    values["on"] if name in coalition else values["off"]
                )
                for name, values in factor_config.items()
            }
            condition["include_reset_hypothesis_after_missing_ack"] = True
            activity_results = {}
            aggregate_runs = []
            for trajectory in range(trajectories):
                campaign_runs = []
                for campaign, value in activity_inputs.items():
                    campaign_runs.append(
                        simulate_calibrated_semantic_trajectory(
                            states,
                            timestamps,
                            value["indices"],
                            calibration_scene_count=value[
                                "calibration_count"
                            ],
                            selected_actions=value["selected_actions"],
                            random_values=value["random"][trajectory],
                            condition=condition,
                            exact_packet_bits=exact_packet_bits,
                            epsilon_db=float(
                                scan_config["task"]["epsilon_db"]
                            ),
                            max_age_minutes=float(
                                semantic_workpoint[
                                    "maximum_state_age_minutes"
                                ]
                            ),
                            ack_frame_bits=ack_bits,
                            outage_penalty_db=float(
                                scan_config["task"]["outage_penalty_db"]
                            ),
                            heartbeat_interval_scenes=semantic_workpoint[
                                "heartbeat_silence_scenes"
                            ],
                            heartbeat_request_frame_bits=(
                                heartbeat_request_bits
                            ),
                            heartbeat_response_frame_bits=(
                                heartbeat_response_bits
                            ),
                            calibration_open_loop_attempts=int(
                                semantic_workpoint[
                                    "calibration_open_loop_attempts"
                                ]
                            ),
                            task_open_loop_attempts=int(
                                semantic_workpoint[
                                    "task_update_open_loop_attempts"
                                ]
                            ),
                            codebook_epoch=codebook_epoch,
                        ).combined
                    )
                aggregate_runs.append(
                    combine_matched_trajectory_results(campaign_runs)
                )
                for campaign, result in zip(activity_inputs, campaign_runs):
                    activity_results.setdefault(campaign, []).append(result)
            arrays = _aggregate_metric_arrays(aggregate_runs)
            coalition_outputs[coalition] = {
                "coalition_id": _coalition_key(
                    tuple(name for name in factors if name in coalition)
                ),
                "enabled_factors": [
                    name for name in factors if name in coalition
                ],
                "metrics": {
                    name: _mean_ci(
                        values,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + coalition_position * 100
                            + metric_position
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    )
                    for metric_position, (name, values) in enumerate(
                        arrays.items()
                    )
                },
            }
            if len(coalition) == len(factors):
                primary_activity_results = activity_results
        if primary_activity_results is None:
            raise ValueError("primary fault coalition is missing")

        shapley_output = {}
        efficiency = {}
        empty = frozenset()
        full = frozenset(factors)
        tolerance = float(
            config["decision_rules"][
                "shapley_efficiency_absolute_tolerance"
            ]
        )
        for metric in config["attribution_metrics"]:
            values = {
                coalition: output["metrics"][metric]["mean"]
                for coalition, output in coalition_outputs.items()
            }
            attributed = _shapley(values, factors)
            delta = float(values[full] - values[empty])
            residual = float(delta - sum(attributed.values()))
            shapley_output[metric] = {
                "no_fault_value": float(values[empty]),
                "primary_fault_value": float(values[full]),
                "primary_minus_no_fault": delta,
                "factor_contributions": attributed,
                "efficiency_residual": residual,
            }
            efficiency[metric] = bool(abs(residual) <= tolerance)

        primary_condition = scan_config["conditions"]["primary_fault"]
        exact_by_activity = {}
        for campaign, value in activity_inputs.items():
            runs = []
            for trajectory in range(trajectories):
                runs.append(
                    simulate_exact_trajectory(
                        states,
                        timestamps,
                        value["indices"],
                        random_values=value["random"][trajectory],
                        packet_loss_probability=float(
                            primary_condition["task_loss_probability"]
                        ),
                        receiver_reset_probability=float(
                            primary_condition[
                                "receiver_context_reset_probability"
                            ]
                        ),
                        task_open_loop_attempts=int(
                            exact_workpoint[
                                "task_packet_open_loop_attempts"
                            ]
                        ),
                        packet_bits=exact_packet_bits,
                        epsilon_db=float(
                            scan_config["task"]["epsilon_db"]
                        ),
                        outage_penalty_db=float(
                            scan_config["task"]["outage_penalty_db"]
                        ),
                    )
                )
            exact_by_activity[campaign] = runs

        counterfactual_rows = []
        for width_position, width in enumerate(
            map(
                int,
                config["short_activation_counterfactual"][
                    "candidate_frame_bits"
                ],
            )
        ):
            scenarios = {}
            for scenario_position, scenario in enumerate(
                ("known_bank_only", "cached_all")
            ):
                scene_semantic = []
                scene_exact = []
                macro_semantic = []
                macro_exact = []
                activity_output = {}
                for trajectory in range(trajectories):
                    semantic_total = 0
                    exact_total = 0
                    scene_total = 0
                    activity_semantic_bits = []
                    activity_exact_bits = []
                    for campaign, value in activity_inputs.items():
                        semantic_result = primary_activity_results[campaign][
                            trajectory
                        ]
                        exact_result = exact_by_activity[campaign][trajectory]
                        adjusted = _adjusted_install_bits(
                            semantic_result,
                            decision=value["decision"],
                            frame_bits=width,
                            scenario=scenario,
                        )
                        scenes = len(semantic_result.clean)
                        semantic_total += adjusted
                        exact_total += (
                            exact_result.bit_breakdown.actual_total_application_bits
                        )
                        scene_total += scenes
                        activity_semantic_bits.append(adjusted / scenes)
                        activity_exact_bits.append(
                            exact_result.bit_breakdown.actual_total_application_bits
                            / scenes
                        )
                    scene_semantic.append(semantic_total / scene_total)
                    scene_exact.append(exact_total / scene_total)
                    macro_semantic.append(
                        float(np.mean(activity_semantic_bits))
                    )
                    macro_exact.append(float(np.mean(activity_exact_bits)))
                scene_semantic_array = np.asarray(scene_semantic)
                scene_exact_array = np.asarray(scene_exact)
                macro_semantic_array = np.asarray(macro_semantic)
                macro_exact_array = np.asarray(macro_exact)
                scene_savings = (
                    100.0
                    * (scene_exact_array - scene_semantic_array)
                    / scene_exact_array
                )
                macro_savings = (
                    100.0
                    * (macro_exact_array - macro_semantic_array)
                    / macro_exact_array
                )
                for campaign_position, (campaign, value) in enumerate(
                    activity_inputs.items()
                ):
                    semantic_bits = np.asarray(
                        [
                            _adjusted_install_bits(
                                result,
                                decision=value["decision"],
                                frame_bits=width,
                                scenario=scenario,
                            )
                            / len(result.clean)
                            for result in primary_activity_results[campaign]
                        ]
                    )
                    exact_bits = np.asarray(
                        [
                            result.bit_breakdown.actual_total_application_bits
                            / len(result.clean)
                            for result in exact_by_activity[campaign]
                        ]
                    )
                    activity_output[campaign] = {
                        "decision": value["decision"],
                        "paired_savings_percentage": _mean_ci(
                            100.0
                            * (exact_bits - semantic_bits)
                            / exact_bits,
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + width_position * 1000
                                + scenario_position * 100
                                + campaign_position
                            ),
                            replicates=replicates,
                            confidence=confidence,
                        ),
                    }
                scenarios[scenario] = {
                    "scene_weighted_paired_savings_percentage": _mean_ci(
                        scene_savings,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + width_position * 1000
                            + scenario_position * 100
                            + 71
                        ),
                        replicates=replicates,
                        confidence=confidence,
                    ),
                    "equal_activity_macro_paired_savings_percentage": (
                        _mean_ci(
                            macro_savings,
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + width_position * 1000
                                + scenario_position * 100
                                + 72
                            ),
                            replicates=replicates,
                            confidence=confidence,
                        )
                    ),
                    "activity_results": activity_output,
                }
            counterfactual_rows.append(
                {
                    "activation_frame_bits": width,
                    "scenarios": scenarios,
                }
            )

        width64 = next(
            row
            for row in counterfactual_rows
            if row["activation_frame_bits"] == 64
        )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "frozen_matched_row": matched,
            "coalition_results": [
                coalition_outputs[coalition]
                for coalition in factor_coalitions
            ],
            "shapley_attribution": shapley_output,
            "activation_counterfactual": counterfactual_rows,
            "checks": {
                "all_shapley_efficiency_checks_passed": all(
                    efficiency.values()
                ),
                "shapley_efficiency_by_metric": efficiency,
                "cached_all_64_scene_weighted_lower_positive": (
                    width64["scenarios"]["cached_all"][
                        "scene_weighted_paired_savings_percentage"
                    ]["ci_lower"]
                    > 0.0
                ),
                "cached_all_64_macro_mean_positive": (
                    width64["scenarios"]["cached_all"][
                        "equal_activity_macro_paired_savings_percentage"
                    ]["mean"]
                    > 0.0
                ),
                "known_bank_64_macro_mean_positive": (
                    width64["scenarios"]["known_bank_only"][
                        "equal_activity_macro_paired_savings_percentage"
                    ]["mean"]
                    > 0.0
                ),
            },
        }
        print(f"S6R recovery attribution N={n_channels} complete", flush=True)

    cached_macro_positive = sum(
        value["checks"]["cached_all_64_macro_mean_positive"]
        for value in n_results.values()
    )
    known_macro_positive = sum(
        value["checks"]["known_bank_64_macro_mean_positive"]
        for value in n_results.values()
    )
    result = {
        "version": "1.0",
        "status": "stage6r_recovery_fault_attribution_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "external_access_status_before_run": access.get("status"),
        "external_access_count_before_run": access.get("access_count"),
        "n_results": n_results,
        "decision_summary": {
            "cached_all_64_macro_positive_n_count": cached_macro_positive,
            "known_bank_64_macro_positive_n_count": known_macro_positive,
            "cached_all_64_scene_weighted_lower_positive_all_n": all(
                value["checks"][
                    "cached_all_64_scene_weighted_lower_positive"
                ]
                for value in n_results.values()
            ),
            "activation_protocol_entry_passed": (
                cached_macro_positive >= 3
                and all(
                    value["checks"][
                        "cached_all_64_scene_weighted_lower_positive"
                    ]
                    for value in n_results.values()
                )
            ),
            "known_bank_protocol_entry_passed": (
                known_macro_positive >= 3
            ),
        },
        "checks": {
            "all_shapley_efficiency_checks_passed": all(
                value["checks"][
                    "all_shapley_efficiency_checks_passed"
                ]
                for value in n_results.values()
            ),
            "external_final_access_count_unchanged": (
                access.get("access_count") == 1
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision_summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
