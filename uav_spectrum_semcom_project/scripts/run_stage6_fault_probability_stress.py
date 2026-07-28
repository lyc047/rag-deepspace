#!/usr/bin/env python
"""Run one-factor fault-probability stress on the frozen Stage-6 candidate."""

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

from run_stage6_component_ablation_development import (  # noqa: E402
    _exact_bundle_bits,
    _paired_intervals,
    _simulate_exact_query_bundle,
)
from run_stage6_context_recovery_development import (  # noqa: E402
    simulate_trajectory,
    summarize_trajectories,
)
from run_stage6_joint_predictive_recovery_development import (  # noqa: E402
    _common_random_with_frozen_prefix,
)
from run_stage6_predictive_repetition_development import (  # noqa: E402
    _score_candidates,
    _upper_quantile_threshold,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from run_stage6_temporal_hazard_development import (  # noqa: E402
    _fit_oof,
    _model,
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
from spectrum_semcom.stage6_temporal_hazard import (  # noqa: E402
    build_temporal_hazard_dataset,
)


FAULT_FIELDS = (
    "install_loss_probability",
    "task_loss_probability",
    "ack_loss_probability",
    "receiver_context_reset_probability",
)


def _condition_key(condition: dict) -> tuple[float, ...]:
    return tuple(float(condition[field]) for field in FAULT_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_fault_probability_stress_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/fault_probability_stress_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "fault_probability_stress_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite fault stress result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    rules = protocol["analysis_rules"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
        or not governance["faults_are_controlled_injections_not_measurements"]
        or rules["results_may_modify_frozen_architecture_or_parameters"]
    ):
        raise ValueError("invalid fault stress governance")
    freeze = protocol["architecture_freeze"]
    if sha256_file(PROJECT_DIR / freeze["result"]) != freeze["sha256"]:
        raise ValueError("architecture freeze changed")
    predecessors = {}
    for name, specification in protocol["frozen_predecessors"].items():
        path = PROJECT_DIR / specification["path"]
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"frozen predecessor changed: {name}")
        predecessors[name] = json.loads(path.read_text(encoding="utf-8"))
    if set(protocol["one_factor_scans"]) != set(FAULT_FIELDS):
        raise ValueError("fault scan dimensions changed")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("fault stress cache changed")
    access_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    ack_bits = cumulative_ack_payload_bits()
    request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=int(grid["codebook_epoch"]),
            expected_update_epoch=0,
        ).size
    )
    response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=int(grid["codebook_epoch"]),
            current_update_epoch=0,
        ).size
    )
    monte_carlo = protocol["monte_carlo"]
    trajectory_count = int(monte_carlo["trajectories"])
    baseline_condition = dict(protocol["baseline_fault_condition"])
    all_results = {}
    baseline_reproduction = {}

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
            [states[int(index)] for index in train_indices]
        )
        sender_session = install_codebook(
            codebook, epoch=int(grid["codebook_epoch"])
        )
        install_packet = encode_context_install(
            sender_session, node_id=1
        )
        receiver_session = decode_context_install(
            install_packet
        ).session
        reservation_bits = maximum_compact_update_bits(sender_session)
        maximum_reservations = int(
            math.ceil(
                float(grid["maximum_update_reservation_fraction"])
                * len(evaluation_indices)
            )
        )
        dataset = build_temporal_hazard_dataset(
            cache[f"channel_power_n{n_channels}"],
            states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            codebook,
        )
        features = dataset["features"]
        labels = dataset["labels"].astype(np.uint8)
        groups = dataset["groups"].astype(str)
        training = np.isin(groups, np.asarray(train_groups))
        evaluation = np.isin(groups, np.asarray(evaluation_groups))
        selected_c = float(
            predecessors["temporal_hazard_result"]["results"][
                str(n_channels)
            ]["selected_c"]
        )
        oof, _ = _fit_oof(
            features[training],
            labels[training],
            groups[training],
            c_value=selected_c,
            maximum_iterations=5000,
        )
        estimator = _model(selected_c, maximum_iterations=5000)
        estimator.fit(features[training], labels[training])
        scores = np.full(labels.size, np.nan)
        scores[training] = oof
        scores[evaluation] = estimator.predict_proba(
            features[evaluation]
        )[:, 1]
        threshold = _upper_quantile_threshold(
            scores[training],
            float(grid["maximum_update_reservation_fraction"]),
        )
        candidates = _score_candidates(
            dataset["source_indices"],
            scores,
            evaluation,
            threshold=threshold,
            state_count=len(states),
        )
        fixed_intervals = np.full(
            len(states),
            int(grid["heartbeat_silence_scenes"]),
            dtype=np.int64,
        )
        exact_bits = _exact_bundle_bits(
            n_channels,
            queries,
            header_bits=int(grid["task_header_bits"]),
        )
        seed = (
            int(monte_carlo["seed"])
            + n_position * 100_000
            + int(
                monte_carlo[
                    "heartbeat_predecessor_condition_seed_offset"
                ]
            )
        )
        common_random = _common_random_with_frozen_prefix(
            trajectories=trajectory_count,
            scene_count=len(evaluation_indices),
            seed=seed,
        )
        run_cache = {}

        def run_condition(condition: dict):
            key = _condition_key(condition)
            if key in run_cache:
                return run_cache[key]
            full_runs = [
                simulate_trajectory(
                    states,
                    cache["timestamps_local"],
                    evaluation_indices,
                    sender_session=sender_session,
                    receiver_session=receiver_session,
                    install_packet=install_packet,
                    method="belief_risk_recovery",
                    condition=condition,
                    random_values=common_random[trajectory],
                    epsilon_db=float(grid["epsilon_db"]),
                    max_age_minutes=float(
                        grid["max_state_age_minutes"]
                    ),
                    ack_bits=ack_bits,
                    outage_penalty_db=float(
                        grid["outage_penalty_db"]
                    ),
                    heartbeat_request_bits=request_bits,
                    heartbeat_response_bits=response_bits,
                    heartbeat_intervals_by_state=fixed_intervals,
                    update_protection_candidates=candidates,
                    maximum_update_reservations=maximum_reservations,
                    reservation_equivalent_bits=reservation_bits,
                )
                for trajectory in range(trajectory_count)
            ]
            exact_runs = [
                _simulate_exact_query_bundle(
                    states,
                    evaluation_indices,
                    random_values=common_random[trajectory],
                    packet_loss_probability=float(
                        condition["task_loss_probability"]
                    ),
                    receiver_reset_probability=float(
                        condition[
                            "receiver_context_reset_probability"
                        ]
                    ),
                    packet_bits=exact_bits,
                    epsilon_db=float(grid["epsilon_db"]),
                    outage_penalty_db=float(
                        grid["outage_penalty_db"]
                    ),
                )
                for trajectory in range(trajectory_count)
            ]
            run_cache[key] = {
                "full_candidate": full_runs,
                "exact_query_bundle_reference": exact_runs,
            }
            return run_cache[key]

        baseline_runs = run_condition(baseline_condition)
        baseline_summaries = {
            method: summarize_trajectories(
                runs, scene_count=len(evaluation_indices)
            )
            for method, runs in baseline_runs.items()
        }
        frozen_methods = predecessors["component_ablation_result"][
            "results"
        ][str(n_channels)]["methods"]
        baseline_reproduction[str(n_channels)] = {
            "full_candidate": baseline_summaries["full_candidate"]
            == frozen_methods["full_candidate"],
            "exact_query_bundle_reference": baseline_summaries[
                "exact_query_bundle_reference"
            ]
            == frozen_methods["exact_query_bundle_reference"],
        }
        panels = {}
        for dimension_position, (dimension, levels) in enumerate(
            protocol["one_factor_scans"].items()
        ):
            level_rows = {}
            lowest_runs = None
            lowest_value = float(levels[0])
            for level_position, level_value in enumerate(levels):
                condition = dict(baseline_condition)
                condition[dimension] = float(level_value)
                runs = run_condition(condition)
                if lowest_runs is None:
                    lowest_runs = runs
                summaries = {
                    method: summarize_trajectories(
                        method_runs,
                        scene_count=len(evaluation_indices),
                    )
                    for method, method_runs in runs.items()
                }
                comparisons = {
                    f"{method}_minus_lowest_fault_level": (
                        _paired_intervals(
                            runs[method],
                            lowest_runs[method],
                            scene_count=len(evaluation_indices),
                            replicates=int(
                                monte_carlo[
                                    "paired_bootstrap_replicates"
                                ]
                            ),
                            confidence_level=float(
                                monte_carlo["confidence_level"]
                            ),
                            seed=(
                                seed
                                + 100_000
                                + dimension_position * 10_000
                                + level_position * 100
                                + method_position
                            ),
                        )
                    )
                    for method_position, method in enumerate(
                        protocol["methods"]
                    )
                }
                level_rows[format(float(level_value), ".6g")] = {
                    "fault_value": float(level_value),
                    "condition": {
                        field: float(condition[field])
                        for field in FAULT_FIELDS
                    },
                    "methods": summaries,
                    "paired_change_from_lowest_fault_level": comparisons,
                }
            ordered_rows = list(level_rows.values())
            full_clean = [
                row["methods"]["full_candidate"]["clean_rate"]
                for row in ordered_rows
            ]
            exact_clean = [
                row["methods"]["exact_query_bundle_reference"][
                    "clean_rate"
                ]
                for row in ordered_rows
            ]
            full_bits = [
                row["methods"]["full_candidate"][
                    "mean_application_bits_per_scene"
                ]
                for row in ordered_rows
            ]
            panels[dimension] = {
                "lowest_fault_value": lowest_value,
                "highest_fault_value": float(levels[-1]),
                "levels": level_rows,
                "endpoint": {
                    "full_candidate_clean_drop_percentage_points": float(
                        100.0 * (full_clean[0] - full_clean[-1])
                    ),
                    "exact_reference_clean_drop_percentage_points": float(
                        100.0 * (exact_clean[0] - exact_clean[-1])
                    ),
                    "full_candidate_application_bit_change_per_scene": float(
                        full_bits[-1] - full_bits[0]
                    ),
                    "full_candidate_clean_monotonic_nonincreasing": all(
                        later <= earlier + 1e-12
                        for earlier, later in zip(
                            full_clean[:-1], full_clean[1:]
                        )
                    ),
                    "exact_reference_clean_monotonic_nonincreasing": all(
                        later <= earlier + 1e-12
                        for earlier, later in zip(
                            exact_clean[:-1], exact_clean[1:]
                        )
                    ),
                },
            }
        all_results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "codebook_size": len(codebook.codewords),
            "exact_query_bundle_bits": exact_bits,
            "panels": panels,
        }

    dimension_ranking = []
    for dimension in FAULT_FIELDS:
        drops = [
            all_results[str(n)]["panels"][dimension]["endpoint"][
                "full_candidate_clean_drop_percentage_points"
            ]
            for n in grid["n_channels"]
        ]
        dimension_ranking.append(
            {
                "fault_dimension": dimension,
                "mean_endpoint_clean_drop_percentage_points": float(
                    np.mean(drops)
                ),
                "minimum_across_n": float(np.min(drops)),
                "maximum_across_n": float(np.max(drops)),
            }
        )
    dimension_ranking.sort(
        key=lambda row: row[
            "mean_endpoint_clean_drop_percentage_points"
        ],
        reverse=True,
    )
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "frozen_baseline_reproduced": all(
            all(value.values()) for value in baseline_reproduction.values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            row["methods"]["full_candidate"][
                "wrong_codebook_decode_count"
            ]
            == 0
            for result in all_results.values()
            for panel in result["panels"].values()
            for row in panel["levels"].values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
        "frozen_architecture_change_forbidden": not rules[
            "results_may_modify_frozen_architecture_or_parameters"
        ],
    }
    if not all(checks.values()):
        raise AssertionError(f"fault stress failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_fault_probability_stress_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "split": {
            "training_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_scene_count": int(len(evaluation_indices)),
        },
        "results": all_results,
        "fault_dimension_ranking": dimension_ranking,
        "baseline_reproduction": baseline_reproduction,
        "checks": checks,
        "governance": {
            "development_only": True,
            "faults_are_controlled_injections_not_measurements": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "environment": environment_snapshot(
            ["numpy", "scikit-learn"]
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "fault_dimension_ranking": dimension_ranking,
                "endpoint_by_n": {
                    n: {
                        dimension: panel["endpoint"]
                        for dimension, panel in row["panels"].items()
                    }
                    for n, row in all_results.items()
                },
                "checks": checks,
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
