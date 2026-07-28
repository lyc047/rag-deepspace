#!/usr/bin/env python
"""Benchmark the frozen Stage-6 candidate on the excluded development cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

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
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6_temporal_hazard import (  # noqa: E402
    build_temporal_hazard_dataset,
)


def _measure(
    operation,
    *,
    repetitions: int,
    warmups: int,
    divisor: float,
    unit_scale: float,
) -> dict:
    for _ in range(warmups):
        operation()
    values = []
    gc_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            operation()
            elapsed = time.perf_counter_ns() - start
            values.append(elapsed / divisor / unit_scale)
    finally:
        if gc_enabled:
            gc.enable()
    array = np.asarray(values, dtype=np.float64)
    return {
        "repetitions": int(repetitions),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _scaling_exponent(
    n_values: list[int], metric_values: list[float]
) -> float:
    n = np.asarray(n_values, dtype=np.float64)
    metric = np.asarray(metric_values, dtype=np.float64)
    if np.any(metric <= 0.0):
        raise ValueError("scaling metric must be positive")
    return float(np.polyfit(np.log2(n), np.log2(metric), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_complexity_benchmark_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/complexity_benchmark_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "complexity_benchmark_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite complexity benchmark")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        governance["frozen_algorithm_or_parameters_may_be_modified"]
        or governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("invalid complexity benchmark governance")
    for specification in (
        protocol["architecture_freeze"],
        protocol["frozen_model_source"],
    ):
        path = PROJECT_DIR / specification[
            "result" if "result" in specification else "path"
        ]
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"frozen benchmark input changed: {path}")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("complexity cache changed")
    access_path = (
        PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    )
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    model_result = json.loads(
        (PROJECT_DIR / protocol["frozen_model_source"]["path"]).read_text(
            encoding="utf-8"
        )
    )
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
    bench = protocol["benchmark"]
    warmups = int(bench["warmup_repetitions"])
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
    results = {}
    for n_position, n_value in enumerate(grid["n_channels"]):
        n_channels = int(n_value)
        powers = cache[f"channel_power_n{n_channels}"]
        demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
        queries = tuple(SpectrumTaskQuery(demand) for demand in demands)

        def build_all_states():
            return [
                build_task_state(
                    values,
                    queries,
                    epsilon_db=float(grid["epsilon_db"]),
                )
                for values in powers
            ]

        states = build_all_states()

        def fit_codebook():
            return fit_greedy_task_codebook(
                [states[int(index)] for index in train_indices]
            )

        codebook = fit_codebook()
        dataset = build_temporal_hazard_dataset(
            powers,
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
            model_result["results"][str(n_channels)]["selected_c"]
        )

        def fit_hazard():
            estimator = _model(selected_c, maximum_iterations=5000)
            estimator.fit(features[training], labels[training])
            return estimator

        estimator = fit_hazard()
        oof, _ = _fit_oof(
            features[training],
            labels[training],
            groups[training],
            c_value=selected_c,
            maximum_iterations=5000,
        )
        learned_scores = np.full(labels.size, np.nan)
        learned_scores[training] = oof
        learned_scores[evaluation] = estimator.predict_proba(
            features[evaluation]
        )[:, 1]
        threshold = _upper_quantile_threshold(
            learned_scores[training],
            float(grid["maximum_update_reservation_fraction"]),
        )
        candidates = _score_candidates(
            dataset["source_indices"],
            learned_scores,
            evaluation,
            threshold=threshold,
            state_count=len(states),
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
        fixed_intervals = np.full(
            len(states),
            int(grid["heartbeat_silence_scenes"]),
            dtype=np.int64,
        )
        maximum_reservations = int(
            math.ceil(
                float(grid["maximum_update_reservation_fraction"])
                * len(evaluation_indices)
            )
        )
        reservation_bits = maximum_compact_update_bits(sender_session)
        common_random = _common_random_with_frozen_prefix(
            trajectories=30,
            scene_count=len(evaluation_indices),
            seed=20260727 + n_position * 100_000 + 40_000,
        )[0]

        def encode_batch():
            return [
                encode_task_state(codebook, states[int(index)])
                for index in evaluation_indices
            ]

        def infer_batch():
            return estimator.predict_proba(features[evaluation])[:, 1]

        def run_controller():
            return simulate_trajectory(
                states,
                cache["timestamps_local"],
                evaluation_indices,
                sender_session=sender_session,
                receiver_session=receiver_session,
                install_packet=install_packet,
                method="belief_risk_recovery",
                condition=protocol["fault_condition"],
                random_values=common_random,
                epsilon_db=float(grid["epsilon_db"]),
                max_age_minutes=float(grid["max_state_age_minutes"]),
                ack_bits=ack_bits,
                outage_penalty_db=float(grid["outage_penalty_db"]),
                heartbeat_request_bits=request_bits,
                heartbeat_response_bits=response_bits,
                heartbeat_intervals_by_state=fixed_intervals,
                update_protection_candidates=candidates,
                maximum_update_reservations=maximum_reservations,
                reservation_equivalent_bits=reservation_bits,
            )

        timings = {
            "task_state_build_microseconds_per_scene": _measure(
                build_all_states,
                repetitions=int(bench["state_build_repetitions"]),
                warmups=warmups,
                divisor=len(states),
                unit_scale=1_000.0,
            ),
            "codebook_fit_milliseconds": _measure(
                fit_codebook,
                repetitions=int(bench["codebook_fit_repetitions"]),
                warmups=warmups,
                divisor=1.0,
                unit_scale=1_000_000.0,
            ),
            "hazard_model_fit_milliseconds": _measure(
                fit_hazard,
                repetitions=int(bench["hazard_fit_repetitions"]),
                warmups=warmups,
                divisor=1.0,
                unit_scale=1_000_000.0,
            ),
            "task_encode_microseconds_per_scene": _measure(
                encode_batch,
                repetitions=int(bench["encode_batch_repetitions"]),
                warmups=warmups,
                divisor=len(evaluation_indices),
                unit_scale=1_000.0,
            ),
            "hazard_inference_microseconds_per_scene": _measure(
                infer_batch,
                repetitions=int(bench["hazard_inference_repetitions"]),
                warmups=warmups,
                divisor=int(np.sum(evaluation)),
                unit_scale=1_000.0,
            ),
            "controller_microseconds_per_scene": _measure(
                run_controller,
                repetitions=int(
                    bench["controller_trajectory_repetitions"]
                ),
                warmups=warmups,
                divisor=len(evaluation_indices),
                unit_scale=1_000.0,
            ),
        }

        gc.collect()
        tracemalloc.start()
        memory_states = build_all_states()
        memory_codebook = fit_greedy_task_codebook(
            [memory_states[int(index)] for index in train_indices]
        )
        memory_dataset = build_temporal_hazard_dataset(
            powers,
            memory_states,
            cache["timestamps_local"],
            cache["cluster_ids"],
            memory_codebook,
        )
        memory_estimator = _model(selected_c, maximum_iterations=5000)
        memory_groups = memory_dataset["groups"].astype(str)
        memory_training = np.isin(
            memory_groups, np.asarray(train_groups)
        )
        memory_estimator.fit(
            memory_dataset["features"][memory_training],
            memory_dataset["labels"][memory_training],
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        controller_summary = summarize_trajectories(
            [run_controller()], scene_count=len(evaluation_indices)
        )
        decisions = encode_batch()
        results[str(n_channels)] = {
            "n_channels": n_channels,
            "demands": list(demands),
            "scene_count": len(states),
            "training_scene_count": int(len(train_indices)),
            "evaluation_scene_count": int(len(evaluation_indices)),
            "codebook_size": len(codebook.codewords),
            "codeword_symbol_width_bits": codebook.symbol_width_bits,
            "evaluation_fallback_count": int(
                sum(decision.uses_fallback for decision in decisions)
            ),
            "learned_candidate_count": int(
                np.sum(candidates[evaluation_indices])
            ),
            "maximum_update_reservations": maximum_reservations,
            "reservation_equivalent_bits": reservation_bits,
            "timings": timings,
            "end_to_end_peak_traced_bytes": int(peak_bytes),
            "end_to_end_peak_traced_mebibytes": float(
                peak_bytes / (1024.0**2)
            ),
            "controller_structural_output": {
                "clean_rate": controller_summary["clean_rate"],
                "application_bits_per_scene": controller_summary[
                    "mean_application_bits_per_scene"
                ],
                "wrong_codebook_decode_count": controller_summary[
                    "wrong_codebook_decode_count"
                ],
            },
        }

    n_values = [int(value) for value in grid["n_channels"]]
    timing_names = tuple(next(iter(results.values()))["timings"])
    scaling = {
        name: _scaling_exponent(
            n_values,
            [
                results[str(n)]["timings"][name]["median"]
                for n in n_values
            ],
        )
        for name in timing_names
    }
    scaling["end_to_end_peak_traced_mebibytes"] = _scaling_exponent(
        n_values,
        [
            results[str(n)]["end_to_end_peak_traced_mebibytes"]
            for n in n_values
        ],
    )
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_timings_positive": all(
            metric["minimum"] > 0.0
            for result in results.values()
            for metric in result["timings"].values()
        ),
        "wrong_codebook_decode_count_equals_zero": all(
            result["controller_structural_output"][
                "wrong_codebook_decode_count"
            ]
            == 0
            for result in results.values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
            and not access_after["final_method_outputs_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"complexity benchmark failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_complexity_benchmark_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "benchmark_classification": "environment_sensitive",
        "results": results,
        "empirical_log_log_scaling_exponent": scaling,
        "checks": checks,
        "environment": {
            **environment_snapshot(["numpy", "scikit-learn"]),
            "logical_cpu_count": os.cpu_count(),
            "processor": platform.processor(),
            "clock": bench["clock"],
        },
        "memory_limitation": bench["memory_limitation"],
        "governance": {
            "development_only": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "medians": {
                    n: {
                        **{
                            name: value["median"]
                            for name, value in row["timings"].items()
                        },
                        "end_to_end_peak_traced_mebibytes": row[
                            "end_to_end_peak_traced_mebibytes"
                        ],
                    }
                    for n, row in results.items()
                },
                "empirical_log_log_scaling_exponent": scaling,
                "checks": checks,
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
