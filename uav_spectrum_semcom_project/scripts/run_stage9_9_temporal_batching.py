#!/usr/bin/env python
"""Run the preregistered Stage-9.9 temporal batching comparison."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage9_9_temporal_batching import (  # noqa: E402
    build_immediate_schedule,
    empirical_cvar,
    markov_loss_mask,
    replay_frozen_schedule,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _access_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    return {
        relative: sha256_file(PROJECT_DIR / relative)
        for relative in protocol["access_state_sha256_before"]
    }


def _strip_regrets(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "scene_regrets_db"}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty policy")
    scenes = int(sum(row["scene_count"] for row in rows))
    events = int(sum(row["event_count"] for row in rows))
    regrets = np.concatenate(
        [np.asarray(row["scene_regrets_db"], dtype=np.float64) for row in rows]
    )
    if regrets.size != scenes:
        raise AssertionError("regret aggregation lost scenes")
    return {
        "temporal_group_count": int(len(rows)),
        "scene_count": scenes,
        "event_count": events,
        "datagram_count": int(sum(row["datagram_count"] for row in rows)),
        "h8_update_path_bits": int(sum(row["h8_update_path_bits"] for row in rows)),
        "clean_rate": float(sum(row["clean_scene_count"] for row in rows) / scenes),
        "stale_action_scene_count": int(
            sum(row["stale_action_scene_count"] for row in rows)
        ),
        "mean_max_query_regret_db": float(np.mean(regrets)),
        "cvar_0_9_max_query_regret_db": empirical_cvar(regrets, 0.9),
        "maximum_max_query_regret_db": float(np.max(regrets)),
        "mean_event_wait_scenes": float(
            sum(row["mean_event_wait_scenes"] * row["event_count"] for row in rows)
            / max(events, 1)
        ),
        "maximum_event_wait_scenes": int(
            max(row["maximum_event_wait_scenes"] for row in rows)
        ),
        "mean_event_wait_seconds": float(
            sum(row["mean_event_wait_seconds"] * row["event_count"] for row in rows)
            / max(events, 1)
        ),
        "maximum_event_wait_seconds": float(
            max(row["maximum_event_wait_seconds"] for row in rows)
        ),
    }


def _comparison_point(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, float]:
    base = _aggregate(baseline)
    cand = _aggregate(candidate)
    bit_saving = (
        0.0
        if base["h8_update_path_bits"] == 0
        else 100.0
        * (base["h8_update_path_bits"] - cand["h8_update_path_bits"])
        / base["h8_update_path_bits"]
    )
    return {
        "update_bit_saving_percent": float(bit_saving),
        "clean_difference_pp": float(100.0 * (cand["clean_rate"] - base["clean_rate"])),
        "mean_regret_increase_db": float(
            cand["mean_max_query_regret_db"] - base["mean_max_query_regret_db"]
        ),
    }


def _paired_group_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("paired group bootstrap requires aligned groups")
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        "update_bit_saving_percent": [],
        "clean_difference_pp": [],
        "mean_regret_increase_db": [],
    }
    for _ in range(int(repetitions)):
        picked = rng.integers(0, len(baseline), size=len(baseline))
        base_rows = [baseline[int(index)] for index in picked]
        candidate_rows = [candidate[int(index)] for index in picked]
        point = _comparison_point(base_rows, candidate_rows)
        for key in samples:
            samples[key].append(point[key])
    output: dict[str, Any] = {
        "analysis_unit": "temporal_group",
        "group_count": int(len(baseline)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "point": _comparison_point(baseline, candidate),
    }
    for key, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        output[key + "_ci95"] = [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ]
    return output


def _make_workloads(
    cache: dict[str, np.ndarray], protocol: dict[str, Any]
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    groups = cache["cluster_ids"].astype(str)
    split = protocol["fixed_split"]
    training_groups = tuple(split["training_groups"])
    evaluation_groups = tuple(split["evaluation_groups"])
    if set(training_groups) & set(evaluation_groups):
        raise ValueError("training and evaluation groups overlap")
    if set(groups.tolist()) != set(training_groups) | set(evaluation_groups):
        raise ValueError("cache groups do not match the frozen split")
    train_indices = np.flatnonzero(np.isin(groups, np.asarray(training_groups)))
    grid = protocol["task_grid"]
    workloads: dict[int, list[dict[str, Any]]] = {}
    codebook_summary: dict[str, Any] = {}
    for n_value in grid["n_channels"]:
        n_channels = int(n_value)
        demands = tuple(
            int(round(n_channels * float(ratio)))
            for ratio in grid["demand_ratios"]
        )
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
        decisions = [encode_task_state(codebook, state) for state in states]
        rows = []
        for group_id in evaluation_groups:
            indices = np.flatnonzero(groups == group_id)
            group_states = [states[int(index)] for index in indices]
            group_decisions = [decisions[int(index)] for index in indices]
            timestamps = cache["timestamps_local"][indices].astype(str).tolist()
            initial, events = build_immediate_schedule(group_states, group_decisions)
            rows.append(
                {
                    "group_id": group_id,
                    "states": group_states,
                    "timestamps": timestamps,
                    "initial_actions": initial,
                    "events": events,
                }
            )
        workloads[n_channels] = rows
        codebook_summary[str(n_channels)] = {
            "demands": list(demands),
            "training_scene_count": int(train_indices.size),
            "codebook_size": int(len(codebook.codewords)),
            "symbol_width_bits": int(codebook.symbol_width_bits),
            "training_coverage_rate": float(
                codebook.training_covered_count / codebook.training_scene_count
            ),
        }
    return workloads, codebook_summary


def _no_loss_results(
    workloads: dict[int, list[dict[str, Any]]], protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[int, list[dict[str, Any]]]]]:
    semantics = protocol["frozen_batch_semantics"]
    batch_sizes = [1] + [int(row["batch_size"]) for row in protocol["candidates"]]
    internal: dict[int, dict[int, list[dict[str, Any]]]] = {}
    result: dict[str, Any] = {}
    statistics = protocol["statistics"]
    for n_channels, groups in workloads.items():
        by_batch: dict[int, list[dict[str, Any]]] = {size: [] for size in batch_sizes}
        per_group = []
        for workload in groups:
            policy_rows = {}
            for size in batch_sizes:
                replay = replay_frozen_schedule(
                    workload["states"],
                    workload["timestamps"],
                    workload["initial_actions"],
                    workload["events"],
                    batch_size=size,
                    item_bits=int(semantics["normal_update_protocol_bits_per_item"]),
                    header_bits=int(semantics["h8_header_bits_per_datagram"]),
                )
                by_batch[size].append(replay)
                policy_rows[f"B{size}"] = _strip_regrets(replay)
            per_group.append({"group_id": workload["group_id"], "policies": policy_rows})
        internal[n_channels] = by_batch
        aggregates = {f"B{size}": _aggregate(by_batch[size]) for size in batch_sizes}
        comparisons = {}
        for offset, size in enumerate(batch_sizes[1:]):
            comparisons[f"B{size}_vs_B1"] = _paired_group_bootstrap(
                by_batch[1],
                by_batch[size],
                repetitions=int(statistics["paired_group_bootstrap_repetitions"]),
                seed=int(statistics["paired_group_bootstrap_seed"])
                + 100 * int(n_channels)
                + offset,
            )
        result[str(n_channels)] = {
            "aggregates": aggregates,
            "comparisons": comparisons,
            "per_group": per_group,
        }
    return result, internal


def _burst_results(
    workloads: dict[int, list[dict[str, Any]]], protocol: dict[str, Any]
) -> dict[str, Any]:
    stress = protocol["burst_loss_stress"]
    semantics = protocol["frozen_batch_semantics"]
    batch_sizes = [1] + [int(row["batch_size"]) for row in protocol["candidates"]]
    runs = int(stress["runs"])
    result: dict[str, Any] = {}
    for n_channels, groups in workloads.items():
        clean_by_batch = {size: [] for size in batch_sizes}
        loss_by_batch = {size: [] for size in batch_sizes}
        for run_index in range(runs):
            masks = {}
            for group_index, workload in enumerate(groups):
                masks[workload["group_id"]] = markov_loss_mask(
                    len(workload["states"]),
                    seed=int(stress["seed"]) + 10000 * run_index + group_index,
                    good_to_bad=float(stress["good_to_bad_probability_per_scene"]),
                    bad_to_good=float(stress["bad_to_good_probability_per_scene"]),
                    drop_good=float(stress["drop_probability_good"]),
                    drop_bad=float(stress["drop_probability_bad"]),
                    stationary_initial_state=bool(stress["stationary_initial_state"]),
                )
            for size in batch_sizes:
                clean_count = 0
                scene_count = 0
                lost = 0
                datagrams = 0
                for workload in groups:
                    replay = replay_frozen_schedule(
                        workload["states"],
                        workload["timestamps"],
                        workload["initial_actions"],
                        workload["events"],
                        batch_size=size,
                        loss_mask=masks[workload["group_id"]],
                        item_bits=int(semantics["normal_update_protocol_bits_per_item"]),
                        header_bits=int(semantics["h8_header_bits_per_datagram"]),
                    )
                    clean_count += replay["clean_scene_count"]
                    scene_count += replay["scene_count"]
                    lost += replay["lost_datagram_count"]
                    datagrams += replay["datagram_count"]
                clean_by_batch[size].append(clean_count / scene_count)
                loss_by_batch[size].append(lost / max(datagrams, 1))
        policies = {}
        baseline = np.asarray(clean_by_batch[1], dtype=np.float64)
        for size in batch_sizes:
            clean = np.asarray(clean_by_batch[size], dtype=np.float64)
            loss = np.asarray(loss_by_batch[size], dtype=np.float64)
            row: dict[str, Any] = {
                "run_count": runs,
                "mean_clean_rate": float(np.mean(clean)),
                "clean_rate_ci95_across_runs": [
                    float(np.quantile(clean, 0.025)),
                    float(np.quantile(clean, 0.975)),
                ],
                "mean_realized_datagram_loss_rate": float(np.mean(loss)),
            }
            if size != 1:
                difference = 100.0 * (clean - baseline)
                row["paired_clean_difference_vs_B1_pp"] = float(np.mean(difference))
                row["paired_clean_difference_vs_B1_pp_ci95"] = [
                    float(np.quantile(difference, 0.025)),
                    float(np.quantile(difference, 0.975)),
                ]
            policies[f"B{size}"] = row
        result[str(n_channels)] = policies
    return result


def _gates(
    no_loss: dict[str, Any], burst: dict[str, Any], protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    rule = protocol["retention_rule"]
    detail: dict[str, Any] = {}
    retained: dict[str, bool] = {}
    for candidate in protocol["candidates"]:
        name = f"B{int(candidate['batch_size'])}"
        scale_rows = {}
        for n_value in protocol["task_grid"]["n_channels"]:
            n_key = str(int(n_value))
            comparison = no_loss[n_key]["comparisons"][f"{name}_vs_B1"]
            burst_row = burst[n_key][name]
            checks = {
                "bit_saving_lower_bound": comparison[
                    "update_bit_saving_percent_ci95"
                ][0]
                >= float(rule["minimum_update_bit_saving_lower_95_percent_bound"]),
                "no_loss_clean_lower_bound": comparison[
                    "clean_difference_pp_ci95"
                ][0]
                >= float(rule["minimum_no_loss_clean_difference_lower_95_percent_bound_pp"]),
                "mean_regret_upper_bound": comparison[
                    "mean_regret_increase_db_ci95"
                ][1]
                <= float(rule["maximum_mean_regret_increase_upper_95_percent_bound_db"]),
                "burst_clean_lower_bound": burst_row[
                    "paired_clean_difference_vs_B1_pp_ci95"
                ][0]
                >= float(rule["minimum_burst_clean_difference_lower_95_percent_bound_pp"]),
            }
            scale_rows[n_key] = {"checks": checks, "passed": bool(all(checks.values()))}
        detail[name] = scale_rows
        retained[name] = bool(all(row["passed"] for row in scale_rows.values()))
    return detail, retained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage9_9_temporal_batching_v1.json",
    )
    parser.add_argument("--phase", choices=("confirmation", "reproduction"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = _read_json(args.protocol)
    governance = protocol["governance"]
    if (
        not governance["development_only"]
        or governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["external_final_access_may_be_consumed"]
        or governance["output_is_confirmatory_final"]
    ):
        raise ValueError("Stage-9.9 governance does not permit execution")
    access_before = _access_hashes(protocol)
    if access_before != protocol["access_state_sha256_before"]:
        raise ValueError("a protected access-state file changed after preregistration")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("development cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    workloads, codebook_summary = _make_workloads(cache, protocol)
    no_loss, _ = _no_loss_results(workloads, protocol)
    burst = _burst_results(workloads, protocol)
    gate_detail, retained = _gates(no_loss, burst, protocol)
    access_after = _access_hashes(protocol)
    if access_after != access_before:
        raise AssertionError("protected access state changed during Stage-9.9")
    result: dict[str, Any] = {
        "schema_version": "stage9_9_temporal_batching_result_v1",
        "run_phase": args.phase,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "candidate_ids": [
            protocol["baseline"]["candidate_id"],
            *[row["candidate_id"] for row in protocol["candidates"]],
        ],
        "data_governance": {
            "source_role": source["role"],
            "cache_sha256_verified": True,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
            "protected_access_state_unchanged": access_after == access_before,
            "access_state_sha256": access_after,
        },
        "split": protocol["fixed_split"],
        "codebook_summary": codebook_summary,
        "no_loss_temporal_replay": no_loss,
        "burst_loss_stress": {
            "model": protocol["burst_loss_stress"],
            "results": burst,
        },
        "retention_gates": gate_detail,
        "retention_decision": {
            "candidate_retained": retained,
            "batching_adopted": bool(any(retained.values())),
            "fallback": (
                None
                if any(retained.values())
                else protocol["retention_rule"]["failure_action"]
            ),
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    normalized = copy.deepcopy(result)
    normalized.pop("run_phase", None)
    result["normalized_result_sha256"] = canonical_json_sha256(normalized)
    atomic_write_json(args.output, result)
    print(json.dumps(result["retention_decision"], ensure_ascii=False, indent=2))
    print(f"normalized_result_sha256={result['normalized_result_sha256']}")


if __name__ == "__main__":
    main()
