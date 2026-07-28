#!/usr/bin/env python
"""Evaluate frozen Stage-5 event semantics under changing bandwidth queries."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from run_stage5_event_semantics_development import (  # noqa: E402
    guarded_state_start,
    load_json,
    ordered_site_indices,
    summarize_rows,
)
from spectrum_semcom.c1_temporal_statistics import paired_cluster_bootstrap  # noqa: E402
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    EventDecision,
    QueryState,
    absolute_index_width,
    block_regret,
    choose_event_update,
    contiguous_block_costs,
    decode_update,
    encode_update,
    state_age_minutes,
)


METHODS = (
    "query_absolute_index_always",
    "query_periodic_absolute_60min",
    "query_event_absolute_fixed",
    "query_event_delta_fixed",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def validate_inputs(protocol_path: Path, protocol: dict) -> tuple[dict, dict, dict]:
    source_path = PROJECT_DIR / protocol["development_source"]["cache_result"]
    policy_path = PROJECT_DIR / protocol["fixed_policy_from_s5_1"]["source_result"]
    stage2_path = PROJECT_DIR / protocol["link"]["stage2_config"]
    for path in (source_path, stage2_path):
        if "c1_v4_final" in path.as_posix().lower():
            raise ValueError("dynamic-query development must not load Final paths")
    expected = (
        (source_path, protocol["development_source"]["cache_result_sha256"]),
        (policy_path, protocol["fixed_policy_from_s5_1"]["source_result_sha256"]),
        (stage2_path, protocol["link"]["stage2_config_sha256"]),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"frozen dynamic-query input hash mismatch: {path}")
    if protocol["fixed_policy_from_s5_1"]["parameters_may_be_retuned"]:
        raise ValueError("dynamic-query experiment must not retune S5.1 parameters")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["may_modify_stage4_final_algorithm_or_claim"]
    ):
        raise ValueError("dynamic-query governance is not Final-closed")
    cache_result = load_json(source_path)
    split = protocol["development_source"]["split"]
    split_meta = cache_result["splits"][split]
    if split_meta["cache_sha256"] != protocol["development_source"]["cache_sha256"]:
        raise ValueError("dynamic-query split declaration changed")
    cache_path = PROJECT_DIR / split_meta["cache"]
    if sha256_file(cache_path) != split_meta["cache_sha256"]:
        raise ValueError("dynamic-query cache hash mismatch")
    cache = load_npz(cache_path)
    return cache, load_json(stage2_path), cache_result


def build_query_schedule(
    cache: dict[str, np.ndarray],
    protocol: dict,
    schedule: str,
) -> np.ndarray:
    demands = np.asarray(
        protocol["resource_queries"]["demand_channels"],
        dtype=int,
    )
    output = np.zeros(len(cache["scene_ids"]), dtype=int)
    site_sequences = ordered_site_indices(cache)
    if schedule == "cyclic":
        for site_index, positions in enumerate(site_sequences):
            output[positions] = demands[
                (np.arange(len(positions)) + site_index) % len(demands)
            ]
        return output
    if schedule != "markov":
        raise KeyError(schedule)
    spec = protocol["resource_queries"]["schedules"]["markov"]
    transition = np.asarray(spec["transition_matrix"], dtype=np.float64)
    if transition.shape != (len(demands), len(demands)):
        raise ValueError("query transition matrix shape mismatch")
    np.testing.assert_allclose(np.sum(transition, axis=1), 1.0)
    for site_index, positions in enumerate(site_sequences):
        rng = np.random.default_rng(int(spec["seed"]) + site_index)
        state = site_index % len(demands)
        values = []
        for _ in positions:
            values.append(int(demands[state]))
            state = int(rng.choice(len(demands), p=transition[state]))
        output[positions] = values
    return output


def _absolute_decision(costs: np.ndarray, header_bits: int, reason: str) -> EventDecision:
    payload = absolute_index_width(len(costs))
    return EventDecision(
        "absolute",
        int(np.argmin(costs)),
        0.0,
        None,
        int(header_bits) + payload,
        payload,
        reason,
    )


def choose_method_decision(
    method: str,
    costs: np.ndarray,
    *,
    state: QueryState | None,
    timestamp_local: str,
    demand_channels: int,
    protocol: dict,
) -> EventDecision:
    header = int(protocol["codec"]["compact_application_header_bits"])
    fixed = protocol["fixed_policy_from_s5_1"]
    if method == "query_absolute_index_always":
        return _absolute_decision(costs, header, "query_absolute_every_scene")
    if method == "query_periodic_absolute_60min":
        if state is None or state_age_minutes(state, timestamp_local) >= 60.0:
            return _absolute_decision(costs, header, "query_periodic_refresh")
        return EventDecision(
            "silence",
            int(np.argmin(costs)),
            block_regret(costs, state.selected_start),
            state_age_minutes(state, timestamp_local),
            0,
            0,
            "query_periodic_wait",
        )
    decision = choose_event_update(
        costs,
        state=state,
        timestamp_local=timestamp_local,
        regret_threshold_db=float(fixed["regret_threshold_db"]),
        max_age_minutes=float(fixed["max_age_minutes"]),
        demand_channels=demand_channels,
        application_header_bits=header,
    )
    if method == "query_event_delta_fixed" or not decision.transmits:
        return decision
    if method != "query_event_absolute_fixed":
        raise KeyError(method)
    return _absolute_decision(
        costs,
        header,
        f"{decision.reason}_forced_absolute",
    )


def method_max_age(method: str, protocol: dict) -> float:
    if method == "query_periodic_absolute_60min":
        return 60.0
    return float(protocol["fixed_policy_from_s5_1"]["max_age_minutes"])


def run_schedule(
    cache: dict[str, np.ndarray],
    stage2: dict,
    protocol: dict,
    schedule_name: str,
    query: np.ndarray,
    *,
    schedule_seed: int,
) -> tuple[dict, dict[str, list[dict]]]:
    channel_models = protocol["link"]["channel_models"]
    ebn0_values = protocol["link"]["ebn0_db"]
    repeats = int(protocol["link"]["repeats"])
    header = int(protocol["codec"]["compact_application_header_bits"])
    n_channels = int(protocol["resource_queries"]["n_channels"])
    count = len(cache["scene_ids"])
    site_sequences = ordered_site_indices(cache)
    accumulators = {
        method: {
            key: np.zeros(count)
            for key in (
                "regret",
                "bits",
                "sent",
                "success",
                "silence",
                "absolute",
                "delta",
            )
        }
        for method in METHODS
    }
    tx_cache: dict[tuple, object] = {}
    trajectories = len(channel_models) * len(ebn0_values) * repeats
    for channel_index, channel in enumerate(channel_models):
        for ebn0_index, ebn0 in enumerate(ebn0_values):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(repeats):
                for method in METHODS:
                    for positions in site_sequences:
                        states: dict[int, QueryState] = {}
                        for position in positions:
                            demand = int(query[position])
                            candidates = n_channels - demand + 1
                            timestamp = str(cache["timestamps_local"][position])
                            costs = contiguous_block_costs(
                                cache["channel_power_dbm"][position],
                                demand,
                            )
                            state = states.get(demand)
                            decision = choose_method_decision(
                                method,
                                costs,
                                state=state,
                                timestamp_local=timestamp,
                                demand_channels=demand,
                                protocol=protocol,
                            )
                            values = accumulators[method]
                            values[decision.mode][position] += 1.0
                            max_age = method_max_age(method, protocol)
                            if not decision.transmits:
                                selected = guarded_state_start(
                                    state,
                                    timestamp,
                                    max_age_minutes=max_age,
                                    candidate_blocks=candidates,
                                )
                                values["regret"][position] += block_regret(
                                    costs,
                                    selected,
                                )
                                continue
                            epoch = 0 if state is None else (state.epoch + 1) % 256
                            encoded = encode_update(
                                mode=decision.mode,
                                target_start=decision.target_start,
                                cached_start=None
                                if state is None
                                else state.selected_start,
                                candidate_blocks=candidates,
                                node_id=0,
                                demand_channels=demand,
                                epoch=epoch,
                                application_header_bits=header,
                            )
                            if encoded.size != decision.application_bits:
                                raise RuntimeError("dynamic query codec length mismatch")
                            seed = (
                                schedule_seed
                                + int(position) * 100_000
                                + channel_index * 1000
                                + ebn0_index * 100
                                + repetition
                            )
                            key = (
                                int(encoded.size),
                                channel,
                                float(ebn0),
                                seed,
                            )
                            if key not in tx_cache:
                                tx_cache[key] = transmit_payload_analytic(
                                    int(encoded.size),
                                    link,
                                    seed,
                                )
                            tx = tx_cache[key]
                            values["sent"][position] += 1.0
                            values["bits"][position] += float(tx.transmitted_bits)
                            if tx.frame_success:
                                decoded = decode_update(
                                    encoded,
                                    candidate_blocks=candidates,
                                    cached_start=None
                                    if state is None
                                    else state.selected_start,
                                    application_header_bits=header,
                                )
                                states[demand] = QueryState(
                                    decoded.selected_start,
                                    timestamp,
                                    decoded.epoch,
                                )
                                selected = decoded.selected_start
                                values["success"][position] += 1.0
                            else:
                                selected = guarded_state_start(
                                    state,
                                    timestamp,
                                    max_age_minutes=max_age,
                                    candidate_blocks=candidates,
                                )
                            values["regret"][position] += block_regret(
                                costs,
                                selected,
                            )
            print(
                f"{schedule_name}: {channel} {float(ebn0):g} dB complete",
                flush=True,
            )
    rows_by_method = {}
    summary = {}
    alpha = float(protocol["statistics"]["cvar_alpha"])
    for method in METHODS:
        values = accumulators[method]
        rows = []
        for position in range(count):
            rows.append(
                {
                    "schedule": schedule_name,
                    "method": method,
                    "scene_id": str(cache["scene_ids"][position]),
                    "site": str(cache["site_ids"][position]),
                    "timestamp_local": str(cache["timestamps_local"][position]),
                    "cluster_id": str(cache["cluster_ids"][position]),
                    "demand_channels": int(query[position]),
                    "regret_db": float(values["regret"][position] / trajectories),
                    "actual_bits": float(values["bits"][position] / trajectories),
                    "transmission_rate": float(
                        values["sent"][position] / trajectories
                    ),
                    "successful_update_rate": float(
                        values["success"][position] / trajectories
                    ),
                    "silence_rate": float(
                        values["silence"][position] / trajectories
                    ),
                    "absolute_rate": float(
                        values["absolute"][position] / trajectories
                    ),
                    "delta_rate": float(values["delta"][position] / trajectories),
                }
            )
        rows_by_method[method] = rows
        summary[method] = summarize_rows(rows, alpha)
    baseline = rows_by_method["query_absolute_index_always"]
    comparisons = {}
    for index, method in enumerate(METHODS[1:]):
        proposed = rows_by_method[method]
        inference = paired_cluster_bootstrap(
            np.asarray([row["regret_db"] for row in proposed]),
            np.asarray([row["regret_db"] for row in baseline]),
            np.asarray([row["actual_bits"] for row in proposed]),
            np.asarray([row["actual_bits"] for row in baseline]),
            [row["cluster_id"] for row in baseline],
            repetitions=int(
                protocol["statistics"]["paired_cluster_bootstrap_repetitions"]
            ),
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + schedule_seed
            + index,
            cvar_alpha=alpha,
            one_sided_confidence=float(
                protocol["statistics"]["one_sided_confidence"]
            ),
        )
        baseline_bits = summary["query_absolute_index_always"][
            "mean_actual_bits"
        ]
        proposed_bits = summary[method]["mean_actual_bits"]
        inference["actual_bit_reduction_pct"] = float(
            100.0 * (baseline_bits - proposed_bits) / baseline_bits
        )
        gate = protocol["descriptive_gate_per_schedule"]
        inference["passes_descriptive_gate"] = bool(
            inference["actual_bit_reduction_pct"]
            >= float(gate["minimum_actual_bit_reduction_pct"])
            and inference["mean_regret_upper_bound"]
            <= float(gate["maximum_mean_regret_upper_bound_db"])
            and inference["cvar_upper_bound"]
            <= float(gate["maximum_cvar_upper_bound_db"])
        )
        comparisons[method] = inference
    output = {
        "demand_counts": {
            str(demand): int(np.count_nonzero(query == demand))
            for demand in sorted(set(query))
        },
        "summary": summary,
        "comparisons_vs_query_absolute_index_always": comparisons,
        "diagnostics": {
            "trajectory_count_per_scene": trajectories,
            "unique_transmission_outcomes_cached": len(tx_cache),
        },
        "rows": [
            row for method in METHODS for row in rows_by_method[method]
        ],
    }
    return output, rows_by_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_dynamic_query_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/dynamic_query_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite dynamic-query output")
    protocol = load_json(args.protocol)
    cache, stage2, cache_result = validate_inputs(args.protocol, protocol)
    schedules = {}
    for index, name in enumerate(
        protocol["resource_queries"]["schedules"].keys()
    ):
        query = build_query_schedule(cache, protocol, name)
        schedules[name], _ = run_schedule(
            cache,
            stage2,
            protocol,
            name,
            query,
            schedule_seed=1000 * (index + 1),
        )
    result = {
        "version": "1.0",
        "status": "stage5_dynamic_query_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_result_sha256": protocol["development_source"][
                "cache_result_sha256"
            ],
            "validation_cache_sha256": cache_result["splits"][
                protocol["development_source"]["split"]
            ]["cache_sha256"],
            "s5_1_result_sha256": protocol["fixed_policy_from_s5_1"][
                "source_result_sha256"
            ],
            "stage2_config_sha256": protocol["link"]["stage2_config_sha256"],
        },
        "fixed_policy": {
            key: protocol["fixed_policy_from_s5_1"][key]
            for key in ("regret_threshold_db", "max_age_minutes")
        },
        "schedules": schedules,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "data_role": protocol["development_source"]["role"],
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "dynamic_query_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "schedules": list(schedules),
                "event_delta_gate": {
                    name: row["comparisons_vs_query_absolute_index_always"][
                        "query_event_delta_fixed"
                    ]["passes_descriptive_gate"]
                    for name, row in schedules.items()
                },
                "final_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
