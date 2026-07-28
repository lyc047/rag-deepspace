#!/usr/bin/env python
"""Evaluate a multi-query block-index bundle after the S5.2 cyclic failure."""

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
from run_stage5_dynamic_query_development import build_query_schedule  # noqa: E402
from run_stage5_event_semantics_development import (  # noqa: E402
    guarded_state_start,
    load_json,
    ordered_site_indices,
)
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
    paired_cluster_bootstrap,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    QueryState,
    absolute_index_width,
    block_regret,
    choose_event_update,
    contiguous_block_costs,
    decode_update,
    encode_update,
)
from spectrum_semcom.stage5_query_bundle import (  # noqa: E402
    decode_query_bundle,
    encode_query_bundle,
    query_bundle_payload_width,
)


METHODS = (
    "query_absolute_index_always",
    "query_event_delta_fixed",
    "query_bundle_always",
    "query_event_bundle_fixed",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def validate_inputs(protocol_path: Path, protocol: dict) -> tuple[dict, dict, dict]:
    source_path = PROJECT_DIR / protocol["development_source"]["cache_result"]
    schedule_path = (
        PROJECT_DIR / protocol["s5_2_boundary_evidence"]["schedule_protocol"]
    )
    boundary_path = PROJECT_DIR / protocol["s5_2_boundary_evidence"]["source_result"]
    stage2_path = PROJECT_DIR / protocol["link"]["stage2_config"]
    for path in (source_path, schedule_path, boundary_path, stage2_path):
        expected = {
            source_path: protocol["development_source"]["cache_result_sha256"],
            schedule_path: protocol["s5_2_boundary_evidence"][
                "schedule_protocol_sha256"
            ],
            boundary_path: protocol["s5_2_boundary_evidence"][
                "source_result_sha256"
            ],
            stage2_path: protocol["link"]["stage2_config_sha256"],
        }[path]
        if sha256_file(path) != expected:
            raise ValueError(f"frozen query-bundle input hash mismatch: {path}")
    if protocol["fixed_policy"]["parameters_may_be_retuned"]:
        raise ValueError("query bundle must not retune the fixed policy")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["fixed_policy_retuned"]
    ):
        raise ValueError("query-bundle governance is invalid")
    cache_result = load_json(source_path)
    split = protocol["development_source"]["split"]
    split_meta = cache_result["splits"][split]
    if split_meta["cache_sha256"] != protocol["development_source"]["cache_sha256"]:
        raise ValueError("query-bundle split hash declaration changed")
    cache_path = PROJECT_DIR / split_meta["cache"]
    if sha256_file(cache_path) != split_meta["cache_sha256"]:
        raise ValueError("query-bundle cache hash mismatch")
    return load_npz(cache_path), load_json(stage2_path), load_json(schedule_path)


def summarize(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows])
    sent = np.asarray([row["transmission_rate"] for row in rows])
    success = np.asarray([row["successful_update_rate"] for row in rows])
    return {
        "scene_count": len(rows),
        "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_transmission_rate": float(np.mean(sent)),
        "conditional_update_success_rate": float(np.sum(success) / np.sum(sent))
        if np.sum(sent) > 0
        else 1.0,
        "mode_rates": {
            mode: float(np.mean([row[f"{mode}_rate"] for row in rows]))
            for mode in ("silence", "absolute", "delta", "bundle")
        },
    }


def compare_rows(
    protocol: dict,
    proposed: list[dict],
    baseline: list[dict],
    *,
    seed: int,
) -> dict:
    result = paired_cluster_bootstrap(
        np.asarray([row["regret_db"] for row in proposed]),
        np.asarray([row["regret_db"] for row in baseline]),
        np.asarray([row["actual_bits"] for row in proposed]),
        np.asarray([row["actual_bits"] for row in baseline]),
        [row["cluster_id"] for row in baseline],
        repetitions=int(
            protocol["statistics"]["paired_cluster_bootstrap_repetitions"]
        ),
        seed=seed,
        cvar_alpha=float(protocol["statistics"]["cvar_alpha"]),
        one_sided_confidence=float(
            protocol["statistics"]["one_sided_confidence"]
        ),
    )
    baseline_bits = float(np.mean([row["actual_bits"] for row in baseline]))
    proposed_bits = float(np.mean([row["actual_bits"] for row in proposed]))
    result["actual_bit_reduction_pct"] = float(
        100.0 * (baseline_bits - proposed_bits) / baseline_bits
    )
    return result


def run_schedule(
    cache: dict[str, np.ndarray],
    stage2: dict,
    protocol: dict,
    query: np.ndarray,
    *,
    name: str,
    seed_offset: int,
) -> dict:
    n_channels = int(protocol["query_bundle"]["n_channels"])
    demands = tuple(int(value) for value in protocol["query_bundle"]["demand_order"])
    header = int(protocol["query_bundle"]["application_header_bits"])
    declared_payload = int(protocol["query_bundle"]["payload_bits"])
    if query_bundle_payload_width(n_channels, demands) != declared_payload:
        raise ValueError("declared bundle payload width is inconsistent")
    threshold = float(protocol["fixed_policy"]["regret_threshold_db"])
    max_age = float(protocol["fixed_policy"]["max_age_minutes"])
    channel_models = protocol["link"]["channel_models"]
    ebn0_values = protocol["link"]["ebn0_db"]
    repeats = int(protocol["link"]["repeats"])
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
                "bundle",
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
                        bundle_epoch = 0
                        for position in positions:
                            current_demand = int(query[position])
                            timestamp = str(cache["timestamps_local"][position])
                            costs_by_demand = {
                                demand: contiguous_block_costs(
                                    cache["channel_power_dbm"][position],
                                    demand,
                                )
                                for demand in demands
                            }
                            current_costs = costs_by_demand[current_demand]
                            current_state = states.get(current_demand)
                            target_by_demand = {
                                demand: int(np.argmin(costs_by_demand[demand]))
                                for demand in demands
                            }
                            mode: str
                            target = target_by_demand[current_demand]
                            if method == "query_absolute_index_always":
                                mode = "absolute"
                            elif method == "query_bundle_always":
                                mode = "bundle"
                            else:
                                decision = choose_event_update(
                                    current_costs,
                                    state=current_state,
                                    timestamp_local=timestamp,
                                    regret_threshold_db=threshold,
                                    max_age_minutes=max_age,
                                    demand_channels=current_demand,
                                    application_header_bits=header,
                                )
                                if not decision.transmits:
                                    mode = "silence"
                                elif method == "query_event_delta_fixed":
                                    mode = decision.mode
                                    target = decision.target_start
                                elif method == "query_event_bundle_fixed":
                                    mode = "bundle"
                                else:
                                    raise KeyError(method)
                            values = accumulators[method]
                            values[mode][position] += 1.0
                            if mode == "silence":
                                selected = guarded_state_start(
                                    current_state,
                                    timestamp,
                                    max_age_minutes=max_age,
                                    candidate_blocks=len(current_costs),
                                )
                                values["regret"][position] += block_regret(
                                    current_costs,
                                    selected,
                                )
                                continue
                            if mode == "bundle":
                                epoch = bundle_epoch % 256
                                encoded = encode_query_bundle(
                                    target_by_demand,
                                    n_channels=n_channels,
                                    demand_order=demands,
                                    node_id=0,
                                    epoch=epoch,
                                    application_header_bits=header,
                                )
                            else:
                                epoch = (
                                    0
                                    if current_state is None
                                    else (current_state.epoch + 1) % 256
                                )
                                encoded = encode_update(
                                    mode=mode,
                                    target_start=target,
                                    cached_start=None
                                    if current_state is None
                                    else current_state.selected_start,
                                    candidate_blocks=len(current_costs),
                                    node_id=0,
                                    demand_channels=current_demand,
                                    epoch=epoch,
                                    application_header_bits=header,
                                )
                            seed = (
                                96260723
                                + seed_offset
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
                                if mode == "bundle":
                                    decoded_bundle = decode_query_bundle(
                                        encoded,
                                        n_channels=n_channels,
                                        demand_order=demands,
                                        application_header_bits=header,
                                    )
                                    for demand, start in (
                                        decoded_bundle.target_starts.items()
                                    ):
                                        states[demand] = QueryState(
                                            start,
                                            timestamp,
                                            decoded_bundle.epoch,
                                        )
                                    bundle_epoch = (bundle_epoch + 1) % 256
                                    selected = decoded_bundle.target_starts[
                                        current_demand
                                    ]
                                else:
                                    decoded = decode_update(
                                        encoded,
                                        candidate_blocks=len(current_costs),
                                        cached_start=None
                                        if current_state is None
                                        else current_state.selected_start,
                                        application_header_bits=header,
                                    )
                                    states[current_demand] = QueryState(
                                        decoded.selected_start,
                                        timestamp,
                                        decoded.epoch,
                                    )
                                    selected = decoded.selected_start
                                values["success"][position] += 1.0
                            else:
                                selected = guarded_state_start(
                                    current_state,
                                    timestamp,
                                    max_age_minutes=max_age,
                                    candidate_blocks=len(current_costs),
                                )
                            values["regret"][position] += block_regret(
                                current_costs,
                                selected,
                            )
            print(f"{name}: {channel} {float(ebn0):g} dB complete", flush=True)
    rows_by_method = {}
    summary = {}
    alpha = float(protocol["statistics"]["cvar_alpha"])
    for method in METHODS:
        values = accumulators[method]
        rows = []
        for position in range(count):
            rows.append(
                {
                    "schedule": name,
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
                    **{
                        f"{mode}_rate": float(values[mode][position] / trajectories)
                        for mode in ("silence", "absolute", "delta", "bundle")
                    },
                }
            )
        rows_by_method[method] = rows
        summary[method] = summarize(rows, alpha)
    comparisons_vs_always = {}
    comparisons_vs_per_query = {}
    baseline = rows_by_method["query_absolute_index_always"]
    per_query = rows_by_method["query_event_delta_fixed"]
    for index, method in enumerate(METHODS[1:]):
        comparisons_vs_always[method] = compare_rows(
            protocol,
            rows_by_method[method],
            baseline,
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + seed_offset
            + index,
        )
    for index, method in enumerate(
        ("query_bundle_always", "query_event_bundle_fixed")
    ):
        comparisons_vs_per_query[method] = compare_rows(
            protocol,
            rows_by_method[method],
            per_query,
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + seed_offset
            + 100
            + index,
        )
    gate = protocol["descriptive_gate_per_schedule"]
    proposed = comparisons_vs_always["query_event_bundle_fixed"]
    proposed["passes_descriptive_gate"] = bool(
        proposed["actual_bit_reduction_pct"]
        >= float(gate["minimum_actual_bit_reduction_pct_vs_always"])
        and proposed["mean_regret_upper_bound"]
        <= float(gate["maximum_mean_regret_upper_bound_db_vs_always"])
        and proposed["cvar_upper_bound"]
        <= float(gate["maximum_cvar_upper_bound_db_vs_always"])
    )
    return {
        "demand_counts": {
            str(demand): int(np.count_nonzero(query == demand))
            for demand in demands
        },
        "summary": summary,
        "comparisons_vs_query_absolute_index_always": comparisons_vs_always,
        "comparisons_vs_per_query_event_delta": comparisons_vs_per_query,
        "diagnostics": {
            "trajectory_count_per_scene": trajectories,
            "unique_transmission_outcomes_cached": len(tx_cache),
        },
        "rows": [
            row for method in METHODS for row in rows_by_method[method]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_query_bundle_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/query_bundle_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite query-bundle output")
    protocol = load_json(args.protocol)
    cache, stage2, schedule_protocol = validate_inputs(args.protocol, protocol)
    schedules = {}
    for index, name in enumerate(
        schedule_protocol["resource_queries"]["schedules"].keys()
    ):
        query = build_query_schedule(cache, schedule_protocol, name)
        schedules[name] = run_schedule(
            cache,
            stage2,
            protocol,
            query,
            name=name,
            seed_offset=(index + 1) * 10_000,
        )
    result = {
        "version": "1.0",
        "status": "stage5_query_bundle_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_result_sha256": protocol["development_source"][
                "cache_result_sha256"
            ],
            "cache_sha256": protocol["development_source"]["cache_sha256"],
            "schedule_protocol_sha256": protocol["s5_2_boundary_evidence"][
                "schedule_protocol_sha256"
            ],
            "s5_2_result_sha256": protocol["s5_2_boundary_evidence"][
                "source_result_sha256"
            ],
            "stage2_config_sha256": protocol["link"]["stage2_config_sha256"],
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
    output = args.out_dir / "query_bundle_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "bundle_gate": {
                    name: row["comparisons_vs_query_absolute_index_always"][
                        "query_event_bundle_fixed"
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
