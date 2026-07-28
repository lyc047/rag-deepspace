#!/usr/bin/env python
"""Evaluate on-demand and piggybacked cumulative confirmation."""

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
from run_stage5_ack_state_recovery_development import (  # noqa: E402
    _states_match,
    compare_rows,
    summarize,
    validate_inputs,
)
from run_stage5_dynamic_query_development import build_query_schedule  # noqa: E402
from run_stage5_event_semantics_development import (  # noqa: E402
    guarded_state_start,
    load_json,
    ordered_site_indices,
)
from spectrum_semcom.digital_link import (  # noqa: E402
    PacketConfig,
    nominal_transmitted_bits,
    transmit_payload_analytic,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
    decode_cumulative_ack,
    encode_cumulative_ack,
)
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    QueryState,
    block_regret,
    choose_event_update,
    contiguous_block_costs,
)
from spectrum_semcom.stage5_query_bundle import encode_query_bundle  # noqa: E402
from spectrum_semcom.stage5_sparse_ack import (  # noqa: E402
    ack_request_payload_bits,
    decode_ack_request,
    deterministic_piggyback_opportunity,
    encode_ack_request,
    piggyback_epoch_bits,
)
from spectrum_semcom.stage5_spectrum_stress import apply_spectrum_regime  # noqa: E402
from spectrum_semcom.stage5_state_sync import should_accept_ack  # noqa: E402


def validate_cumulative_predecessor(protocol: dict) -> None:
    predecessor = protocol["cumulative_ack_predecessor"]
    for key, hash_key in (
        ("protocol", "protocol_sha256"),
        ("result", "result_sha256"),
    ):
        path = PROJECT_DIR / predecessor[key]
        if sha256_file(path) != predecessor[hash_key]:
            raise ValueError(f"frozen sparse-ACK predecessor mismatch: {path}")


def _control_link(stage2: dict, protocol: dict, ebn0: float):
    base = build_link_config(stage2, ebn0)
    control = protocol["control_link"]
    packet = PacketConfig(
        payload_bits=64,
        header_bits=int(control["packet_header_bits"]),
        crc_bits=int(control["crc_bits"]),
        alignment_bits=int(control["alignment_bits"]),
    )
    return replace(
        base,
        packet=packet,
        fec=control["fec"],
        modulation=control["modulation"],
        max_retransmissions=int(control["max_retransmissions"]),
        latency_budget_s=None,
    )


def _control_costs(stage2: dict, protocol: dict, ebn0: float) -> dict[str, int]:
    control = protocol["control_link"]
    link = _control_link(stage2, protocol, ebn0)
    ack_bits = nominal_transmitted_bits(cumulative_ack_payload_bits(), link)
    request_bits = nominal_transmitted_bits(ack_request_payload_bits(), link)
    base_payload = int(control["piggyback_base_payload_bits"])
    increment = piggyback_epoch_bits()
    piggyback_bits = (
        nominal_transmitted_bits(base_payload + increment, link)
        - nominal_transmitted_bits(base_payload, link)
    )
    return {
        "ack_response_bits": int(ack_bits),
        "ack_request_bits": int(request_bits),
        "piggyback_increment_bits": int(piggyback_bits),
    }


def run_fault_condition(
    cache: dict,
    stage2: dict,
    protocol: dict,
    query: np.ndarray,
    *,
    control_erasure_probability: float,
    duplicate_probability: float,
    condition_index: int,
) -> dict:
    demands = tuple(protocol["query_bundle"]["demand_order"])
    n_channels = int(protocol["query_bundle"]["n_channels"])
    header = int(protocol["query_bundle"]["application_header_bits"])
    threshold = float(protocol["fixed_policy"]["regret_threshold_db"])
    max_age = float(protocol["fixed_policy"]["max_age_minutes"])
    count = len(cache["scene_ids"])
    site_sequences = ordered_site_indices(cache)
    metrics = (
        "regret",
        "data_bits",
        "control_bits",
        "initial_ack_bits",
        "request_bits",
        "request_response_bits",
        "piggyback_bits",
        "sent",
        "forced",
        "divergence",
        "control_sent",
        "control_loss",
        "stale_rejected",
        "requests",
        "piggybacks",
    )
    accumulators = {
        method: {name: np.zeros(count) for name in metrics}
        for method in protocol["methods"]
    }
    channel_models = protocol["link"]["channel_models"]
    ebn0_values = protocol["link"]["ebn0_db"]
    repeats = int(protocol["link"]["repeats"])
    trajectories = len(channel_models) * len(ebn0_values) * repeats
    tx_cache = {}

    for channel_index, channel in enumerate(channel_models):
        for ebn0_index, ebn0 in enumerate(ebn0_values):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            costs = _control_costs(stage2, protocol, float(ebn0))
            for repetition in range(repeats):
                for method, policy in protocol["methods"].items():
                    for positions in site_sequences:
                        tx_states: dict[int, QueryState] = {}
                        rx_states: dict[int, QueryState] = {}
                        sent_history: dict[int, dict[int, QueryState]] = {}
                        confirmed_epoch = None
                        latest_rx_epoch = None
                        next_epoch = 0
                        uncertain = False
                        previous_ack_epoch = None

                        def process_ack(epoch: int) -> bool:
                            nonlocal confirmed_epoch, uncertain
                            if epoch not in sent_history:
                                return False
                            if not should_accept_ack(
                                epoch,
                                confirmed_epoch,
                                guard_stale_epochs=True,
                            ):
                                return False
                            confirmed_epoch = epoch
                            tx_states.clear()
                            tx_states.update(sent_history[epoch])
                            uncertain = False
                            return True

                        for position in positions:
                            timestamp = str(cache["timestamps_local"][position])
                            scene_id = str(cache["scene_ids"][position])
                            current_demand = int(query[position])
                            values = accumulators[method]
                            seed = (
                                99260724
                                + condition_index * 10_000_000
                                + int(position) * 100_000
                                + channel_index * 1000
                                + ebn0_index * 100
                                + repetition
                            )
                            faults = bool(policy["faults_enabled"])

                            opportunity = deterministic_piggyback_opportunity(
                                scene_id,
                                float(
                                    policy["piggyback_opportunity_probability"]
                                ),
                            )
                            if opportunity and latest_rx_epoch is not None:
                                values["control_bits"][position] += costs[
                                    "piggyback_increment_bits"
                                ]
                                values["piggyback_bits"][position] += costs[
                                    "piggyback_increment_bits"
                                ]
                                values["control_sent"][position] += 1.0
                                values["piggybacks"][position] += 1.0
                                lost = bool(
                                    faults
                                    and np.random.default_rng(seed + 71_000).random()
                                    < control_erasure_probability
                                )
                                if lost:
                                    values["control_loss"][position] += 1.0
                                else:
                                    ack = decode_cumulative_ack(
                                        encode_cumulative_ack(
                                            node_id=0,
                                            epoch=latest_rx_epoch,
                                        )
                                    )
                                    accepted = process_ack(ack.epoch)
                                    values["stale_rejected"][position] += float(
                                        not accepted
                                    )

                            if uncertain and policy["request_on_uncertainty"]:
                                request = decode_ack_request(encode_ack_request())
                                if not request:
                                    raise AssertionError("ACK request codec failed")
                                values["control_bits"][position] += costs[
                                    "ack_request_bits"
                                ]
                                values["request_bits"][position] += costs[
                                    "ack_request_bits"
                                ]
                                values["control_sent"][position] += 1.0
                                values["requests"][position] += 1.0
                                random = np.random.default_rng(seed + 72_000).random(2)
                                request_lost = bool(
                                    faults
                                    and random[0] < control_erasure_probability
                                )
                                if request_lost:
                                    values["control_loss"][position] += 1.0
                                elif latest_rx_epoch is not None:
                                    values["control_bits"][position] += costs[
                                        "ack_response_bits"
                                    ]
                                    values["request_response_bits"][position] += costs[
                                        "ack_response_bits"
                                    ]
                                    values["control_sent"][position] += 1.0
                                    response_lost = bool(
                                        faults
                                        and random[1] < control_erasure_probability
                                    )
                                    if response_lost:
                                        values["control_loss"][position] += 1.0
                                    else:
                                        ack = decode_cumulative_ack(
                                            encode_cumulative_ack(
                                                node_id=0,
                                                epoch=latest_rx_epoch,
                                            )
                                        )
                                        accepted = process_ack(ack.epoch)
                                        values["stale_rejected"][position] += float(
                                            not accepted
                                        )

                            costs_by_demand = {
                                demand: contiguous_block_costs(
                                    cache["channel_power_dbm"][position],
                                    demand,
                                )
                                for demand in demands
                            }
                            current_costs = costs_by_demand[current_demand]
                            decision = choose_event_update(
                                current_costs,
                                state=tx_states.get(current_demand),
                                timestamp_local=timestamp,
                                regret_threshold_db=threshold,
                                max_age_minutes=max_age,
                                demand_channels=current_demand,
                                application_header_bits=header,
                            )
                            forced = bool(
                                uncertain and policy["semantic_recovery"]
                            )
                            if not decision.transmits and not forced:
                                selected = guarded_state_start(
                                    rx_states.get(current_demand),
                                    timestamp,
                                    max_age_minutes=max_age,
                                    candidate_blocks=len(current_costs),
                                )
                                values["regret"][position] += block_regret(
                                    current_costs, selected
                                )
                                values["divergence"][position] += float(
                                    not _states_match(tx_states, rx_states, demands)
                                )
                                continue

                            targets = {
                                demand: int(np.argmin(costs_by_demand[demand]))
                                for demand in demands
                            }
                            epoch = next_epoch
                            next_epoch = (next_epoch + 1) % 256
                            encoded = encode_query_bundle(
                                targets,
                                n_channels=n_channels,
                                demand_order=demands,
                                node_id=0,
                                epoch=epoch,
                                application_header_bits=header,
                            )
                            key = (encoded.size, channel, float(ebn0), seed)
                            if key not in tx_cache:
                                tx_cache[key] = transmit_payload_analytic(
                                    int(encoded.size), link, seed
                                )
                            tx = tx_cache[key]
                            values["sent"][position] += 1.0
                            values["forced"][position] += float(forced)
                            values["data_bits"][position] += float(
                                tx.transmitted_bits
                            )
                            current_ack_epoch = None
                            if tx.frame_success:
                                received_states = {
                                    demand: QueryState(
                                        targets[demand], timestamp, epoch
                                    )
                                    for demand in demands
                                }
                                rx_states = dict(received_states)
                                sent_history[epoch] = received_states
                                latest_rx_epoch = epoch
                                current_ack_epoch = epoch

                            if current_ack_epoch is not None:
                                ack_cost = costs["ack_response_bits"]
                                values["control_bits"][position] += ack_cost
                                values["initial_ack_bits"][position] += ack_cost
                                values["control_sent"][position] += 1.0
                                ack = decode_cumulative_ack(
                                    encode_cumulative_ack(
                                        node_id=0,
                                        epoch=current_ack_epoch,
                                    )
                                )
                                random = np.random.default_rng(seed + 55_000).random(2)
                                lost = bool(
                                    faults
                                    and random[0] < control_erasure_probability
                                )
                                if lost:
                                    values["control_loss"][position] += 1.0
                                    uncertain = True
                                else:
                                    process_ack(ack.epoch)
                                if (
                                    faults
                                    and previous_ack_epoch is not None
                                    and random[1] < duplicate_probability
                                ):
                                    values["control_bits"][position] += ack_cost
                                    values["initial_ack_bits"][position] += ack_cost
                                    values["control_sent"][position] += 1.0
                                    accepted = process_ack(previous_ack_epoch)
                                    values["stale_rejected"][position] += float(
                                        not accepted
                                    )
                                previous_ack_epoch = current_ack_epoch
                            else:
                                uncertain = True

                            selected = guarded_state_start(
                                rx_states.get(current_demand),
                                timestamp,
                                max_age_minutes=max_age,
                                candidate_blocks=len(current_costs),
                            )
                            values["regret"][position] += block_regret(
                                current_costs, selected
                            )
                            values["divergence"][position] += float(
                                not _states_match(tx_states, rx_states, demands)
                            )
            print(
                f"control_loss={control_erasure_probability:g} "
                f"duplicate={duplicate_probability:g}: "
                f"{channel} {float(ebn0):g} dB complete",
                flush=True,
            )

    rows_by_method = {}
    summaries = {}
    for method, values in accumulators.items():
        rows = []
        for position in range(count):
            data_bits = float(values["data_bits"][position] / trajectories)
            control_bits = float(values["control_bits"][position] / trajectories)
            rows.append(
                {
                    "method": method,
                    "scene_id": str(cache["scene_ids"][position]),
                    "site": str(cache["site_ids"][position]),
                    "timestamp_local": str(cache["timestamps_local"][position]),
                    "cluster_id": str(cache["cluster_ids"][position]),
                    "demand_channels": int(query[position]),
                    "regret_db": float(values["regret"][position] / trajectories),
                    "actual_bits": data_bits + control_bits,
                    "data_bits": data_bits,
                    "ack_control_bits": control_bits,
                    "initial_ack_bits": float(
                        values["initial_ack_bits"][position] / trajectories
                    ),
                    "request_bits": float(
                        values["request_bits"][position] / trajectories
                    ),
                    "request_response_bits": float(
                        values["request_response_bits"][position] / trajectories
                    ),
                    "piggyback_increment_bits": float(
                        values["piggyback_bits"][position] / trajectories
                    ),
                    "transmission_rate": float(values["sent"][position] / trajectories),
                    "forced_recovery_rate": float(
                        values["forced"][position] / trajectories
                    ),
                    "state_divergence_rate": float(
                        values["divergence"][position] / trajectories
                    ),
                    "ack_loss_rate": float(
                        values["control_loss"][position]
                        / max(values["control_sent"][position], 1.0)
                    ),
                    "stale_ack_rejection_rate": float(
                        values["stale_rejected"][position] / trajectories
                    ),
                    "request_rate": float(
                        values["requests"][position] / trajectories
                    ),
                    "piggyback_rate": float(
                        values["piggybacks"][position] / trajectories
                    ),
                }
            )
        rows_by_method[method] = rows
        summary = summarize(rows, float(protocol["statistics"]["cvar_alpha"]))
        for key in (
            "data_bits",
            "ack_control_bits",
            "initial_ack_bits",
            "request_bits",
            "request_response_bits",
            "piggyback_increment_bits",
            "request_rate",
            "piggyback_rate",
        ):
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
        summaries[method] = summary

    comparisons = {}
    ideal = rows_by_method["ideal_ack"]
    immediate = rows_by_method["immediate_data_recovery"]
    for index, method in enumerate(
        method for method in protocol["methods"] if method != "ideal_ack"
    ):
        comparisons[f"{method}_vs_ideal"] = compare_rows(
            protocol,
            rows_by_method[method],
            ideal,
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + condition_index * 100
            + index,
        )
    gate = protocol["sparse_ack_gate_vs_immediate"]
    candidates = (
        "request_ack_then_data",
        "piggyback_ack_p25",
        "piggyback_ack_p50",
        "piggyback_ack_p100",
    )
    for index, method in enumerate(candidates):
        comparison = compare_rows(
            protocol,
            rows_by_method[method],
            immediate,
            seed=int(protocol["statistics"]["bootstrap_seed"])
            + condition_index * 100
            + 30
            + index,
        )
        comparison["passes_sparse_ack_gate"] = bool(
            comparison["actual_bit_reduction_pct"]
            >= float(gate["minimum_total_bit_reduction_pct"])
            and comparison["mean_regret_upper_bound"]
            <= float(gate["maximum_mean_regret_upper_bound_db"])
            and comparison["cvar_upper_bound"]
            <= float(gate["maximum_cvar_upper_bound_db"])
        )
        comparisons[f"{method}_vs_immediate_data_recovery"] = comparison
    return {
        "control_costs": _control_costs(
            stage2, protocol, float(ebn0_values[0])
        ),
        "summary": summaries,
        "comparisons": comparisons,
        "diagnostics": {
            "trajectory_count_per_scene": trajectories,
            "unique_transmission_outcomes_cached": len(tx_cache),
        },
        "rows": [
            row for method in protocol["methods"] for row in rows_by_method[method]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_sparse_ack_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/sparse_ack_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite sparse-ACK output")
    protocol = load_json(args.protocol)
    validate_cumulative_predecessor(protocol)
    cache, stage2, schedule_protocol = validate_inputs(protocol)
    transformed = dict(cache)
    transformed["channel_power_dbm"] = apply_spectrum_regime(
        cache["channel_power_dbm"],
        cache["site_ids"],
        regime=protocol["stress_predecessor"]["regime"],
        parameters=protocol["stress_predecessor"]["parameters"],
    )
    query = build_query_schedule(
        transformed, schedule_protocol, protocol["schedule"]["name"]
    )
    conditions = {}
    condition_index = 0
    for control_loss in protocol["fault_model"]["control_erasure_probability"]:
        for duplicate in protocol["fault_model"]["delayed_duplicate_probability"]:
            key = f"control_loss_{control_loss:g}_duplicate_{duplicate:g}"
            conditions[key] = run_fault_condition(
                transformed,
                stage2,
                protocol,
                query,
                control_erasure_probability=float(control_loss),
                duplicate_probability=float(duplicate),
                condition_index=condition_index,
            )
            condition_index += 1
    result = {
        "version": "1.0",
        "status": "stage5_sparse_ack_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "conditions": conditions,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "control_faults_are_real_measurements": False,
            "reverse_business_traffic_is_measured": False,
            "incremental_control_bits_counted": True,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "sparse_ack_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "conditions": len(conditions),
                "incremental_control_bits_counted": True,
                "final_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
