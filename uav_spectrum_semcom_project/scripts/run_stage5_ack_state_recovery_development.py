#!/usr/bin/env python
"""Evaluate ACK-state divergence and uncertainty-driven bundle recovery."""

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
from run_stage5_query_bundle_development import (  # noqa: E402
    compare_rows,
    load_npz,
)
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    QueryState,
    block_regret,
    choose_event_update,
    contiguous_block_costs,
    state_age_minutes,
)
from spectrum_semcom.stage5_query_bundle import encode_query_bundle  # noqa: E402
from spectrum_semcom.stage5_spectrum_stress import apply_spectrum_regime  # noqa: E402
from spectrum_semcom.stage5_state_sync import (  # noqa: E402
    belief_worst_case_force_refresh,
    hysteresis_force_refresh,
    must_force_refresh,
    should_accept_ack,
)


def validate_inputs(protocol: dict) -> tuple[dict, dict, dict]:
    declared = (
        (
            PROJECT_DIR / protocol["development_source"]["cache_result"],
            protocol["development_source"]["cache_result_sha256"],
        ),
        (
            PROJECT_DIR / protocol["schedule"]["protocol"],
            protocol["schedule"]["protocol_sha256"],
        ),
        (
            PROJECT_DIR / protocol["stress_predecessor"]["protocol"],
            protocol["stress_predecessor"]["protocol_sha256"],
        ),
        (
            PROJECT_DIR / protocol["stress_predecessor"]["result"],
            protocol["stress_predecessor"]["result_sha256"],
        ),
        (
            PROJECT_DIR / protocol["link"]["stage2_config"],
            protocol["link"]["stage2_config_sha256"],
        ),
    )
    for path, expected in declared:
        if sha256_file(path) != expected:
            raise ValueError(f"frozen ACK-state input hash mismatch: {path}")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["fixed_policy_retuned"]
        or governance["ack_faults_are_measured_claims"]
    ):
        raise ValueError("ACK-state governance is invalid")
    cache_result = load_json(declared[0][0])
    split = protocol["development_source"]["split"]
    split_row = cache_result["splits"][split]
    if split_row["cache_sha256"] != protocol["development_source"]["cache_sha256"]:
        raise ValueError("ACK-state cache declaration changed")
    cache_path = PROJECT_DIR / split_row["cache"]
    if sha256_file(cache_path) != split_row["cache_sha256"]:
        raise ValueError("ACK-state cache hash mismatch")
    return (
        load_npz(cache_path),
        load_json(declared[4][0]),
        load_json(declared[1][0]),
    )


def _states_match(
    tx_states: dict[int, QueryState],
    rx_states: dict[int, QueryState],
    demands: tuple[int, ...],
) -> bool:
    for demand in demands:
        tx_state = tx_states.get(demand)
        rx_state = rx_states.get(demand)
        if tx_state is None and rx_state is None:
            continue
        if (
            tx_state is None
            or rx_state is None
            or tx_state.selected_start != rx_state.selected_start
            or tx_state.epoch != rx_state.epoch
        ):
            return False
    return True


def _accept_ack(
    *,
    ack: dict,
    tx_states: dict[int, QueryState],
    confirmed_epoch: int | None,
    guard_stale_epochs: bool,
) -> tuple[int | None, bool]:
    accepted = should_accept_ack(
        ack["epoch"],
        confirmed_epoch,
        guard_stale_epochs=guard_stale_epochs,
    )
    if accepted:
        tx_states.clear()
        tx_states.update(ack["states"])
        return int(ack["epoch"]), True
    return confirmed_epoch, False


def summarize(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows])
    return {
        "scene_count": len(rows),
        "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_transmission_rate": float(
            np.mean([row["transmission_rate"] for row in rows])
        ),
        "mean_forced_recovery_rate": float(
            np.mean([row["forced_recovery_rate"] for row in rows])
        ),
        "mean_state_divergence_rate": float(
            np.mean([row["state_divergence_rate"] for row in rows])
        ),
        "mean_ack_loss_rate": float(
            np.mean([row["ack_loss_rate"] for row in rows])
        ),
        "mean_stale_ack_rejection_rate": float(
            np.mean([row["stale_ack_rejection_rate"] for row in rows])
        ),
    }


def run_fault_condition(
    cache: dict,
    stage2: dict,
    protocol: dict,
    query: np.ndarray,
    *,
    ack_loss_probability: float,
    duplicate_probability: float,
    condition_index: int,
) -> dict:
    n_channels = int(protocol["query_bundle"]["n_channels"])
    demands = tuple(protocol["query_bundle"]["demand_order"])
    header = int(protocol["query_bundle"]["application_header_bits"])
    threshold = float(protocol["fixed_policy"]["regret_threshold_db"])
    max_age = float(protocol["fixed_policy"]["max_age_minutes"])
    count = len(cache["scene_ids"])
    site_sequences = ordered_site_indices(cache)
    metric_names = (
        "regret",
        "bits",
        "sent",
        "forced",
        "divergence",
        "ack_loss",
        "stale_rejected",
    )
    accumulators = {
        method: {name: np.zeros(count) for name in metric_names}
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
            for repetition in range(repeats):
                for method_index, (method, policy) in enumerate(
                    protocol["methods"].items()
                ):
                    for positions in site_sequences:
                        tx_states: dict[int, QueryState] = {}
                        rx_states: dict[int, QueryState] = {}
                        confirmed_epoch = None
                        next_epoch = 0
                        uncertain = False
                        unconfirmed_count = 0
                        previous_delivered_ack = None
                        receiver_hypotheses: list[dict[int, QueryState]] = [{}]
                        for position in positions:
                            timestamp = str(cache["timestamps_local"][position])
                            current_demand = int(query[position])
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
                            if policy.get("semantic_belief_recovery", False):
                                plausible_starts = [
                                    guarded_state_start(
                                        hypothesis.get(current_demand),
                                        timestamp,
                                        max_age_minutes=max_age,
                                        candidate_blocks=len(current_costs),
                                    )
                                    for hypothesis in receiver_hypotheses
                                ]
                                forced = belief_worst_case_force_refresh(
                                    state_uncertain=uncertain,
                                    plausible_selected_starts=plausible_starts,
                                    costs_dbm=current_costs,
                                    regret_threshold_db=threshold,
                                )
                            elif "force_after_unconfirmed_count" in policy:
                                tx_state = tx_states.get(current_demand)
                                age = (
                                    None
                                    if tx_state is None
                                    else state_age_minutes(tx_state, timestamp)
                                )
                                forced = hysteresis_force_refresh(
                                    unconfirmed_transmissions=unconfirmed_count,
                                    force_after_count=policy[
                                        "force_after_unconfirmed_count"
                                    ],
                                    state_age_minutes=age,
                                    max_age_minutes=max_age,
                                    age_guard_fraction=policy[
                                        "age_guard_fraction"
                                    ],
                                )
                            else:
                                forced = must_force_refresh(
                                    state_uncertain=uncertain,
                                    force_while_uncertain=bool(
                                        policy["force_while_uncertain"]
                                    ),
                                )
                            transmits = bool(decision.transmits or forced)
                            values = accumulators[method]
                            if not transmits:
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
                            attempted_states = {
                                demand: QueryState(
                                    targets[demand], timestamp, epoch
                                )
                                for demand in demands
                            }
                            encoded = encode_query_bundle(
                                targets,
                                n_channels=n_channels,
                                demand_order=demands,
                                node_id=0,
                                epoch=epoch,
                                application_header_bits=header,
                            )
                            seed = (
                                97260724
                                + condition_index * 10_000_000
                                + int(position) * 100_000
                                + channel_index * 1000
                                + ebn0_index * 100
                                + repetition
                            )
                            key = (encoded.size, channel, float(ebn0), seed)
                            if key not in tx_cache:
                                tx_cache[key] = transmit_payload_analytic(
                                    int(encoded.size), link, seed
                                )
                            tx = tx_cache[key]
                            values["sent"][position] += 1.0
                            values["forced"][position] += float(forced)
                            values["bits"][position] += float(tx.transmitted_bits)
                            current_ack = None
                            if tx.frame_success:
                                received_states = attempted_states
                                rx_states = dict(received_states)
                                current_ack = {
                                    "epoch": epoch,
                                    "states": received_states,
                                }
                            faults = bool(policy["faults_enabled"])
                            random = np.random.default_rng(seed + 55_000).random(2)
                            ack_lost = bool(
                                current_ack is not None
                                and faults
                                and random[0] < ack_loss_probability
                            )
                            if current_ack is not None and not ack_lost:
                                confirmed_epoch, accepted = _accept_ack(
                                    ack=current_ack,
                                    tx_states=tx_states,
                                    confirmed_epoch=confirmed_epoch,
                                    guard_stale_epochs=bool(
                                        policy["guard_stale_epochs"]
                                    ),
                                )
                                if accepted:
                                    receiver_hypotheses = [dict(tx_states)]
                                uncertain = False
                                unconfirmed_count = 0
                            else:
                                uncertain = True
                                unconfirmed_count += 1
                                if policy.get("semantic_belief_recovery", False):
                                    receiver_hypotheses.append(
                                        dict(attempted_states)
                                    )
                                    unique_hypotheses = {}
                                    for hypothesis in receiver_hypotheses:
                                        signature = tuple(
                                            (
                                                demand,
                                                state.selected_start,
                                                state.success_timestamp,
                                                state.epoch,
                                            )
                                            for demand, state in sorted(
                                                hypothesis.items()
                                            )
                                        )
                                        unique_hypotheses[signature] = hypothesis
                                    receiver_hypotheses = list(
                                        unique_hypotheses.values()
                                    )
                                values["ack_loss"][position] += float(
                                    current_ack is not None and ack_lost
                                )
                            inject_duplicate = bool(
                                faults
                                and previous_delivered_ack is not None
                                and random[1] < duplicate_probability
                            )
                            if inject_duplicate:
                                confirmed_epoch, accepted = _accept_ack(
                                    ack=previous_delivered_ack,
                                    tx_states=tx_states,
                                    confirmed_epoch=confirmed_epoch,
                                    guard_stale_epochs=bool(
                                        policy["guard_stale_epochs"]
                                    ),
                                )
                                values["stale_rejected"][position] += float(
                                    not accepted
                                )
                                if accepted:
                                    uncertain = False
                                    receiver_hypotheses = [dict(tx_states)]
                            if current_ack is not None and not ack_lost:
                                previous_delivered_ack = current_ack
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
                f"ack_loss={ack_loss_probability:g} duplicate={duplicate_probability:g}: "
                f"{channel} {float(ebn0):g} dB complete",
                flush=True,
            )
    rows_by_method = {}
    summaries = {}
    for method, values in accumulators.items():
        rows = []
        for position in range(count):
            rows.append(
                {
                    "method": method,
                    "scene_id": str(cache["scene_ids"][position]),
                    "site": str(cache["site_ids"][position]),
                    "timestamp_local": str(cache["timestamps_local"][position]),
                    "cluster_id": str(cache["cluster_ids"][position]),
                    "demand_channels": int(query[position]),
                    "regret_db": float(values["regret"][position] / trajectories),
                    "actual_bits": float(values["bits"][position] / trajectories),
                    "transmission_rate": float(values["sent"][position] / trajectories),
                    "forced_recovery_rate": float(
                        values["forced"][position] / trajectories
                    ),
                    "state_divergence_rate": float(
                        values["divergence"][position] / trajectories
                    ),
                    "ack_loss_rate": float(
                        values["ack_loss"][position] / trajectories
                    ),
                    "stale_ack_rejection_rate": float(
                        values["stale_rejected"][position] / trajectories
                    ),
                }
            )
        rows_by_method[method] = rows
        summaries[method] = summarize(
            rows, float(protocol["statistics"]["cvar_alpha"])
        )
    comparisons = {}
    ideal = rows_by_method["ideal_ack"]
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
    if "uncertainty_recovery" in rows_by_method:
        naive = rows_by_method["naive_ack"]
        for index, method in enumerate(("epoch_guard", "uncertainty_recovery")):
            comparisons[f"{method}_vs_naive"] = compare_rows(
                protocol,
                rows_by_method[method],
                naive,
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + condition_index * 100
                + 20
                + index,
            )
        recovery = comparisons["uncertainty_recovery_vs_naive"]
        gate = protocol["recovery_gate_vs_naive"]
        recovery["actual_bit_increase_pct"] = -float(
            recovery["actual_bit_reduction_pct"]
        )
        recovery["passes_recovery_gate"] = bool(
            recovery["mean_regret_upper_bound"]
            <= float(gate["maximum_mean_regret_upper_bound_db"])
            and recovery["cvar_upper_bound"]
            <= float(gate["maximum_cvar_upper_bound_db"])
            and recovery["actual_bit_increase_pct"]
            <= float(gate["maximum_actual_bit_increase_pct"])
        )
        if "belief_risk_recovery" in rows_by_method:
            belief_vs_naive = compare_rows(
                protocol,
                rows_by_method["belief_risk_recovery"],
                naive,
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + condition_index * 100
                + 40,
            )
            belief_vs_naive["actual_bit_increase_pct"] = -float(
                belief_vs_naive["actual_bit_reduction_pct"]
            )
            belief_vs_naive["passes_recovery_gate"] = bool(
                belief_vs_naive["mean_regret_upper_bound"]
                <= float(gate["maximum_mean_regret_upper_bound_db"])
                and belief_vs_naive["cvar_upper_bound"]
                <= float(gate["maximum_cvar_upper_bound_db"])
                and belief_vs_naive["actual_bit_increase_pct"]
                <= float(gate["maximum_actual_bit_increase_pct"])
            )
            comparisons["belief_risk_recovery_vs_naive"] = belief_vs_naive

            belief_gate = protocol["belief_gate_vs_immediate_recovery"]
            belief_vs_immediate = compare_rows(
                protocol,
                rows_by_method["belief_risk_recovery"],
                rows_by_method["uncertainty_recovery"],
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + condition_index * 100
                + 41,
            )
            belief_vs_immediate["passes_belief_gate"] = bool(
                belief_vs_immediate["actual_bit_reduction_pct"]
                >= float(belief_gate["minimum_actual_bit_reduction_pct"])
                and belief_vs_immediate["mean_regret_upper_bound"]
                <= float(belief_gate["maximum_mean_regret_upper_bound_db"])
                and belief_vs_immediate["cvar_upper_bound"]
                <= float(belief_gate["maximum_cvar_upper_bound_db"])
            )
            comparisons[
                "belief_risk_recovery_vs_uncertainty_recovery"
            ] = belief_vs_immediate
    else:
        immediate = rows_by_method["immediate_recovery"]
        gate = protocol["hysteresis_gate_vs_immediate"]
        for index, method in enumerate(("hysteresis2", "hysteresis2_age80")):
            comparison = compare_rows(
                protocol,
                rows_by_method[method],
                immediate,
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + condition_index * 100
                + 30
                + index,
            )
            comparison["passes_hysteresis_gate"] = bool(
                comparison["actual_bit_reduction_pct"]
                >= float(gate["minimum_actual_bit_reduction_pct"])
                and comparison["mean_regret_upper_bound"]
                <= float(gate["maximum_mean_regret_upper_bound_db"])
                and comparison["cvar_upper_bound"]
                <= float(gate["maximum_cvar_upper_bound_db"])
            )
            comparisons[f"{method}_vs_immediate_recovery"] = comparison
    return {
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
        default=PROJECT_DIR
        / "configs/stage5_ack_state_recovery_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/ack_state_recovery_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite ACK-state output")
    protocol = load_json(args.protocol)
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
    for ack_loss in protocol["ack_fault_grid"]["ack_loss_probability"]:
        for duplicate in protocol["ack_fault_grid"][
            "delayed_duplicate_probability"
        ]:
            key = f"ack_loss_{ack_loss:g}_duplicate_{duplicate:g}"
            conditions[key] = run_fault_condition(
                transformed,
                stage2,
                protocol,
                query,
                ack_loss_probability=float(ack_loss),
                duplicate_probability=float(duplicate),
                condition_index=condition_index,
            )
            condition_index += 1
    result = {
        "version": "1.0",
        "status": "stage5_ack_state_recovery_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "conditions": conditions,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "ack_faults_are_real_measurements": False,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "ack_state_recovery_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "conditions": len(conditions),
                "final_loaded": False,
                "fixed_policy_retuned": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
