#!/usr/bin/env python
"""Run Stage-5 calibration and frozen development validation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
    paired_cluster_bootstrap,
)
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


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    regret_threshold_db: float
    max_age_minutes: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def candidate_name(threshold: float, age: float) -> str:
    threshold_token = f"{threshold:g}".replace(".", "p")
    age_token = f"{age:g}".replace(".", "p")
    return f"event_delta_tau{threshold_token}_age{age_token}"


def validate_inputs(protocol_path: Path, protocol: dict) -> tuple[dict, dict]:
    source_path = PROJECT_DIR / protocol["development_source"]["cache_result"]
    stage2_path = PROJECT_DIR / protocol["link"]["stage2_config"]
    for path in (source_path, stage2_path):
        if "c1_v4_final" in path.as_posix().lower():
            raise ValueError("Stage-5 development must not load a C1-v4 Final path")
    if sha256_file(source_path) != protocol["development_source"]["cache_result_sha256"]:
        raise ValueError("development cache registry hash mismatch")
    if sha256_file(stage2_path) != protocol["link"]["stage2_config_sha256"]:
        raise ValueError("Stage-2 link hash mismatch")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["may_modify_stage4_final_algorithm_or_claim"]
    ):
        raise ValueError("Stage-5 governance must fail closed against Stage-4 Final")
    cache_result = load_json(source_path)
    if cache_result.get("governance", {}).get("development_only") is not True:
        raise ValueError("source cache is not development-only")
    return cache_result, load_json(stage2_path)


def split_cache(protocol: dict, cache_result: dict, split: str) -> dict[str, np.ndarray]:
    meta = cache_result["splits"][split]
    path = PROJECT_DIR / meta["cache"]
    if sha256_file(path) != meta["cache_sha256"]:
        raise ValueError(f"{split} cache hash mismatch")
    cache = load_npz(path)
    required = {
        "scene_ids",
        "site_ids",
        "timestamps_local",
        "cluster_ids",
        "channel_power_dbm",
    }
    if not required.issubset(cache):
        raise ValueError(f"{split} cache is missing required arrays")
    return cache


def ordered_site_indices(cache: dict[str, np.ndarray]) -> list[np.ndarray]:
    output = []
    sites = cache["site_ids"].astype(str)
    timestamps = cache["timestamps_local"].astype(str)
    for site in sorted(set(sites)):
        positions = np.flatnonzero(sites == site)
        output.append(positions[np.argsort(timestamps[positions])])
    return output


def guarded_state_start(
    state: QueryState | None,
    timestamp_local: str,
    *,
    max_age_minutes: float,
    candidate_blocks: int,
) -> int:
    if (
        state is None
        or state.selected_start >= candidate_blocks
        or state_age_minutes(state, timestamp_local) > max_age_minutes
    ):
        return 0
    return int(state.selected_start)


def always_absolute_decision(
    costs: np.ndarray,
    *,
    header_bits: int,
    reason: str,
) -> EventDecision:
    payload = absolute_index_width(len(costs))
    return EventDecision(
        mode="absolute",
        target_start=int(np.argmin(costs)),
        reuse_regret_db=0.0,
        age_minutes=None,
        application_bits=int(header_bits) + payload,
        payload_bits=payload,
        reason=reason,
    )


def periodic_decision(
    costs: np.ndarray,
    *,
    state: QueryState | None,
    timestamp_local: str,
    period_minutes: float,
    header_bits: int,
) -> EventDecision:
    if state is None or state_age_minutes(state, timestamp_local) >= period_minutes:
        return always_absolute_decision(
            costs,
            header_bits=header_bits,
            reason="periodic_refresh_due",
        )
    target = int(np.argmin(costs))
    reuse = block_regret(costs, state.selected_start)
    return EventDecision(
        mode="silence",
        target_start=target,
        reuse_regret_db=reuse,
        age_minutes=state_age_minutes(state, timestamp_local),
        application_bits=0,
        payload_bits=0,
        reason="periodic_wait",
    )


def policy_decision(
    policy: Policy,
    costs: np.ndarray,
    *,
    state: QueryState | None,
    timestamp_local: str,
    demand_channels: int,
    header_bits: int,
) -> EventDecision:
    if policy.kind == "absolute_always":
        return always_absolute_decision(
            costs,
            header_bits=header_bits,
            reason="absolute_index_every_scene",
        )
    if policy.kind == "periodic_absolute":
        return periodic_decision(
            costs,
            state=state,
            timestamp_local=timestamp_local,
            period_minutes=policy.max_age_minutes,
            header_bits=header_bits,
        )
    decision = choose_event_update(
        costs,
        state=state,
        timestamp_local=timestamp_local,
        regret_threshold_db=policy.regret_threshold_db,
        max_age_minutes=policy.max_age_minutes,
        demand_channels=demand_channels,
        application_header_bits=header_bits,
    )
    if policy.kind == "event_delta" or not decision.transmits:
        return decision
    if policy.kind != "event_absolute":
        raise KeyError(policy.kind)
    return always_absolute_decision(
        costs,
        header_bits=header_bits,
        reason=f"{decision.reason}_forced_absolute",
    )


def summarize_rows(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows], dtype=np.float64)
    transmissions = np.asarray([row["transmission_rate"] for row in rows])
    successes = np.asarray([row["successful_update_rate"] for row in rows])
    return {
        "scene_count": len(rows),
        "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_transmission_rate": float(np.mean(transmissions)),
        "mean_successful_update_rate": float(np.mean(successes)),
        "conditional_update_success_rate": float(
            np.sum(successes) / np.sum(transmissions)
        )
        if np.sum(transmissions) > 0
        else 1.0,
        "mode_rates": {
            mode: float(np.mean([row[f"{mode}_rate"] for row in rows]))
            for mode in ("silence", "absolute", "delta")
        },
    }


def run_policies(
    protocol: dict,
    stage2: dict,
    cache: dict[str, np.ndarray],
    policies: list[Policy],
    *,
    split_seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    task = protocol["resource_task"]
    demand = int(task["demand_channels"])
    n_channels = int(task["n_channels"])
    candidate_blocks = int(task["candidate_blocks"])
    header_bits = int(protocol["codec"]["compact_application_header_bits"])
    if candidate_blocks != n_channels - demand + 1:
        raise ValueError("candidate block declaration is inconsistent")
    channel_models = protocol["link"]["channel_models"]
    ebn0_values = protocol["link"]["ebn0_db"]
    repeats = int(protocol["link"]["repeats"])
    count = len(cache["scene_ids"])
    site_sequences = ordered_site_indices(cache)
    accumulators = {
        policy.name: {
            "regret": np.zeros(count),
            "bits": np.zeros(count),
            "sent": np.zeros(count),
            "success": np.zeros(count),
            "silence": np.zeros(count),
            "absolute": np.zeros(count),
            "delta": np.zeros(count),
        }
        for policy in policies
    }
    tx_cache: dict[tuple, object] = {}
    trajectory_count = len(channel_models) * len(ebn0_values) * repeats
    for channel_index, channel in enumerate(channel_models):
        for ebn0_index, ebn0 in enumerate(ebn0_values):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(repeats):
                for policy_index, policy in enumerate(policies):
                    for positions in site_sequences:
                        state: QueryState | None = None
                        for position in positions:
                            timestamp = str(cache["timestamps_local"][position])
                            costs = contiguous_block_costs(
                                cache["channel_power_dbm"][position],
                                demand,
                            )
                            decision = policy_decision(
                                policy,
                                costs,
                                state=state,
                                timestamp_local=timestamp,
                                demand_channels=demand,
                                header_bits=header_bits,
                            )
                            values = accumulators[policy.name]
                            values[decision.mode][position] += 1.0
                            if not decision.transmits:
                                selected = guarded_state_start(
                                    state,
                                    timestamp,
                                    max_age_minutes=policy.max_age_minutes,
                                    candidate_blocks=candidate_blocks,
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
                                cached_start=None if state is None else state.selected_start,
                                candidate_blocks=candidate_blocks,
                                node_id=0,
                                demand_channels=demand,
                                epoch=epoch,
                                application_header_bits=header_bits,
                            )
                            if encoded.size != decision.application_bits:
                                raise RuntimeError("policy length differs from real codec")
                            seed = (
                                split_seed
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
                                    candidate_blocks=candidate_blocks,
                                    cached_start=None
                                    if state is None
                                    else state.selected_start,
                                    application_header_bits=header_bits,
                                )
                                if decoded.selected_start != decision.target_start:
                                    raise RuntimeError("successful codec changed task decision")
                                state = QueryState(
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
                                    max_age_minutes=policy.max_age_minutes,
                                    candidate_blocks=candidate_blocks,
                                )
                            values["regret"][position] += block_regret(
                                costs,
                                selected,
                            )
            print(
                f"link condition complete: {channel} {float(ebn0):g} dB",
                flush=True,
            )
    rows_by_policy: dict[str, list[dict]] = {}
    summary = {}
    alpha = float(protocol["statistics"]["cvar_alpha"])
    for policy in policies:
        values = accumulators[policy.name]
        rows = []
        for position in range(count):
            rows.append(
                {
                    "policy": policy.name,
                    "scene_id": str(cache["scene_ids"][position]),
                    "site": str(cache["site_ids"][position]),
                    "timestamp_local": str(cache["timestamps_local"][position]),
                    "cluster_id": str(cache["cluster_ids"][position]),
                    "regret_db": float(values["regret"][position] / trajectory_count),
                    "actual_bits": float(values["bits"][position] / trajectory_count),
                    "transmission_rate": float(values["sent"][position] / trajectory_count),
                    "successful_update_rate": float(
                        values["success"][position] / trajectory_count
                    ),
                    "silence_rate": float(
                        values["silence"][position] / trajectory_count
                    ),
                    "absolute_rate": float(
                        values["absolute"][position] / trajectory_count
                    ),
                    "delta_rate": float(
                        values["delta"][position] / trajectory_count
                    ),
                }
            )
        rows_by_policy[policy.name] = rows
        summary[policy.name] = summarize_rows(rows, alpha)
    diagnostics = {
        "trajectory_count_per_scene": trajectory_count,
        "unique_transmission_outcomes_cached": len(tx_cache),
    }
    return rows_by_policy, {"summary": summary, "diagnostics": diagnostics}


def compare(
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


def calibration_candidate_is_eligible(protocol: dict, comparison: dict) -> bool:
    rule = protocol["calibration_selection_rule"]
    return bool(
        comparison["actual_bit_reduction_pct"]
        >= float(rule["minimum_actual_bit_reduction_pct"])
        and comparison["mean_regret_upper_bound"]
        <= float(rule["maximum_mean_regret_upper_bound_db"])
        and comparison["cvar_upper_bound"]
        <= float(rule["maximum_cvar_upper_bound_db"])
    )


def select_candidate(
    protocol: dict,
    candidates: list[Policy],
    comparisons: dict,
    summaries: dict,
) -> Policy | None:
    eligible = [
        policy
        for policy in candidates
        if calibration_candidate_is_eligible(protocol, comparisons[policy.name])
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda policy: (
            summaries[policy.name]["mean_actual_bits"],
            summaries[policy.name]["cvar_0_9_regret_db"],
            summaries[policy.name]["mean_regret_db"],
            summaries[policy.name]["mean_transmission_rate"],
            policy.regret_threshold_db,
            policy.max_age_minutes,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_event_semantics_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/event_semantics_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "event_semantics_result.json"
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite Stage-5 development output")
    protocol = load_json(args.protocol)
    cache_result, stage2 = validate_inputs(args.protocol, protocol)
    source = protocol["development_source"]
    calibration = split_cache(protocol, cache_result, source["calibration_split"])
    baseline = Policy("absolute_index_always", "absolute_always", 0.0, 60.0)
    candidates = [
        Policy(
            candidate_name(float(threshold), float(age)),
            "event_delta",
            float(threshold),
            float(age),
        )
        for threshold in protocol["candidate_grid"]["regret_threshold_db"]
        for age in protocol["candidate_grid"]["max_age_minutes"]
    ]
    calibration_rows, calibration_output = run_policies(
        protocol,
        stage2,
        calibration,
        [baseline, *candidates],
        split_seed=75260723,
    )
    baseline_rows = calibration_rows[baseline.name]
    calibration_comparisons = {}
    for index, candidate in enumerate(candidates):
        calibration_comparisons[candidate.name] = compare(
            protocol,
            calibration_rows[candidate.name],
            baseline_rows,
            seed=int(protocol["statistics"]["bootstrap_seed"]) + index,
        )
        calibration_comparisons[candidate.name]["eligible"] = (
            calibration_candidate_is_eligible(
                protocol,
                calibration_comparisons[candidate.name],
            )
        )
    selected = select_candidate(
        protocol,
        candidates,
        calibration_comparisons,
        calibration_output["summary"],
    )
    result = {
        "version": "1.0",
        "status": "calibration_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_result_sha256": protocol["development_source"][
                "cache_result_sha256"
            ],
            "stage2_config_sha256": protocol["link"]["stage2_config_sha256"],
            "calibration_cache_sha256": cache_result["splits"][
                source["calibration_split"]
            ]["cache_sha256"],
            "validation_cache_sha256": cache_result["splits"][
                source["validation_split"]
            ]["cache_sha256"],
        },
        "calibration": {
            **calibration_output,
            "comparisons_vs_absolute_index_always": calibration_comparisons,
            "selected_policy": None
            if selected is None
            else {
                "name": selected.name,
                "regret_threshold_db": selected.regret_threshold_db,
                "max_age_minutes": selected.max_age_minutes,
            },
        },
        "validation": None,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "changes_stage4_final_algorithm_or_claim": False,
            "confirmatory_final": False,
            "data_role": "development",
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    if selected is not None:
        validation = split_cache(
            protocol,
            cache_result,
            source["validation_split"],
        )
        policies = [
            baseline,
            Policy(
                "periodic_absolute_60min",
                "periodic_absolute",
                0.0,
                60.0,
            ),
            Policy(
                "event_absolute_selected",
                "event_absolute",
                selected.regret_threshold_db,
                selected.max_age_minutes,
            ),
            Policy(
                "event_delta_selected",
                "event_delta",
                selected.regret_threshold_db,
                selected.max_age_minutes,
            ),
        ]
        validation_rows, validation_output = run_policies(
            protocol,
            stage2,
            validation,
            policies,
            split_seed=86260723,
        )
        validation_comparisons = {}
        for index, policy in enumerate(policies[1:]):
            validation_comparisons[policy.name] = compare(
                protocol,
                validation_rows[policy.name],
                validation_rows[baseline.name],
                seed=int(protocol["statistics"]["bootstrap_seed"]) + 100 + index,
            )
        result["status"] = "stage5_fixed_demand_development_validation_complete"
        result["validation"] = {
            **validation_output,
            "comparisons_vs_absolute_index_always": validation_comparisons,
            "rows": [
                row
                for policy in policies
                for row in validation_rows[policy.name]
            ],
        }
    else:
        result["status"] = "stage5_calibration_negative_no_validation_run"
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "selected_policy": result["calibration"]["selected_policy"],
                "final_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
