#!/usr/bin/env python
"""Run post-Final supplementary baselines on development caches only."""

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
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy, paired_cluster_bootstrap  # noqa: E402
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    block_regret_db, guarded_fallback_start, quantize_unit_interval, selected_block_start,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def load_cache(protocol_entry: dict) -> tuple[dict, dict[str, np.ndarray]]:
    result_path = PROJECT_DIR / protocol_entry["path"]
    if "c1_v4_final" in result_path.as_posix().lower():
        raise ValueError("supplementary baselines must not load a Final path")
    if sha256_file(result_path) != protocol_entry["sha256"]:
        raise ValueError(f"development cache result hash mismatch: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("governance", {}).get("development_only") is not True:
        raise ValueError("supplementary input is not marked development-only")
    meta = result["splits"][protocol_entry["split"]]
    cache_path = PROJECT_DIR / meta["cache"]
    if sha256_file(cache_path) != meta["cache_sha256"]:
        raise ValueError("development cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    return result, cache


def summarize(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows], dtype=np.float64)
    return {
        "scene_count": len(rows),
        "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_frame_success_rate": float(np.mean([row["frame_success_rate"] for row in rows])),
        "mean_link_latency_ms": float(np.mean([row["link_latency_ms"] for row in rows])),
        "application_bits": int(rows[0]["application_bits"]),
    }


def paired(proposed: list[dict], baseline: list[dict], stats: dict, seed: int) -> dict:
    if [row["scene_id"] for row in proposed] != [row["scene_id"] for row in baseline]:
        raise RuntimeError("supplementary comparison is not paired")
    return paired_cluster_bootstrap(
        np.asarray([row["regret_db"] for row in proposed]),
        np.asarray([row["regret_db"] for row in baseline]),
        np.asarray([row["actual_bits"] for row in proposed]),
        np.asarray([row["actual_bits"] for row in baseline]),
        [row["cluster_id"] for row in baseline],
        repetitions=int(stats["paired_cluster_bootstrap_repetitions"]),
        seed=int(seed), cvar_alpha=float(stats["cvar_alpha"]),
        one_sided_confidence=float(stats["one_sided_confidence"]),
    )


def representation_values(method: str, cache: dict[str, np.ndarray], position: int, hard_threshold: float) -> tuple[np.ndarray, str, int]:
    if method == "hard_mean_energy_1bit":
        return (cache["features"][position, :, 1] > hard_threshold).astype(np.float64), "channel", 1
    if method == "occupancy_fraction_4bit":
        return quantize_unit_interval(cache["occupancy"][position], 4), "channel", 4
    if method == "soft_channel_power_4bit":
        return quantize_unit_interval(cache["normalized_channel_power"][position], 4), "channel", 4
    if method == "soft_channel_power_8bit":
        return quantize_unit_interval(cache["normalized_channel_power"][position], 8), "channel", 8
    if method == "proposed_best_block_indicator_1bit":
        return cache["best_block_indicator"][position].astype(np.float64), "block", 1
    raise KeyError(method)


def run_representation(protocol: dict, cache: dict[str, np.ndarray]) -> dict:
    methods = protocol["representation_methods"]
    stage2 = json.loads((PROJECT_DIR / protocol["link"]["stage2_config"]).read_text(encoding="utf-8"))
    demand = int(protocol["resource_task"]["demand_channels"])
    threshold = float(protocol["resource_task"]["hard_mean_z_threshold"])
    acc = {method["name"]: {position: {"regret": [], "bits": [], "success": [], "latency": []} for position in range(len(cache["scene_ids"]))} for method in methods}
    for channel_index, channel in enumerate(protocol["link"]["channel_models"]):
        for ebn0_index, ebn0 in enumerate(protocol["link"]["ebn0_db"]):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(int(protocol["link"]["repeats"])):
                for position in range(len(cache["scene_ids"])):
                    seed = 81260723 + position * 100_000 + channel_index * 1000 + ebn0_index * 100 + repetition
                    for method in methods:
                        values, kind, _ = representation_values(method["name"], cache, position, threshold)
                        true_start = selected_block_start(values, kind, demand)
                        tx = transmit_payload_analytic(int(method["application_bits"]), link, seed)
                        start = true_start if tx.frame_success else 0
                        row = acc[method["name"]][position]
                        row["regret"].append(block_regret_db(start, cache["block_cost_dbm"][position]))
                        row["bits"].append(tx.transmitted_bits); row["success"].append(tx.frame_success); row["latency"].append(tx.duration_s * 1000.0)
    rows = []
    for method in methods:
        name = method["name"]
        for position in range(len(cache["scene_ids"])):
            values = acc[name][position]
            rows.append({
                "scene_id": str(cache["scene_ids"][position]), "site": str(cache["site_ids"][position]),
                "cluster_id": str(cache["cluster_ids"][position]), "method": name,
                "regret_db": float(np.mean(values["regret"])), "actual_bits": float(np.mean(values["bits"])),
                "frame_success_rate": float(np.mean(values["success"])), "link_latency_ms": float(np.mean(values["latency"])),
                "application_bits": int(method["application_bits"]),
            })
    by_method = {method["name"]: [row for row in rows if row["method"] == method["name"]] for method in methods}
    alpha = float(protocol["statistics"]["cvar_alpha"])
    summaries = {name: summarize(values, alpha) for name, values in by_method.items()}
    proposed = by_method["proposed_best_block_indicator_1bit"]
    comparisons = {}
    for index, baseline in enumerate(name for name in by_method if name != "proposed_best_block_indicator_1bit"):
        comparisons[baseline] = paired(proposed, by_method[baseline], protocol["statistics"], int(protocol["statistics"]["representation_seed"]) + index)
    cost = cache["block_cost_dbm"].astype(np.float64)
    static_regret = cost[:, 0] - np.min(cost, axis=1)
    random_expected_regret = np.mean(cost - np.min(cost, axis=1, keepdims=True), axis=1)
    references = {
        "static_block0_no_reporting": {"mean_regret_db": float(np.mean(static_regret)), "cvar_0_9_regret_db": empirical_cvar_numpy(static_regret, alpha), "actual_bits": 0.0},
        "uniform_random_block_no_reporting": {"mean_regret_db": float(np.mean(random_expected_regret)), "cvar_0_9_regret_db": empirical_cvar_numpy(random_expected_regret, alpha), "actual_bits": 0.0, "note": "per-scene expected regret over five uniform block choices"},
        "oracle_best_block_free_information": {"mean_regret_db": 0.0, "cvar_0_9_regret_db": 0.0, "actual_bits": 0.0, "note": "infeasible free-information lower bound"},
    }
    return {"summary": summaries, "comparisons_proposed_minus_baseline": comparisons, "references": references, "rows": rows}


def run_reliability(protocol: dict, cache: dict[str, np.ndarray]) -> dict:
    methods = protocol["reliability_methods"]
    stage2 = json.loads((PROJECT_DIR / protocol["link"]["stage2_config"]).read_text(encoding="utf-8"))
    demand = int(protocol["resource_task"]["demand_channels"])
    app_bits = 157
    order = np.lexsort((cache["timestamps_local"].astype(str), cache["site_ids"].astype(str)))
    acc = {method["name"]: {int(position): {"regret": [], "bits": [], "success": [], "latency": []} for position in order} for method in methods}
    method_map = {method["name"]: method for method in methods}
    for channel_index, channel in enumerate(protocol["link"]["channel_models"]):
        for ebn0_index, ebn0 in enumerate(protocol["link"]["ebn0_db"]):
            base = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(int(protocol["link"]["repeats"])):
                state_by_site: dict[str, tuple[int, str]] = {}
                for position_value in order:
                    position = int(position_value); site = str(cache["site_ids"][position]); timestamp = str(cache["timestamps_local"][position])
                    true_start = selected_block_start(cache["best_block_indicator"][position], "block", demand)
                    seed = 82260723 + position * 100_000 + channel_index * 1000 + ebn0_index * 100 + repetition
                    tx_by_retry = {retry: transmit_payload_analytic(app_bits, replace(base, max_retransmissions=retry), seed) for retry in (0, 1, 2)}
                    standard = tx_by_retry[1]; previous = state_by_site.get(site)
                    starts = {
                        "no_arq_stateless": true_start if tx_by_retry[0].frame_success else 0,
                        "standard_arq_stateless": true_start if standard.frame_success else 0,
                        "extra_arq_stateless": true_start if tx_by_retry[2].frame_success else 0,
                        "standard_arq_unlimited_hold": true_start if standard.frame_success else (previous[0] if previous else 0),
                        "standard_arq_age60_hold": true_start if standard.frame_success else guarded_fallback_start(previous[0] if previous else None, previous[1] if previous else None, timestamp, max_age_minutes=60.0, default_start=0),
                    }
                    if standard.frame_success:
                        state_by_site[site] = (true_start, timestamp)
                    for name, start in starts.items():
                        tx = tx_by_retry[int(method_map[name]["max_retransmissions"])]
                        row = acc[name][position]
                        row["regret"].append(block_regret_db(start, cache["block_cost_dbm"][position]))
                        row["bits"].append(tx.transmitted_bits); row["success"].append(tx.frame_success); row["latency"].append(tx.duration_s * 1000.0)
    rows = []
    for method in methods:
        name = method["name"]
        for position_value in order:
            position = int(position_value); values = acc[name][position]
            rows.append({
                "scene_id": str(cache["scene_ids"][position]), "site": str(cache["site_ids"][position]),
                "cluster_id": str(cache["cluster_ids"][position]), "method": name,
                "regret_db": float(np.mean(values["regret"])), "actual_bits": float(np.mean(values["bits"])),
                "frame_success_rate": float(np.mean(values["success"])), "link_latency_ms": float(np.mean(values["latency"])),
                "application_bits": app_bits,
            })
    by_method = {method["name"]: [row for row in rows if row["method"] == method["name"]] for method in methods}
    alpha = float(protocol["statistics"]["cvar_alpha"])
    summaries = {name: summarize(values, alpha) for name, values in by_method.items()}
    proposed = by_method["standard_arq_age60_hold"]
    comparisons = {}
    for index, baseline in enumerate(name for name in by_method if name != "standard_arq_age60_hold"):
        comparisons[baseline] = paired(proposed, by_method[baseline], protocol["statistics"], int(protocol["statistics"]["reliability_seed"]) + index)
    return {"summary": summaries, "comparisons_age60_minus_baseline": comparisons, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_supplementary_baselines.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_supplementary_baselines_v1")
    args = parser.parse_args()
    output = args.out_dir / "supplementary_baselines_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite supplementary baseline result v1")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["governance"].get("final_measurements_or_final_metrics_may_be_loaded") is not False:
        raise ValueError("supplementary governance must prohibit Final loading")
    final_access = json.loads((PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json").read_text(encoding="utf-8"))
    if final_access.get("access_count") != 1:
        raise RuntimeError("supplementary baselines require the completed Final state without rerunning it")
    rep_result, rep_cache = load_cache(protocol["representation_cache_result"])
    rel_result, rel_cache = load_cache(protocol["reliability_cache_result"])
    result = {
        "version": "1.0", "status": "postfinal_supplementary_development_baselines_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "representation_cache_result_sha256": sha256_file(PROJECT_DIR / protocol["representation_cache_result"]["path"]),
            "reliability_cache_result_sha256": sha256_file(PROJECT_DIR / protocol["reliability_cache_result"]["path"]),
            "representation_scene_count": len(rep_cache["scene_ids"]), "reliability_scene_count": len(rel_cache["scene_ids"]),
        },
        "representation_baselines": run_representation(protocol, rep_cache),
        "reliability_baselines": run_reliability(protocol, rel_cache),
        "governance": {
            "final_access_count_observed": 1, "final_measurement_values_loaded": False,
            "final_metrics_loaded": False, "changes_final_claim": False,
            "data_role": "already-seen development validation only", "confirmatory": False,
        },
        "environment": environment_snapshot(["numpy"]), "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "representation_methods": len(protocol["representation_methods"]), "reliability_methods": len(protocol["reliability_methods"]), "final_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
