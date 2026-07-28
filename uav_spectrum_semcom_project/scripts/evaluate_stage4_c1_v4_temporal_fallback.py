#!/usr/bin/env python
"""Evaluate zero-bit last-success fallback on a new C1-v4 development date."""

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
    ResourceSemanticPayload,
    block_regret_db,
    encode_resource_semantic_payload,
    selected_block_start,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_temporal_fallback_protocol.json")
    parser.add_argument("--cache-result", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_temporal_fallback_cache_v1/c1_v4_cache_result.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_temporal_fallback_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base_path = PROJECT_DIR / protocol["base_representation_result"]["path"]
    if sha256_file(base_path) != protocol["base_representation_result"]["sha256"]:
        raise ValueError("base C1-v4 representation result changed after fallback freeze")
    cache_result = json.loads(args.cache_result.read_text(encoding="utf-8"))
    if cache_result["governance"]["prior_holdout_measurement_values_loaded"] is not False:
        raise ValueError("fallback cache governance is invalid")
    split = "temporal_fallback_validation"
    meta = cache_result["splits"][split]
    cache_path = PROJECT_DIR / meta["cache"]
    if sha256_file(cache_path) != meta["cache_sha256"]:
        raise ValueError("fallback validation cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    order = np.lexsort((cache["timestamps_local"].astype(str), cache["site_ids"].astype(str)))
    stage2_path = PROJECT_DIR / protocol["development_link"]["stage2_config"]
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    link_rule = protocol["development_link"]
    demand = int(protocol["resource_task"]["demand_channels"])
    accumulators = {
        "stateless_block0": {index: {"regret": [], "bits": [], "success": []} for index in range(len(order))},
        "last_success_same_site": {index: {"regret": [], "bits": [], "success": []} for index in range(len(order))},
    }
    for channel_index, channel in enumerate(link_rule["channel_models"]):
        for ebn0_index, ebn0 in enumerate(link_rule["ebn0_db"]):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(int(link_rule["repeats"])):
                last_by_site: dict[str, int] = {}
                for position in order:
                    site = str(cache["site_ids"][position]); scene_id = str(cache["scene_ids"][position])
                    payload = ResourceSemanticPayload("best_block_indicator1", cache["best_block_indicator"][position], 1, "block", 152)
                    encoded = encode_resource_semantic_payload(payload, node_id=0, scene_id=scene_id)
                    seed = 60260722 + int(position) * 100_000 + channel_index * 1000 + ebn0_index * 100 + repetition
                    transmission = transmit_payload_analytic(int(encoded.size), link, seed)
                    true_start = selected_block_start(payload.quantized_values, "block", demand)
                    stateless_start = true_start if transmission.frame_success else 0
                    stateful_start = true_start if transmission.frame_success else last_by_site.get(site, 0)
                    if transmission.frame_success:
                        last_by_site[site] = true_start
                    for method, start in (("stateless_block0", stateless_start), ("last_success_same_site", stateful_start)):
                        row = accumulators[method][int(position)]
                        row["regret"].append(block_regret_db(start, cache["block_cost_dbm"][position]))
                        row["bits"].append(int(transmission.transmitted_bits)); row["success"].append(bool(transmission.frame_success))

    rows = []
    for position in range(len(order)):
        for method in accumulators:
            values = accumulators[method][position]
            rows.append(
                {
                    "scene_id": str(cache["scene_ids"][position]), "site": str(cache["site_ids"][position]),
                    "cluster_id": str(cache["cluster_ids"][position]), "method": method,
                    "regret_db": float(np.mean(values["regret"])), "actual_bits": float(np.mean(values["bits"])),
                    "frame_success_rate": float(np.mean(values["success"])), "application_bits": 157,
                }
            )
    by_method = {method: [row for row in rows if row["method"] == method] for method in accumulators}
    baseline = by_method["stateless_block0"]; proposed = by_method["last_success_same_site"]
    if [row["scene_id"] for row in baseline] != [row["scene_id"] for row in proposed]:
        raise RuntimeError("fallback rows are not paired")
    stats_rule = protocol["development_statistics"]
    inference = paired_cluster_bootstrap(
        np.asarray([row["regret_db"] for row in proposed]), np.asarray([row["regret_db"] for row in baseline]),
        np.asarray([row["actual_bits"] for row in proposed]), np.asarray([row["actual_bits"] for row in baseline]),
        [row["cluster_id"] for row in baseline], repetitions=int(stats_rule["paired_cluster_bootstrap_repetitions"]),
        seed=int(stats_rule["bootstrap_seed"]), cvar_alpha=float(stats_rule["cvar_alpha"]),
    )
    summary = {}
    for method, values in by_method.items():
        regret = np.asarray([row["regret_db"] for row in values])
        summary[method] = {
            "mean_regret_db": float(np.mean(regret)), "cvar_0_9_regret_db": empirical_cvar_numpy(regret, float(stats_rule["cvar_alpha"])),
            "mean_actual_bits": float(np.mean([row["actual_bits"] for row in values])),
            "mean_frame_success_rate": float(np.mean([row["frame_success_rate"] for row in values])),
        }
    gates = {
        "mean_regret_superiority": inference["mean_regret_upper_bound"] < 0.0,
        "cvar_superiority": inference["cvar_upper_bound"] < 0.0,
        "actual_bits_exactly_equal": inference["actual_bit_difference"] == 0.0 and inference["actual_bit_upper_bound"] == 0.0,
    }
    result = {
        "version": "1.0", "status": "c1_v4_last_success_fallback_development_complete",
        "protocol_sha256": sha256_file(args.protocol), "cache_result_sha256": sha256_file(args.cache_result),
        "summary": summary, "paired_cluster_bootstrap": inference, "gates": gates,
        "development_success": all(gates.values()), "rows": rows,
        "governance": {"feb12_used_once_after_method_freeze": True, "prior_holdout_values_loaded": False, "result_is_confirmatory": False},
        "environment": environment_snapshot(["numpy"]), "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "c1_v4_temporal_fallback_result.json"
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "development_success": result["development_success"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
