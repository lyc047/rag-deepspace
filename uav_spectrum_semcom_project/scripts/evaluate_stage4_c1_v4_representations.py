#!/usr/bin/env python
"""Evaluate frozen C1-v4 resource representations on time-blocked development data."""

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
    selected_block_start,
    transmit_resource_payload,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def payloads_for_index(cache: dict[str, np.ndarray], index: int, overhead: int) -> tuple[ResourceSemanticPayload, ...]:
    return (
        ResourceSemanticPayload("occupancy_fixed4", cache["occupancy"][index], 4, "channel", overhead),
        ResourceSemanticPayload("channel_power_fixed4", cache["normalized_channel_power"][index], 4, "channel", overhead),
        ResourceSemanticPayload("block_score_fixed4", cache["normalized_block_cost"][index], 4, "block", overhead),
        ResourceSemanticPayload("block_score_fixed6", cache["normalized_block_cost"][index], 6, "block", overhead),
        ResourceSemanticPayload("best_block_indicator1", cache["best_block_indicator"][index], 1, "block", overhead),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_development_protocol.json")
    parser.add_argument("--cache-result", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_real_cache_v1/c1_v4_cache_result.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_representation_v1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cache_result = json.loads(args.cache_result.read_text(encoding="utf-8"))
    if cache_result.get("governance", {}).get("prior_holdout_measurement_values_loaded") is not False:
        raise ValueError("C1-v4 cache governance is invalid")
    stage2_path = PROJECT_DIR / protocol["development_link"]["stage2_config"]
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    task = protocol["resource_task"]
    demand = int(task["demand_channels"])
    overhead = int(protocol["application_overhead_bits"])
    link_rule = protocol["development_link"]
    expected_names = [row["name"] for row in protocol["frozen_representations"]]
    all_rows = []
    split_summary = {}
    scene_values = {}
    for split in ("calibration", "validation"):
        meta = cache_result["splits"][split]
        cache_path = PROJECT_DIR / meta["cache"]
        if sha256_file(cache_path) != meta["cache_sha256"]:
            raise ValueError(f"{split} cache hash mismatch")
        cache = load_cache(cache_path)
        method_scene = {name: {"regret": [], "bits": [], "success": [], "clean": []} for name in expected_names}
        for index, scene_id in enumerate(cache["scene_ids"].astype(str)):
            block_cost = cache["block_cost_dbm"][index]
            payloads = payloads_for_index(cache, index, overhead)
            if [payload.name for payload in payloads] != expected_names:
                raise ValueError("runtime payload order differs from frozen protocol")
            for payload in payloads:
                clean_start = selected_block_start(payload.quantized_values, payload.representation_kind, demand)
                clean_regret = block_regret_db(clean_start, block_cost)
                regrets, bits, successes = [], [], []
                for channel_index, channel in enumerate(link_rule["channel_models"]):
                    for ebn0_index, ebn0 in enumerate(link_rule["ebn0_db"]):
                        link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
                        for repetition in range(int(link_rule["repeats"])):
                            common_seed = 50260722 + index * 100_000 + channel_index * 1000 + ebn0_index * 100 + repetition
                            outcome = transmit_resource_payload(
                                payload,
                                scene_id=scene_id,
                                block_cost_dbm=block_cost,
                                demand_channels=demand,
                                link=link,
                                seed=common_seed,
                            )
                            regrets.append(outcome.regret_db); bits.append(outcome.transmitted_bits); successes.append(outcome.frame_success)
                row = {
                    "split": split, "scene_id": scene_id, "site": str(cache["site_ids"][index]),
                    "cluster_id": str(cache["cluster_ids"][index]), "method": payload.name,
                    "application_bits": payload.application_bits, "clean_regret_db": clean_regret,
                    "regret_db": float(np.mean(regrets)), "actual_bits": float(np.mean(bits)),
                    "frame_success_rate": float(np.mean(successes)),
                }
                all_rows.append(row)
                method_scene[payload.name]["regret"].append(row["regret_db"])
                method_scene[payload.name]["bits"].append(row["actual_bits"])
                method_scene[payload.name]["success"].append(row["frame_success_rate"])
                method_scene[payload.name]["clean"].append(clean_regret)
            if (index + 1) % 30 == 0:
                print(f"{split}: evaluated {index + 1}/{len(cache['scene_ids'])}", flush=True)
        summary = {}
        for name, values in method_scene.items():
            regret = np.asarray(values["regret"], dtype=np.float64)
            summary[name] = {
                "application_bits": next(row["application_bits"] for row in all_rows if row["split"] == split and row["method"] == name),
                "mean_clean_regret_db": float(np.mean(values["clean"])),
                "mean_regret_db": float(np.mean(regret)),
                "cvar_0_9_regret_db": empirical_cvar_numpy(regret, float(protocol["development_statistics"]["cvar_alpha"])),
                "mean_actual_bits": float(np.mean(values["bits"])),
                "mean_frame_success_rate": float(np.mean(values["success"])),
            }
        split_summary[split] = summary
        scene_values[split] = (cache, method_scene)

    primary = protocol["development_statistics"]["primary_split"]
    cache, methods = scene_values[primary]
    baseline = methods["occupancy_fixed4"]
    comparisons = {}
    stats_rule = protocol["development_statistics"]
    for name in expected_names[1:]:
        proposed = methods[name]
        comparison = paired_cluster_bootstrap(
            np.asarray(proposed["regret"]), np.asarray(baseline["regret"]),
            np.asarray(proposed["bits"]), np.asarray(baseline["bits"]),
            cache["cluster_ids"].astype(str),
            repetitions=int(stats_rule["paired_cluster_bootstrap_repetitions"]),
            seed=int(stats_rule["bootstrap_seed"]),
            cvar_alpha=float(stats_rule["cvar_alpha"]),
        )
        comparison["admissible_mean_regret_noninferior"] = comparison["mean_regret_upper_bound"] <= 0.0
        comparison["admissible_actual_bits_lower"] = comparison["actual_bit_upper_bound"] < 0.0
        comparison["admissible"] = comparison["admissible_mean_regret_noninferior"] and comparison["admissible_actual_bits_lower"]
        comparisons[name] = comparison
    admissible = [name for name, row in comparisons.items() if row["admissible"]]
    selected = None
    if admissible:
        selected = min(
            admissible,
            key=lambda name: (
                split_summary[primary][name]["mean_regret_db"], split_summary[primary][name]["cvar_0_9_regret_db"],
                split_summary[primary][name]["mean_actual_bits"], split_summary[primary][name]["application_bits"], name,
            ),
        )
    result = {
        "version": "1.0", "status": "c1_v4_frozen_representation_development_complete",
        "protocol_sha256": sha256_file(args.protocol), "cache_result_sha256": sha256_file(args.cache_result),
        "stage2_config_sha256": sha256_file(stage2_path), "split_summary": split_summary,
        "validation_cluster_bootstrap_vs_occupancy_fixed4": comparisons,
        "selected_representation": selected,
        "selection_status": "candidate_selected_for_future_independent_confirmation" if selected else "no_representation_met_preregistered_admissibility",
        "rows": all_rows,
        "governance": {"prior_holdout_values_loaded": False, "prior_holdout_metrics_loaded": False, "validation_used_after_representation_freeze": True, "result_is_confirmatory": False},
        "environment": environment_snapshot(["numpy"]), "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "c1_v4_representation_result.json"
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "selected_representation": selected, "confirmatory": False}, indent=2))


if __name__ == "__main__":
    main()
