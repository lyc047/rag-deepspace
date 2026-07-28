#!/usr/bin/env python
"""Run the frozen, single-use C1-v4 temporal Final and inference."""

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
    ResourceSemanticPayload, block_regret_db, encode_resource_semantic_payload,
    guarded_fallback_start, selected_block_start,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def verify_frozen(protocol_path: Path, cache_result_path: Path, snapshot_path: Path, state_path: Path) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache_result = json.loads(cache_result_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "access_consumed" or state.get("access_count") != 1:
        raise RuntimeError("Final inference requires the single-use receipt")
    if canonical_json_sha256(snapshot.get("files", [])) != snapshot.get("executable_snapshot_sha256"):
        raise ValueError("snapshot digest mismatch")
    if state["receipt"].get("executable_snapshot_sha256") != snapshot["executable_snapshot_sha256"]:
        raise ValueError("access receipt and executable snapshot differ")
    for row in snapshot["files"]:
        path = PROJECT_DIR / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"frozen file missing or changed: {row['path']}")
    if cache_result.get("protocol_sha256") != sha256_file(protocol_path) or cache_result.get("governance", {}).get("access_count") != 1:
        raise ValueError("Final cache is not bound to this frozen protocol/access")
    return protocol, cache_result


def method_summary(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows], dtype=np.float64)
    return {
        "scene_count": len(rows), "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_frame_success_rate": float(np.mean([row["frame_success_rate"] for row in rows])),
        "application_bits": int(rows[0]["application_bits"]),
    }


def compare(proposed: list[dict], baseline: list[dict], stats: dict, seed: int) -> dict:
    if [row["scene_id"] for row in proposed] != [row["scene_id"] for row in baseline]:
        raise RuntimeError("Final comparison rows are not paired")
    return paired_cluster_bootstrap(
        np.asarray([row["regret_db"] for row in proposed]), np.asarray([row["regret_db"] for row in baseline]),
        np.asarray([row["actual_bits"] for row in proposed]), np.asarray([row["actual_bits"] for row in baseline]),
        [row["cluster_id"] for row in baseline], repetitions=int(stats["paired_cluster_bootstrap_repetitions"]),
        seed=int(seed), cvar_alpha=float(stats["cvar_alpha"]), one_sided_confidence=float(stats["one_sided_confidence"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json")
    parser.add_argument("--cache-result", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_cache_v1/c1_v4_final_cache_result.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_freeze_v1/executable_snapshot.json")
    parser.add_argument("--state", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_v1")
    args = parser.parse_args()
    output = args.out_dir / "c1_v4_final_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite the single-use C1-v4 Final result")
    protocol, cache_result = verify_frozen(args.protocol, args.cache_result, args.snapshot, args.state)
    cache_path = PROJECT_DIR / cache_result["cache"]
    if sha256_file(cache_path) != cache_result["cache_sha256"]:
        raise ValueError("Final cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    order = np.lexsort((cache["timestamps_local"].astype(str), cache["site_ids"].astype(str)))
    link_rule = protocol["link"]
    stage2 = json.loads((PROJECT_DIR / link_rule["stage2_config"]).read_text(encoding="utf-8"))
    demand = int(protocol["resource_task"]["demand_channels"])
    max_age = float(protocol["frozen_methods"]["indicator_guarded_last_success"]["max_state_age_minutes"])
    methods = ("occupancy_stateless", "indicator_stateless", "indicator_guarded_last_success")
    acc = {method: {int(index): {"regret": [], "bits": [], "success": []} for index in order} for method in methods}
    for channel_index, channel in enumerate(link_rule["channel_models"]):
        for ebn0_index, ebn0 in enumerate(link_rule["ebn0_db"]):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for repetition in range(int(link_rule["repeats"])):
                state_by_site: dict[str, tuple[int, str]] = {}
                for position_value in order:
                    position = int(position_value); site = str(cache["site_ids"][position]); timestamp = str(cache["timestamps_local"][position])
                    scene_id = str(cache["scene_ids"][position])
                    occupancy = ResourceSemanticPayload("occupancy_fixed4", cache["occupancy"][position], 4, "channel", 152)
                    indicator = ResourceSemanticPayload("best_block_indicator1", cache["best_block_indicator"][position], 1, "block", 152)
                    seed = 70260722 + position * 100_000 + channel_index * 1000 + ebn0_index * 100 + repetition
                    occupancy_tx = transmit_payload_analytic(int(encode_resource_semantic_payload(occupancy, node_id=0, scene_id=scene_id).size), link, seed)
                    indicator_tx = transmit_payload_analytic(int(encode_resource_semantic_payload(indicator, node_id=0, scene_id=scene_id).size), link, seed)
                    occupancy_true = selected_block_start(occupancy.quantized_values, "channel", demand)
                    indicator_true = selected_block_start(indicator.quantized_values, "block", demand)
                    occupancy_start = occupancy_true if occupancy_tx.frame_success else 0
                    indicator_start = indicator_true if indicator_tx.frame_success else 0
                    previous = state_by_site.get(site)
                    guarded_start = indicator_true if indicator_tx.frame_success else guarded_fallback_start(
                        previous[0] if previous else None, previous[1] if previous else None, timestamp,
                        max_age_minutes=max_age, default_start=0,
                    )
                    if indicator_tx.frame_success:
                        state_by_site[site] = (indicator_true, timestamp)
                    for method, start, tx in (
                        ("occupancy_stateless", occupancy_start, occupancy_tx),
                        ("indicator_stateless", indicator_start, indicator_tx),
                        ("indicator_guarded_last_success", guarded_start, indicator_tx),
                    ):
                        row = acc[method][position]
                        row["regret"].append(block_regret_db(start, cache["block_cost_dbm"][position]))
                        row["bits"].append(int(tx.transmitted_bits)); row["success"].append(bool(tx.frame_success))
    rows = []
    for position_value in order:
        position = int(position_value)
        for method in methods:
            values = acc[method][position]
            rows.append({
                "scene_id": str(cache["scene_ids"][position]), "site": str(cache["site_ids"][position]),
                "timestamp_local": str(cache["timestamps_local"][position]), "cluster_id": str(cache["cluster_ids"][position]),
                "method": method, "regret_db": float(np.mean(values["regret"])),
                "actual_bits": float(np.mean(values["bits"])), "frame_success_rate": float(np.mean(values["success"])),
                "application_bits": int(protocol["frozen_methods"][method]["application_bits"]),
            })
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    stats = protocol["statistics"]
    h1 = compare(by_method["indicator_stateless"], by_method["occupancy_stateless"], stats, int(stats["bootstrap_seed_representation"]))
    h2 = compare(by_method["indicator_guarded_last_success"], by_method["indicator_stateless"], stats, int(stats["bootstrap_seed_fallback"]))
    gates_h1 = {"mean_regret_superiority": h1["mean_regret_upper_bound"] < 0, "cvar_superiority": h1["cvar_upper_bound"] < 0, "actual_bit_superiority": h1["actual_bit_upper_bound"] < 0}
    gates_h2 = {"mean_regret_superiority": h2["mean_regret_upper_bound"] < 0, "cvar_superiority": h2["cvar_upper_bound"] < 0, "actual_bits_exactly_equal": h2["actual_bit_difference"] == 0 and h2["actual_bit_upper_bound"] == 0}
    h1_pass = all(gates_h1.values()); h2_raw_pass = all(gates_h2.values()); h2_confirmatory_pass = h1_pass and h2_raw_pass
    alpha = float(stats["cvar_alpha"])
    summary = {method: method_summary(values, alpha) for method, values in by_method.items()}
    site_summary = {site: {method: float(np.mean([row["regret_db"] for row in values if row["site"] == site])) for method, values in by_method.items()} for site in sorted(set(cache["site_ids"].astype(str)))}
    result = {
        "version": "1.0", "status": "c1_v4_single_use_temporal_final_complete",
        "protocol_sha256": sha256_file(args.protocol), "cache_result_sha256": sha256_file(args.cache_result),
        "access_receipt_sha256": json.loads(args.state.read_text(encoding="utf-8"))["receipt_sha256"],
        "summary": summary, "site_summary_mean_regret_db": site_summary,
        "hypotheses": {
            "H1_task_sufficient_representation": {"inference": h1, "gates": gates_h1, "confirmatory_pass": h1_pass},
            "H2_age_guarded_zero_bit_fallback": {"inference": h2, "gates": gates_h2, "raw_gates_pass": h2_raw_pass, "confirmatory_pass_under_fixed_sequence": h2_confirmatory_pass},
        },
        "overall_final_success": h1_pass and h2_confirmatory_pass, "rows": rows,
        "governance": {"access_count": 1, "algorithm_or_threshold_changed_after_access": False, "same_campaign_temporal_final": True},
        "environment": environment_snapshot(["numpy"]), "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "overall_final_success": result["overall_final_success"], "H1": h1_pass, "H2_fixed_sequence": h2_confirmatory_pass}, indent=2))


if __name__ == "__main__":
    main()
