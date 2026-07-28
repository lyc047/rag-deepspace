#!/usr/bin/env python
"""Run the single-access C1 temporal confirmation after candidate freezing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.aerpaw_final_inference import occupancy_quality, power_block_regret_db  # noqa: E402
from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    channel_occupancy,
    channel_power_dbm,
    load_aerpaw_power_sweep,
    robust_power_threshold_dbm,
    select_frequency_band,
)
from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.gate_a_model import FixedPrecisionOccupancyHead  # noqa: E402
from spectrum_semcom.multigranular_semantics import SemanticMessage, transmit_semantic_message  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def verify_snapshot(snapshot: dict) -> list[str]:
    errors = []
    if canonical_json_sha256(snapshot.get("files", [])) != snapshot.get("executable_snapshot_sha256"):
        errors.append("snapshot file-list digest mismatch")
    for row in snapshot.get("files", []):
        path = PROJECT_DIR / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            errors.append(f"frozen file missing or changed: {row['path']}")
    return errors


def verify_scene_files(scenes: list[dict]) -> list[str]:
    errors = []
    for scene in scenes:
        for source in scene["source_files"]:
            path = Path(source["path"])
            if not path.is_file() or path.stat().st_size != int(source["size_bytes"]) or sha256_file(path) != source["sha256"]:
                errors.append(f"source missing or changed: {scene['scene_id']}/{source['role']}")
    return errors


def transmit_fixed4(scene_id: str, occupancy: np.ndarray, link, seed: int):
    message = SemanticMessage("G2", 0, scene_id, occupancy, 4, occupancy_quality(occupancy), ())
    result = transmit_semantic_message(message, link, seed)
    decoded = None if result.decoded is None else result.decoded.occupancy
    return decoded, int(result.link.transmitted_bits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/aerpaw_c1_temporal_holdout_protocol_v1.json")
    parser.add_argument("--inventory", type=Path, default=PROJECT_DIR / "results/stage4/aerpaw_c1_temporal_prefinal_v1/prefinal_inventory.json")
    parser.add_argument("--candidate", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_calibration_v1/selected_candidate_manifest.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_freeze_v1/executable_snapshot.json")
    parser.add_argument("--access-state", type=Path, default=PROJECT_DIR / "configs/stage4_c1_temporal_access_state.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs/stage2_digital_link.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_confirmation_v1/scene_metrics.json")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    access = json.loads(args.access_state.read_text(encoding="utf-8"))
    scenes = inventory["scenes"]["confirmation_holdout"]
    errors = verify_snapshot(snapshot)
    errors.extend(verify_scene_files(scenes))
    if len(scenes) != int(protocol["confirmation_statistics"]["scene_count"]):
        errors.append("confirmation scene count mismatch")
    if candidate.get("status") != "candidate_frozen_before_confirmation_access" or candidate.get("confirmation_accessed") is not False:
        errors.append("candidate manifest was not frozen before access")
    if args.output.exists():
        errors.append("confirmation output already exists")
    if args.preflight:
        if access.get("status") != "not_accessed" or access.get("access_count") != 0:
            errors.append("preflight requires unused temporal access")
        if errors:
            raise ValueError("preflight failed: " + "; ".join(errors))
        print(json.dumps({"status": "ready_for_single_temporal_access", "scenes": len(scenes), "clusters": inventory["confirmation_cluster_count"], "access_count": 0}, indent=2))
        return
    if errors:
        raise ValueError("confirmation audit failed: " + "; ".join(errors))
    if access.get("status") != "access_consumed" or access.get("access_count") != 1:
        raise RuntimeError("normal confirmation requires one consumed temporal access")
    receipt = access.get("receipt", {})
    bindings = {
        "protocol_sha256": sha256_file(args.protocol),
        "inventory_sha256": sha256_file(args.inventory),
        "candidate_manifest_sha256": sha256_file(args.candidate),
        "executable_snapshot_sha256": snapshot["executable_snapshot_sha256"],
    }
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise ValueError("temporal access receipt is not bound to current frozen inputs")

    proxy = protocol["resource_proxy"]
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in candidate["training_seeds"]]
    if len(seeds) != len(candidate["candidate_checkpoints"]) or len(seeds) != len(candidate["baseline_checkpoints"]):
        raise ValueError("checkpoint and seed counts differ")
    methods = {"baseline_fixed4": [], "c1_v3_tail_ranking_fixed4": []}
    for row in candidate["baseline_checkpoints"]:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("baseline checkpoint changed")
        model = FixedPrecisionOccupancyHead(8, 32, 4)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model.eval()
        methods["baseline_fixed4"].append(model)
    for row in candidate["candidate_checkpoints"]:
        path = PROJECT_DIR / row["checkpoint"]
        if sha256_file(path) != row["checkpoint_sha256"]:
            raise ValueError("candidate checkpoint changed")
        model = FixedPrecisionOccupancyHead(8, 32, 4)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model.eval()
        methods["c1_v3_tail_ranking_fixed4"].append(model)

    channel_models = ("awgn", "rayleigh", "rician")
    ebn0_values = (0.0, 3.0, 6.0, 9.0)
    repeats = 5
    rows = []
    with torch.inference_mode():
        for scene_index, scene in enumerate(scenes):
            source = {row["role"]: Path(row["path"]) for row in scene["source_files"]}
            sweep = load_aerpaw_power_sweep(source["meta"], source["power"])
            band = select_frequency_band(sweep, low_mhz=float(proxy["frequency_low_mhz_inclusive"]), high_mhz=float(proxy["frequency_high_mhz_exclusive"]))
            threshold = robust_power_threshold_dbm(band, sigma_multiplier=float(proxy["threshold_sigma_multiplier"]), gaussian_mad_scale=float(proxy["threshold_gaussian_mad_scale"]), sigma_floor_db=float(proxy["threshold_sigma_floor_db"]))
            base = channel_occupancy(band, 8, threshold)
            power = channel_power_dbm(band, 8)
            tensor = torch.as_tensor(base[None], dtype=torch.float32)
            for method_index, (method, models) in enumerate(methods.items()):
                regrets, bits = [], []
                for seed_index, model in enumerate(models):
                    occupancy = model.infer(tensor)[0].cpu().numpy()
                    for channel_index, channel in enumerate(channel_models):
                        for ebn0_index, ebn0 in enumerate(ebn0_values):
                            link = replace(build_link_config(stage2, ebn0), channel=channel)
                            for repeat_index in range(repeats):
                                common_seed = 40260716 + scene_index * 1_000_000 + seed_index * 10_000 + channel_index * 1000 + ebn0_index * 100 + repeat_index
                                decoded, used = transmit_fixed4(scene["scene_id"], occupancy, link, common_seed)
                                belief = np.full(8, 0.5) if decoded is None else decoded
                                regrets.append(power_block_regret_db(belief, power, int(proxy["demand_channels"])))
                                bits.append(used)
                rows.append({"scene_id": scene["scene_id"], "cluster_id": scene["cluster_id"], "stratum": scene["stratum"], "site": scene["site"], "method": method, "regret_db": float(np.mean(regrets)), "actual_bits": float(np.mean(bits)), "nominal_application_bits": 183})
            if (scene_index + 1) % 25 == 0:
                print(f"completed {scene_index + 1}/{len(scenes)} scenes", flush=True)
    payload = {
        "version": "1.0",
        "status": "temporal_confirmation_inference_complete",
        "access_receipt_sha256": canonical_json_sha256(receipt),
        "bindings": bindings,
        "selected_candidate_id": candidate["selected_candidate_id"],
        "scenes": len(scenes),
        "rows": rows,
        "aggregation": {"training_seeds": seeds, "channel_models": list(channel_models), "ebn0_db": list(ebn0_values), "link_repeats": repeats, "paired_common_random_numbers": True},
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "scenes": len(scenes), "rows": len(rows), "access_count": 1}, indent=2))


if __name__ == "__main__":
    main()
