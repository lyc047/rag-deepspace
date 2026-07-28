#!/usr/bin/env python
"""Generate the frozen AERPAW scene metrics after single-use access is consumed.

``--preflight`` performs only byte-integrity and snapshot checks and is safe
before access.  The normal mode refuses to run until the access receipt exists;
it never consumes access itself and writes the metrics payload atomically.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.aerpaw_final_inference import occupancy_quality, power_block_regret_db  # noqa: E402
from spectrum_semcom.counterfactual_value import discrete_scene_regret  # noqa: E402
from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_path,
    verify_catalog_files,
)
from spectrum_semcom.final_statistics import FINAL_METRICS_SCHEMA, validate_final_metrics_payload  # noqa: E402
from spectrum_semcom.gate_a_model import VariableRateOccupancyHead  # noqa: E402
from spectrum_semcom.multigranular_semantics import SemanticMessage, transmit_semantic_message  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


RECOVERY_ALLOWED_CHANGES = {
    "configs/stage4_final_access_state.json",
    "scripts/run_aerpaw_three_site_final_inference.py",
}


def verify_snapshot(snapshot: dict, ignored_paths: set[str] | None = None) -> list[str]:
    errors = []
    ignored = ignored_paths or set()
    if canonical_json_sha256(snapshot.get("files", [])) != snapshot.get("executable_snapshot_sha256"):
        errors.append("snapshot file list does not match its executable SHA-256")
    for row in snapshot.get("files", []):
        if row["path"] in ignored:
            continue
        path = PROJECT_DIR / row["path"]
        if not path.is_file():
            errors.append(f"missing frozen file: {row['path']}")
        elif sha256_path(path) != row["sha256"]:
            errors.append(f"frozen file changed: {row['path']}")
    return errors


def load_label(scene: dict) -> dict:
    path = Path(scene["label_ref"])
    if sha256_path(path) != scene["label_sha256"]:
        raise ValueError(f"label changed after registration: {scene['scene_id']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("scene_id") != scene["scene_id"]:
        raise ValueError(f"label scene mismatch: {scene['scene_id']}")
    return value


def transmit_one(scene_id: str, node: int, occupancy: np.ndarray, bits: int, link, seed: int) -> tuple[np.ndarray | None, int]:
    message = SemanticMessage("G2", node, scene_id, occupancy, int(bits), occupancy_quality(occupancy), ())
    result = transmit_semantic_message(message, link, seed)
    decoded = None if result.decoded is None else result.decoded.occupancy
    return decoded, int(result.link.transmitted_bits)


def run_fixed_reporting(method: str, scene_id: str, occupancy: np.ndarray, links: tuple, prior_node: int, seed: int) -> tuple[np.ndarray, int]:
    plan = [(node, 1) for node in range(len(occupancy))]
    if method == "selective_G2":
        plan.append((prior_node, 4))
    elif method == "all_G2":
        plan = [(node, 4) for node in range(len(occupancy))]
    else:
        raise ValueError(f"unknown reporting method: {method}")
    latest: dict[int, np.ndarray] = {}
    transmitted = 0
    for index, (node, bits) in enumerate(plan):
        if bits == 1:
            message = SemanticMessage("G1", node, scene_id, occupancy[node], 1, None, ())
            result = transmit_semantic_message(message, links[node], seed + index * 1009)
            decoded = None if result.decoded is None else result.decoded.occupancy
            transmitted += int(result.link.transmitted_bits)
        else:
            decoded, used = transmit_one(scene_id, node, occupancy[node], bits, links[node], seed + index * 1009)
            transmitted += used
        if decoded is not None:
            latest[node] = decoded
    fused = np.mean(np.stack(list(latest.values())), axis=0) if latest else np.full(occupancy.shape[1], 0.5)
    return fused, transmitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-protocol", type=Path, default=PROJECT_DIR / "configs/aerpaw_final_inference_protocol_v1.json")
    parser.add_argument("--registry", type=Path, default=PROJECT_DIR / "configs/stage4_final_registry_v1.json")
    parser.add_argument("--access-state", type=Path, default=PROJECT_DIR / "configs/stage4_final_access_state.json")
    parser.add_argument("--snapshot", type=Path, default=PROJECT_DIR / "results/stage4/pre_final_freeze_v1/pre_final_freeze_result.json")
    parser.add_argument("--recovery-snapshot", type=Path)
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs/stage2_digital_link.json")
    parser.add_argument("--stage4-protocol", type=Path, default=PROJECT_DIR / "configs/stage4_protocol.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/final_holdout_v1/final_scene_metrics.json")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    inference = json.loads(args.inference_protocol.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    access = json.loads(args.access_state.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    recovery = None
    ignored = RECOVERY_ALLOWED_CHANGES if args.recovery_snapshot is not None else set()
    errors = verify_snapshot(snapshot, ignored)
    if args.recovery_snapshot is not None:
        recovery = json.loads(args.recovery_snapshot.read_text(encoding="utf-8"))
        errors.extend(verify_snapshot(recovery))
        original_rows = {row["path"]: row["sha256"] for row in snapshot.get("files", [])}
        recovery_rows = {row["path"]: row["sha256"] for row in recovery.get("files", [])}
        changed = {path for path in set(original_rows) | set(recovery_rows) if original_rows.get(path) != recovery_rows.get(path)}
        if changed != RECOVERY_ALLOWED_CHANGES:
            errors.append(f"recovery changed unexpected frozen files: {sorted(changed)}")
    errors.extend(verify_catalog_files({"scenes": registry["scenes"]}, args.registry.resolve().parent))
    if registry.get("scene_count") != 200 or registry.get("real_multi_receiver_scene_count") != 200:
        errors.append("registered final set must contain exactly 200 three-site scenes")
    if args.preflight:
        if access.get("status") != "not_accessed" or access.get("access_count") != 0:
            errors.append("preflight requires unconsumed final access")
        if errors:
            raise ValueError("preflight failed: " + "; ".join(errors))
        print(json.dumps({"status": "ready_for_single_access", "scene_count": 200, "snapshot_sha256": snapshot["executable_snapshot_sha256"], "final_access_consumed": False}, indent=2))
        return
    if errors:
        raise ValueError("frozen input audit failed: " + "; ".join(errors))
    if access.get("status") != "access_consumed" or access.get("access_count") != 1:
        raise RuntimeError("normal inference requires the single-use access receipt")
    if recovery is None:
        raise RuntimeError("post-access inference requires an audited recovery snapshot")
    receipt = access.get("receipt", {})
    if receipt.get("registry_sha256") != canonical_json_sha256(registry):
        raise ValueError("access receipt is not bound to this registry")
    if receipt.get("code_snapshot_sha256") != snapshot.get("executable_snapshot_sha256"):
        raise ValueError("access receipt is not bound to this executable snapshot")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing final metrics payload")

    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    stage4 = json.loads(args.stage4_protocol.read_text(encoding="utf-8"))
    hidden = int(stage4["gate_a_controlled_training"]["hidden_dim"])
    seeds = [int(x) for x in inference["c1"]["training_seeds"]]
    models = {}
    for method in inference["c1"]["methods"]:
        models[method] = []
        for seed in seeds:
            checkpoint = PROJECT_DIR / inference["c1"]["checkpoint_pattern"].format(method=method, seed=seed)
            model = VariableRateOccupancyHead(8, hidden)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            models[method].append(model)

    channels = inference["reporting_link"]["channel_models"]
    ebn0_values = [float(x) for x in inference["reporting_link"]["global_ebn0_db"]]
    assignments = list(permutations(float(x) for x in inference["reporting_link"]["relative_site_offsets_db"]))
    repeats = int(inference["reporting_link"]["repeats"])
    sites = inference["sites"]
    prior_node = sites.index(inference["selective_g2"]["scheduler_prior_site"])
    demand = int(stage4["gate_a_development_registry"]["selected_task"]["demand_channels"])
    rows_c1, rows_g2 = [], []
    with torch.inference_mode():
        for scene_index, scene in enumerate(registry["scenes"]):
            label = load_label(scene)
            base = np.asarray([label["per_site"][site]["occupancy_proxy"] for site in sites], dtype=np.float32)
            truth = np.asarray(label["three_site_reference_occupancy_proxy"], dtype=np.float64)
            power = np.asarray(label["per_site"][inference["c1"]["reference_site"]]["channel_mean_power_dbm"], dtype=np.float64)
            outputs = {}
            tensor = torch.as_tensor(base)
            for method, method_models in models.items():
                outputs[method] = [tuple(x.cpu().numpy() if hasattr(x, "cpu") else x for x in model.infer(tensor)) for model in method_models]

            for method_index, method in enumerate(inference["c1"]["methods"]):
                regrets, bits_used = [], []
                for seed_index, (occupancy, probability_bits, _) in enumerate(outputs[method]):
                    for channel_index, channel in enumerate(channels):
                        for ebn0_index, ebn0 in enumerate(ebn0_values):
                            for assignment_index, offsets in enumerate(assignments):
                                link = replace(build_link_config(stage2, ebn0 + offsets[0]), channel=channel)
                                for repeat in range(repeats):
                                    seed_value = 20260716 + scene_index * 10_000_000 + method_index * 1_000_000 + seed_index * 100_000 + channel_index * 20_000 + ebn0_index * 3000 + assignment_index * 100 + repeat
                                    decoded, used = transmit_one(scene["scene_id"], 0, occupancy[0], int(probability_bits[0]), link, seed_value)
                                    belief = np.full(8, 0.5) if decoded is None else decoded
                                    regrets.append(power_block_regret_db(belief, power, demand)); bits_used.append(used)
                rows_c1.append({"scene_id": scene["scene_id"], "method": method, "regret": float(np.mean(regrets)), "actual_bits": float(np.mean(bits_used))})

            resource_outputs = outputs["detection_plus_resource"]
            for method_index, method in enumerate(inference["selective_g2"]["methods"]):
                regrets, bits_used = [], []
                for seed_index, (occupancy, _, _) in enumerate(resource_outputs):
                    for channel_index, channel in enumerate(channels):
                        for ebn0_index, ebn0 in enumerate(ebn0_values):
                            for assignment_index, offsets in enumerate(assignments):
                                links = tuple(replace(build_link_config(stage2, ebn0 + offsets[node]), channel=channel) for node in range(3))
                                for repeat in range(repeats):
                                    seed_value = 30260716 + scene_index * 10_000_000 + method_index * 1_000_000 + seed_index * 100_000 + channel_index * 20_000 + ebn0_index * 3000 + assignment_index * 100 + repeat
                                    fused, used = run_fixed_reporting(method, scene["scene_id"], occupancy, links, prior_node, seed_value)
                                    regrets.append(discrete_scene_regret(fused, truth, demand)); bits_used.append(used)
                rows_g2.append({"scene_id": scene["scene_id"], "method": method, "regret": float(np.mean(regrets)), "actual_bits": float(np.mean(bits_used))})
            if (scene_index + 1) % 10 == 0:
                print(f"completed {scene_index + 1}/200 scenes", flush=True)

    payload = {
        "schema_version": FINAL_METRICS_SCHEMA,
        "registry_sha256": receipt["registry_sha256"],
        "access_receipt_sha256": canonical_json_sha256(receipt),
        "families": {"C1_resource_loss": {"rows": rows_c1}, "selective_G2_digital_reporting": {"rows": rows_g2}},
        "audit": {"inference_protocol_sha256": sha256_file(args.inference_protocol), "stage2_config_sha256": sha256_file(args.stage2_config), "stage4_protocol_sha256": sha256_file(args.stage4_protocol), "aggregation_is_scene_level": True, "original_access_receipt_snapshot_sha256": snapshot["executable_snapshot_sha256"], "post_access_recovery_snapshot_sha256": recovery["executable_snapshot_sha256"], "recovery_changed_paths": sorted(RECOVERY_ALLOWED_CHANGES)},
    }
    validation = validate_final_metrics_payload(payload, 200)
    if validation:
        raise ValueError("generated final metrics failed schema validation: " + "; ".join(validation))
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "scenes": 200, "final_access_count": 1}, indent=2))


if __name__ == "__main__":
    main()
