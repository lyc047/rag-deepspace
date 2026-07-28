"""Build split-separated Gate A caches from the frozen development registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.correlated_scenes import compose_max_hold_scene
from spectrum_semcom.gate_a_data import pool_probability_mask, resolve_raddet_paths, validate_cache_arrays
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import parse_raddet_label_file
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.resource_losses import discrete_occupancy_regret, empirical_cvar
from spectrum_semcom.stage4_development import validate_development_registry
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from evaluate_stage2_task_link import normalize_image
from simulate_resource_optimization import frame_channel_occupancy
from train_raddet_occupancy_mask import yolo_to_xyxy


def scene_arrays(root: Path, scene: dict, n_channels: int, axis: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    for frame_id in scene["source_frame_ids"]:
        image_path, label_path = resolve_raddet_paths(root, frame_id)
        images.append(np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8))
        for box in parse_raddet_label_file(label_path):
            boxes.append(yolo_to_xyxy(np.asarray([box.x_center, box.y_center, box.width, box.height], dtype=np.float32)))
    return compose_max_hold_scene(images), frame_channel_occupancy(boxes, n_channels, axis)


def infer_masks(arrays: list[np.ndarray], model, device: str, batch_size: int) -> list[np.ndarray]:
    import torch

    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            batch = np.stack([normalize_image(value) for value in arrays[start : start + batch_size]])[:, None]
            probabilities = torch.sigmoid(model(torch.as_tensor(batch, dtype=torch.float32, device=device)))[:, 0]
            output.extend(probabilities.cpu().numpy())
    return output


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument("--registry", type=Path, default=PROJECT_DIR / "configs" / "stage4_development_registry_v1.json")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--model-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage4" / "gate_a_cache_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_development_registry(registry)
    if errors:
        raise ValueError("invalid development registry: " + "; ".join(errors))
    task = registry["selected_task"]
    n_channels, demand, axis = int(task["n_channels"]), int(task["demand_channels"]), str(task["channel_axis"])
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)); model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, dict] = {}
    report_rows: list[dict] = []
    for split_name, entry in registry["splits"].items():
        arrays: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        scene_ids = [scene["scene_id"] for scene in entry["scenes"]]
        for scene in entry["scenes"]:
            image, truth = scene_arrays(args.root, scene, n_channels, axis)
            arrays.append(image); truths.append(truth)
        masks = infer_masks(arrays, model, device, args.batch_size)
        base = np.stack([pool_probability_mask(mask, n_channels, axis) for mask in masks]).astype(np.float32)
        truth = np.stack(truths).astype(np.float32)
        entropy = -(np.clip(base, 1e-7, 1 - 1e-7) * np.log2(np.clip(base, 1e-7, 1 - 1e-7)) + (1 - np.clip(base, 1e-7, 1 - 1e-7)) * np.log2(1 - np.clip(base, 1e-7, 1 - 1e-7)))
        confidence = np.clip(1.0 - entropy.mean(axis=1), 0.0, 1.0).astype(np.float32)
        cache_errors = validate_cache_arrays(scene_ids, base, truth, confidence, n_channels)
        if cache_errors:
            raise ValueError(f"invalid {split_name} cache: " + "; ".join(cache_errors))
        cache_path = args.out_dir / f"gate_a_{split_name}.npz"
        np.savez_compressed(cache_path, scene_ids=np.asarray(scene_ids), base_occupancy=base, truth_occupancy=truth, detector_confidence=confidence)
        pred_t = torch.as_tensor(base); truth_t = torch.as_tensor(truth)
        regrets = discrete_occupancy_regret(pred_t, truth_t, demand, reduction="none")
        row = {"split": split_name, "scenes": len(scene_ids), "mean_discrete_regret": float(regrets.mean()), "cvar_0_9_regret": float(empirical_cvar(regrets, 0.9)), "brier": float(np.mean((base-truth)**2))}
        report_rows.append(row)
        split_records[split_name] = {**row, "cache": str(cache_path.relative_to(PROJECT_DIR)).replace("\\", "/"), "cache_sha256": sha256_file(cache_path), "scene_ids_sha256": sha256_strings(scene_ids)}
    result = {"experiment_id": "stage4_gate_a_frozen_detector_cache_v1", "protocol_sha256": sha256_file(args.protocol), "registry_sha256": sha256_file(args.registry), "checkpoint_sha256": sha256_file(args.checkpoint), "model_result_sha256": sha256_file(args.model_result), "task": task, "splits": split_records, "environment": environment_snapshot(["numpy", "torch", "pillow"]), "final_holdout_accessed": False, "claim_boundary": "Development cache only; no final H2/H3 claim."}
    (args.out_dir / "gate_a_cache_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Stage 4 Gate A Frozen-Detector Cache", "", "Development splits only; final holdout was not created or accessed.", "", "| Split | Scenes | Mean discrete regret | CVaR0.9 | Brier |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {row['split']} | {row['scenes']} | {row['mean_discrete_regret']:.6f} | {row['cvar_0_9_regret']:.6f} | {row['brier']:.6f} |" for row in report_rows]
    (args.out_dir / "gate_a_cache_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(args.out_dir / "gate_a_cache_report.md")


if __name__ == "__main__":
    main()
