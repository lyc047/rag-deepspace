"""Train a small DeepSets fusion pilot on source-disjoint correlated scenes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.correlated_scenes import compose_max_hold_scene, grouped_source_indices
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.multi_uav_fusion import NodeOccupancyReport, fuse_occupancy, perturb_correlated_spectrogram
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_task_link import infer_arrays, truth_boxes
from run_stage2_digital_link import build_link_config, write_csv
from run_stage3_fusion_baselines import evaluate_estimates
from simulate_resource_optimization import frame_channel_occupancy


def make_dataset(split: str, scene_count: int, config: dict[str, Any], stage2: dict[str, Any], model, device: str) -> tuple[np.ndarray, np.ndarray, list[list[str]], float]:
    protocol = config["protocol"]; sensing = config["sensing_channel"]; reporting = config["reporting_channel"]
    n_nodes = len(sensing["snr_db_per_node"]); sources_per_scene = int(config["learned_fusion_pilot"]["sources_per_scene"])
    root = PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2"
    source_frames = iter_raddet_frames(root, split, read_metadata=False, default_sequence_length=1_000_000, max_frames=scene_count * sources_per_scene)
    groups = grouped_source_indices(len(source_frames), sources_per_scene)
    source_arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in source_frames]
    scenes = [compose_max_hold_scene([source_arrays[int(index)] for index in group]) for group in groups]
    truths = [[box for index in group for box in truth_boxes(source_frames[int(index)])] for group in groups]
    ids = [[source_frames[int(index)].stem for index in group] for group in groups]
    views: list[np.ndarray] = []
    split_offset = {"train": 0, "val": 10_000_000, "test": 20_000_000}[split]
    for scene_index, scene in enumerate(scenes):
        for node in range(n_nodes):
            rng = np.random.default_rng(int(config["seed"]) + split_offset + scene_index * 100 + node)
            views.append(perturb_correlated_spectrogram(scene, float(sensing["snr_db_per_node"][node]), rng, float(sensing["gain_db_per_node"][node]), int(sensing["frequency_shift_bins_per_node"][node])))
    result_path = PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json"
    model_info = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = infer_arrays(views, model, float(model_info["threshold"]), int(model_info["min_cells"]), device, batch_size=64)
    n_channels = int(protocol["n_channels"]); feature_rows: list[np.ndarray] = []; target_rows: list[np.ndarray] = []; total_bits = 0
    for scene_index, truth in enumerate(truths):
        node_rows: list[np.ndarray] = []
        for node in range(n_nodes):
            boxes = predictions[scene_index * n_nodes + node]
            link = build_link_config(stage2, float(reporting["ebn0_db_per_node"][node]))
            result = transmit_payload_analytic(hard_box_packet_bits(len(boxes)), link, int(config["seed"]) + split_offset + 50_000_000 + scene_index * 100 + node)
            total_bits += result.transmitted_bits
            delivered = float(result.frame_success)
            occupancy = frame_channel_occupancy(boxes, n_channels, str(protocol["channel_axis"])) if delivered else np.zeros(n_channels, dtype=np.float32)
            snr_norm = float(np.clip((float(sensing["snr_db_per_node"][node]) + 20.0) / 40.0, 0.0, 1.0))
            reliability = float(1.0 / (1.0 + np.exp(-(float(reporting["ebn0_db_per_node"][node]) - 3.0) / 2.0)))
            age_norm = float(np.clip(float(reporting["age_s_per_node"][node]) / float(protocol["freshness_tau_s"]), 0.0, 4.0) / 4.0)
            node_rows.append(np.concatenate([occupancy, np.asarray([snr_norm, reliability, age_norm, delivered], dtype=np.float32)]))
        feature_rows.append(np.stack(node_rows))
        target_rows.append(frame_channel_occupancy(truth, n_channels, str(protocol["channel_axis"])))
    return np.stack(feature_rows).astype(np.float32), np.stack(target_rows).astype(np.float32), ids, float(total_bits / max(len(groups), 1))


def evaluate_classic(features: np.ndarray, targets: np.ndarray, config: dict[str, Any], method: str) -> np.ndarray:
    protocol = config["protocol"]; reports_out: list[np.ndarray] = []
    for sample in features:
        reports: list[NodeOccupancyReport] = []
        for node in sample:
            if node[-1] < 0.5:
                continue
            snr = float(node[-4] * 40.0 - 20.0); reliability = float(node[-3]); age = float(node[-2] * 4.0 * float(protocol["freshness_tau_s"]))
            reports.append(NodeOccupancyReport(node[: int(protocol["n_channels"])], snr, 1.0, reliability, age))
        reports_out.append(np.zeros(int(protocol["n_channels"]), dtype=np.float32) if not reports else fuse_occupancy(reports, method, float(protocol["occupancy_vote_threshold"]), float(protocol["freshness_tau_s"])))
    return np.stack(reports_out)


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DeepSetFusion(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        def forward(self, values):
            delivered = values[..., -1:].clamp(0.0, 1.0)
            embedded = self.phi(values) * delivered
            pooled = embedded.sum(dim=1) / delivered.sum(dim=1).clamp(min=1.0)
            return torch.sigmoid(self.rho(pooled))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_multi_uav_fusion.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "deepset_pilot")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); assert_valid_experiment_metadata(config)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8")); pilot = config["learned_fusion_pilot"]; protocol = config["protocol"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    detector = TinyOccupancyCNN(channels=16).to(device); detector.load_state_dict(torch.load(args.mask_checkpoint, map_location=device)); detector.eval()
    train_x, train_y, train_ids, bits_train = make_dataset(str(pilot["train_split"]), int(pilot["train_scenes"]), config, stage2, detector, device)
    val_x, val_y, val_ids, bits_val = make_dataset(str(pilot["validation_split"]), int(pilot["validation_scenes"]), config, stage2, detector, device)
    test_x, test_y, test_ids, bits_test = make_dataset(str(pilot["test_split"]), int(pilot["test_scenes"]), config, stage2, detector, device)
    model = DeepSetFusion(train_x.shape[-1], train_y.shape[-1], int(pilot["hidden_dim"])).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=float(pilot["learning_rate"]))
    tx, ty = torch.tensor(train_x, device=device), torch.tensor(train_y, device=device); vx, vy = torch.tensor(val_x, device=device), torch.tensor(val_y, device=device)
    best_state = None; best_value = float("inf"); history: list[dict[str, float]] = []
    demand = int(protocol["demand_channels"]); temperature = float(pilot["softmin_temperature"])
    def loss_fn(pred, truth):
        mse = F.mse_loss(pred, truth)
        pred_blocks = pred.unfold(1, demand, 1).mean(dim=-1); truth_blocks = truth.unfold(1, demand, 1).mean(dim=-1)
        task = (torch.softmax(-temperature * pred_blocks, dim=1) * truth_blocks).sum(dim=1).mean()
        return mse + float(pilot["resource_loss_weight"]) * task
    for epoch in range(int(pilot["epochs"])):
        model.train(); optimizer.zero_grad(); loss = loss_fn(model(tx), ty); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): value = float(loss_fn(model(vx), vy).item())
        history.append({"epoch": epoch + 1, "train_loss": float(loss.item()), "validation_loss": value})
        if value < best_value: best_value = value; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): learned = model(torch.tensor(test_x, device=device)).cpu().numpy()
    # Convert occupancy targets back to box-like per-scene truths through direct resource evaluation.
    # evaluate_estimates only needs truth boxes, so use a synthetic box-free adapter below.
    def metric_from_channels(estimates, truths_channels):
        pred_blocks = np.stack([estimates[:, start : start + demand].mean(axis=1) for start in range(estimates.shape[1] - demand + 1)], axis=1)
        true_blocks = np.stack([truths_channels[:, start : start + demand].mean(axis=1) for start in range(truths_channels.shape[1] - demand + 1)], axis=1)
        chosen = np.argmin(pred_blocks, axis=1); oracle = np.argmin(true_blocks, axis=1); rows = np.arange(len(chosen)); selected = true_blocks[rows, chosen]; optimal = true_blocks[rows, oracle]
        return {"clean_resource_rate": float(np.mean(selected <= float(protocol["clean_threshold"]))), "mean_occupancy_regret": float(np.mean(selected-optimal)), "oracle_equivalent_rate": float(np.mean(np.abs(selected-optimal)<=1e-9))}
    rows: list[dict[str, Any]] = []
    for method in ["majority", "mean", "or", "snr_weighted", "quality_weighted"]:
        rows.append({"method": method, **metric_from_channels(evaluate_classic(test_x, test_y, config, method), test_y), "mean_transmitted_bits": bits_test})
    rows.append({"method": "deepset_resource_fusion", **metric_from_channels(learned, test_y), "mean_transmitted_bits": bits_test})
    args.out_dir.mkdir(parents=True, exist_ok=True); write_csv(args.out_dir / "training_history.csv", history); write_csv(args.out_dir / "fusion_test_summary.csv", rows)
    torch.save(model.state_dict(), args.out_dir / "deepset_fusion.pt")
    report = ["# Stage 3 DeepSets Fusion Pilot", "", "Source frames are disjoint across train/validation/test. Scenes are controlled max-hold composites and therefore remain semi-synthetic.", "", "| Method | Clean rate | Regret | Oracle-equivalent | bit/decision |", "|---|---:|---:|---:|---:|"]
    report += [f"| {row['method']} | {row['clean_resource_rate']:.4f} | {row['mean_occupancy_regret']:.6f} | {row['oracle_equivalent_rate']:.4f} | {row['mean_transmitted_bits']:.1f} |" for row in rows]
    report += ["", "The learned method and classical methods consume the same realized node reports. This pilot does not establish real UAV or synchronized-SDR performance."]
    (args.out_dir / "stage3_deepset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {"config_sha256": sha256_file(args.config), "best_validation_loss": best_value, "train_scene_hash": sha256_strings("+".join(ids) for ids in train_ids), "validation_scene_hash": sha256_strings("+".join(ids) for ids in val_ids), "test_scene_hash": sha256_strings("+".join(ids) for ids in test_ids), "summary": rows, "environment": environment_snapshot(["numpy", "torch"]), "claim_boundary": config["claim_boundary"]}
    (args.out_dir / "stage3_deepset_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_deepset_report.md")


if __name__ == "__main__":
    main()
