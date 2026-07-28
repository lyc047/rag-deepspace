"""Stage-3 smoke study on correlated same-scene multi-UAV fusion baselines."""

from __future__ import annotations

import argparse
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
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from simulate_resource_optimization import all_channel_occupancy, block_occupancy, frame_channel_occupancy


def evaluate_estimates(estimates: np.ndarray, truths: list[list[np.ndarray]], protocol: dict[str, Any]) -> dict[str, float]:
    truth_channels = all_channel_occupancy(truths, int(protocol["n_channels"]), str(protocol["channel_axis"]))
    truth_resources = block_occupancy(truth_channels, int(protocol["demand_channels"]))
    estimated_resources = block_occupancy(estimates, int(protocol["demand_channels"]))
    chosen = np.argmin(estimated_resources, axis=1)
    oracle = np.argmin(truth_resources, axis=1)
    rows = np.arange(len(truths))
    chosen_occ = truth_resources[rows, chosen]
    oracle_occ = truth_resources[rows, oracle]
    return {
        "clean_resource_rate": float(np.mean(chosen_occ <= float(protocol["clean_threshold"]))),
        "mean_occupancy_regret": float(np.mean(chosen_occ - oracle_occ)),
        "mean_selected_occupancy": float(np.mean(chosen_occ)),
        "oracle_equivalent_rate": float(np.mean(np.abs(chosen_occ - oracle_occ) <= 1e-9)),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["clean_resource_rate", "mean_occupancy_regret", "mean_selected_occupancy", "oracle_equivalent_rate", "mean_transmitted_bits"]
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        merged: dict[str, Any] = {"method": method, "repeat_count": len(items)}
        for metric in metrics:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def make_report(summary: list[dict[str, Any]], scenes: int, nodes: int) -> str:
    lines = [
        "# Stage 3 Correlated Multi-UAV Fusion Smoke Report", "", "## Protocol", "",
        f"{scenes} underlying RadDet scenes are each observed by {nodes} controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.", "",
        "This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.", "", "## Results", "",
        "| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['clean_resource_rate_mean']:.4f} [{max(0,row['clean_resource_rate_ci95_low']):.4f}, {min(1,row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} [{max(0,row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['oracle_equivalent_rate_mean']:.4f} | {row['mean_transmitted_bits_mean']:.1f} |"
        )
    lines.extend(["", "## Claim boundary", "", "- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.", "- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.", "- A learned fusion model is not yet evaluated.", "- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available."])
    return "\n".join(lines) + "\n"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_multi_uav_fusion.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sources-per-scene", type=int, default=1, help="Number of frozen source frames max-held into one controlled scene.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--n-channels", type=int, default=None)
    parser.add_argument("--demand-channels", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "fusion_smoke")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8")); assert_valid_experiment_metadata(config)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    protocol = dict(config["protocol"])
    if args.split is not None:
        protocol["test_split"] = args.split
    if args.n_channels is not None:
        protocol["n_channels"] = int(args.n_channels)
    if args.demand_channels is not None:
        protocol["demand_channels"] = int(args.demand_channels)
    if int(protocol["demand_channels"]) > int(protocol["n_channels"]):
        raise ValueError("demand channels cannot exceed candidate channels")
    max_scenes = int(args.max_scenes or protocol["smoke_max_scenes"])
    sources_per_scene = int(args.sources_per_scene)
    source_frames = iter_raddet_frames(args.root, protocol["test_split"], read_metadata=False, default_sequence_length=1_000_000, max_frames=max_scenes * sources_per_scene)
    groups = grouped_source_indices(len(source_frames), sources_per_scene)
    source_arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in source_frames]
    arrays = [compose_max_hold_scene([source_arrays[int(index)] for index in group]) for group in groups]
    truths = [[box for index in group for box in truth_boxes(source_frames[int(index)])] for group in groups]
    scene_ids = ["+".join(source_frames[int(index)].stem for index in group) for group in groups]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    model_info = json.loads(args.mask_result.read_text(encoding="utf-8"))
    model = TinyOccupancyCNN(channels=16).to(device); model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device)); model.eval()

    sensing = config["sensing_channel"]; reporting = config["reporting_channel"]
    snrs = [float(x) for x in sensing["snr_db_per_node"]]; gains = [float(x) for x in sensing["gain_db_per_node"]]
    shifts = [int(x) for x in sensing["frequency_shift_bins_per_node"]]; ebn0s = [float(x) for x in reporting["ebn0_db_per_node"]]
    ages = [float(x) for x in reporting["age_s_per_node"]]
    if len({len(snrs), len(gains), len(shifts), len(ebn0s), len(ages)}) != 1: raise ValueError("all per-node arrays must have equal length")
    nodes = len(snrs); rows: list[dict[str, Any]] = []
    for repeat in range(int(config["repeat_count"])):
        node_views: list[np.ndarray] = []
        for scene_index, array in enumerate(arrays):
            for node in range(nodes):
                rng = np.random.default_rng(int(config["seed"]) + repeat * 1_000_000 + scene_index * 100 + node)
                node_views.append(perturb_correlated_spectrogram(array, snrs[node], rng, gains[node], shifts[node]))
        predictions = infer_arrays(node_views, model, float(model_info["threshold"]), int(model_info["min_cells"]), device, batch_size=64)
        estimates = {method: [] for method in protocol["fusion_methods"]}; total_bits = 0
        for scene_index in range(len(arrays)):
            received: list[NodeOccupancyReport] = []
            for node in range(nodes):
                boxes = predictions[scene_index * nodes + node]
                link = build_link_config(stage2, ebn0s[node])
                result = transmit_payload_analytic(hard_box_packet_bits(len(boxes)), link, int(config["seed"]) + repeat * 1_000_000 + scene_index * 100 + node + 50_000_000)
                total_bits += result.transmitted_bits
                if not result.frame_success: continue
                occupancy = frame_channel_occupancy(boxes, int(protocol["n_channels"]), str(protocol["channel_axis"]))
                reliability = float(1.0 / (1.0 + np.exp(-(ebn0s[node] - 3.0) / 2.0)))
                received.append(NodeOccupancyReport(occupancy, snrs[node], 1.0, reliability, ages[node]))
            for method in protocol["fusion_methods"]:
                estimates[method].append(np.zeros(int(protocol["n_channels"]), dtype=np.float32) if not received else fuse_occupancy(received, method, float(protocol["occupancy_vote_threshold"]), float(protocol["freshness_tau_s"])))
        for method, values in estimates.items():
            rows.append({"method": method, "repeat": repeat, **evaluate_estimates(np.stack(values), truths, protocol), "mean_transmitted_bits": float(total_bits / len(arrays))})
    summary = aggregate(rows); args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fusion_trials.csv", rows); write_csv(args.out_dir / "fusion_summary.csv", summary)
    (args.out_dir / "stage3_fusion_smoke_report.md").write_text(make_report(summary, len(arrays), nodes), encoding="utf-8")
    result = {"config_sha256": sha256_file(args.config), "scene_ids_sha256": sha256_strings(scene_ids), "sources_per_scene": sources_per_scene, "summary": summary, "environment": environment_snapshot(["numpy", "torch"]), "claim_boundary": config["claim_boundary"]}
    (args.out_dir / "stage3_fusion_smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_fusion_smoke_report.md")


if __name__ == "__main__":
    main()
