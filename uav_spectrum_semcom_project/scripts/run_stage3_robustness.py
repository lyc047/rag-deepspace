"""Fresh-holdout Stage-3 robustness study with calibrated detector confidence."""

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
from spectrum_semcom.correlated_scenes import compose_max_hold_scene, sliding_source_indices
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.multi_uav_fusion import (
    NodeOccupancyReport,
    fit_monotonic_confidence_calibrator,
    fuse_occupancy,
    perturb_correlated_spectrogram,
)
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_task_link import infer_arrays_with_confidence, truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from run_stage3_fusion_baselines import evaluate_estimates
from simulate_resource_optimization import frame_channel_occupancy


def load_temporal_scenes(root: Path, split: str, offset: int, count: int, sources_per_scene: int):
    source_count = count + sources_per_scene - 1
    frames = iter_raddet_frames(
        root,
        split,
        read_metadata=False,
        default_sequence_length=1_000_000,
        max_frames=source_count,
        frame_offset=offset,
    )
    groups = sliding_source_indices(len(frames), sources_per_scene, stride=1)[:count]
    arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in frames]
    scenes = [compose_max_hold_scene([arrays[int(index)] for index in group]) for group in groups]
    truths = [[box for index in group for box in truth_boxes(frames[int(index)])] for group in groups]
    ids = ["+".join(frames[int(index)].stem for index in group) for group in groups]
    if len(scenes) != count:
        raise ValueError(f"requested {count} scenes but only constructed {len(scenes)}")
    return scenes, truths, ids


def infer_node_views(scenes, config, model, model_info, device: str, repeat: int, seed_offset: int = 0):
    sensing = config["sensing_channel"]
    nodes = len(sensing["snr_db_per_node"])
    views: list[np.ndarray] = []
    for scene_index, scene in enumerate(scenes):
        for node in range(nodes):
            rng = np.random.default_rng(int(config["seed"]) + seed_offset + repeat * 1_000_000 + scene_index * 100 + node)
            views.append(
                perturb_correlated_spectrogram(
                    scene,
                    float(sensing["snr_db_per_node"][node]),
                    rng,
                    float(sensing["gain_db_per_node"][node]),
                    int(sensing["frequency_shift_bins_per_node"][node]),
                )
            )
    predictions, raw_confidence = infer_arrays_with_confidence(
        views,
        model,
        float(model_info["threshold"]),
        int(model_info["min_cells"]),
        device,
        batch_size=64,
    )
    return predictions, np.asarray(raw_confidence, dtype=np.float64).reshape(len(scenes), nodes)


def report_quality(occupancy: np.ndarray, truth: np.ndarray, demand_channels: int) -> float:
    """Return task-aligned report correctness without exposing truth at inference."""

    demand = int(demand_channels)
    estimated_blocks = np.asarray([
        np.mean(occupancy[start : start + demand])
        for start in range(len(occupancy) - demand + 1)
    ])
    truth_blocks = np.asarray([
        np.mean(truth[start : start + demand])
        for start in range(len(truth) - demand + 1)
    ])
    chosen = int(np.argmin(estimated_blocks))
    return float(abs(float(truth_blocks[chosen]) - float(np.min(truth_blocks))) <= 1e-9)


def fit_calibrator(scenes, truths, config, model, model_info, device: str):
    protocol = config["protocol"]
    predictions, raw = infer_node_views(scenes, config, model, model_info, device, repeat=0, seed_offset=30_000_000)
    nodes = raw.shape[1]
    scores: list[float] = []
    qualities: list[float] = []
    for scene_index, truth in enumerate(truths):
        truth_occ = frame_channel_occupancy(truth, int(protocol["n_channels"]), str(protocol["channel_axis"]))
        for node in range(nodes):
            boxes = predictions[scene_index * nodes + node]
            occupancy = frame_channel_occupancy(boxes, int(protocol["n_channels"]), str(protocol["channel_axis"]))
            scores.append(float(raw[scene_index, node]))
            qualities.append(report_quality(occupancy, truth_occ, int(protocol["demand_channels"])))
    calibrator = fit_monotonic_confidence_calibrator(
        np.asarray(scores),
        np.asarray(qualities),
        int(config["robustness_protocol"]["confidence_calibration_bins"]),
    )
    before = float(np.mean((np.asarray(scores) - np.asarray(qualities)) ** 2))
    after = float(np.mean((calibrator.predict(np.asarray(scores)) - np.asarray(qualities)) ** 2))
    return calibrator, {"validation_raw_mse": before, "validation_calibrated_mse": after, "sample_count": len(scores)}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["clean_resource_rate", "mean_occupancy_regret", "oracle_equivalent_rate", "mean_transmitted_bits"]
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["scenario"]), str(row["method"])) for row in rows})
    for scenario, method in keys:
        items = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        merged: dict[str, Any] = {"scenario": scenario, "method": method, "repeat_count": len(items)}
        for metric in metrics:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{name}"] = value
        output.append(merged)
    return output


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        mean_rows = {int(row["repeat"]): row for row in rows if row["scenario"] == scenario and row["method"] == "mean"}
        quality_rows = {int(row["repeat"]): row for row in rows if row["scenario"] == scenario and row["method"] == "quality_weighted"}
        repeats = sorted(set(mean_rows) & set(quality_rows))
        regret = [float(quality_rows[r]["mean_occupancy_regret"] - mean_rows[r]["mean_occupancy_regret"]) for r in repeats]
        clean = [float(quality_rows[r]["clean_resource_rate"] - mean_rows[r]["clean_resource_rate"]) for r in repeats]
        row: dict[str, Any] = {"scenario": scenario, "comparison": "quality_weighted_minus_mean"}
        for prefix, values in [("regret_delta", regret), ("clean_rate_delta", clean)]:
            for name, value in mean_std_ci(values).items():
                row[f"{prefix}_{name}"] = value
        output.append(row)
    return output


def make_report(summary, deltas, calibration, scene_count: int, repeat_count: int) -> str:
    lines = [
        "# Stage 3 Robustness and Failure-Boundary Report",
        "",
        f"Fresh holdout: {scene_count} temporally overlapping semi-synthetic scenes, {repeat_count} repeated link/noise realizations.",
        "The first 800 previously consumed test sources are excluded; this run starts at frozen test-source offset 1200.",
        "All fusion methods within a scenario use identical delivered reports and transmitted-bit traces.",
        "",
        "## Confidence calibration",
        "",
        f"Validation samples: {calibration['sample_count']}; raw quality MSE {calibration['validation_raw_mse']:.6f}; calibrated MSE {calibration['validation_calibrated_mse']:.6f}.",
        f"Fresh-holdout raw quality MSE {calibration['holdout_raw_mse']:.6f}; calibrated MSE {calibration['holdout_calibrated_mse']:.6f}.",
        "",
        "## Results",
        "",
        "| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['clean_resource_rate_mean']:.4f} | "
            f"{row['mean_occupancy_regret_mean']:.6f} | {row['oracle_equivalent_rate_mean']:.4f} | {row['mean_transmitted_bits_mean']:.1f} |"
        )
    lines += ["", "## Paired proposed-minus-mean differences", "", "Negative regret delta favors quality-aware fusion.", "", "| Scenario | Regret delta (95% CI) | Clean-rate delta (95% CI) |", "|---|---:|---:|"]
    for row in deltas:
        lines.append(
            f"| {row['scenario']} | {row['regret_delta_mean']:.6f} "
            f"[{row['regret_delta_ci95_low']:.6f}, {row['regret_delta_ci95_high']:.6f}] | "
            f"{row['clean_rate_delta_mean']:.4f} [{row['clean_rate_delta_ci95_low']:.4f}, {row['clean_rate_delta_ci95_high']:.4f}] |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "These are controlled RadDet spectrogram perturbations and sliding max-hold scenes, not synchronized multi-receiver measurements.",
        "A scenario supports H3 only when the paired regret-delta 95% interval is below zero at identical bit cost.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage3_multi_uav_fusion.json")
    parser.add_argument("--stage2-config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-scenes", type=int, default=None, help="Diagnostic override; omit for the frozen 200-scene holdout.")
    parser.add_argument("--repeats", type=int, default=None, help="Diagnostic override; omit for the frozen five repeats.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage3" / "robustness_holdout")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    stage2 = json.loads(args.stage2_config.read_text(encoding="utf-8"))
    robust = config["robustness_protocol"]
    protocol = config["protocol"]
    scenes_count = int(args.max_scenes or robust["holdout_scenes"])
    repeats = int(args.repeats or robust["repeat_count"])
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_info = json.loads(args.mask_result.read_text(encoding="utf-8"))
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()

    val_scenes, val_truths, val_ids = load_temporal_scenes(
        args.root,
        str(robust["validation_split"]),
        int(robust["validation_source_offset"]),
        int(robust["validation_scenes"]),
        int(robust["sources_per_scene"]),
    )
    calibrator, calibration = fit_calibrator(val_scenes, val_truths, config, model, model_info, device)
    scenes, truths, scene_ids = load_temporal_scenes(
        args.root,
        str(robust["holdout_split"]),
        int(robust["holdout_source_offset"]),
        scenes_count,
        int(robust["sources_per_scene"]),
    )

    sensing = config["sensing_channel"]
    reporting = config["reporting_channel"]
    nodes = len(sensing["snr_db_per_node"])
    rows: list[dict[str, Any]] = []
    holdout_raw_confidence_errors: list[float] = []
    holdout_calibrated_confidence_errors: list[float] = []
    for repeat in range(repeats):
        predictions, raw_confidence = infer_node_views(scenes, config, model, model_info, device, repeat, seed_offset=60_000_000)
        for scenario_index, scenario in enumerate(robust["scenarios"]):
            estimates = {method: [] for method in robust["methods"]}
            total_bits = 0
            active = set(int(x) for x in scenario["active_nodes"])
            forced_failure = set(int(x) for x in scenario.get("forced_failure_nodes", []))
            anomaly = set(int(x) for x in scenario.get("anomaly_nodes", []))
            lag_steps = [int(x) for x in scenario["lag_steps"]]
            for scene_index, truth in enumerate(truths):
                received: list[NodeOccupancyReport] = []
                truth_occ = frame_channel_occupancy(truth, int(protocol["n_channels"]), str(protocol["channel_axis"]))
                for node in range(nodes):
                    if node not in active or node in forced_failure:
                        continue
                    observation_index = max(0, scene_index - lag_steps[node])
                    boxes = predictions[observation_index * nodes + node]
                    occupancy = frame_channel_occupancy(boxes, int(protocol["n_channels"]), str(protocol["channel_axis"]))
                    confidence = float(calibrator.predict(raw_confidence[observation_index, node]))
                    if node in anomaly:
                        occupancy = np.roll(occupancy, int(scenario.get("anomaly_channel_shift", 1)))
                        confidence = float(scenario.get("anomaly_confidence", 0.99))
                    link = build_link_config(stage2, float(reporting["ebn0_db_per_node"][node]))
                    link_seed = int(config["seed"]) + repeat * 1_000_000 + scene_index * 100 + node + 80_000_000
                    result = transmit_payload_analytic(hard_box_packet_bits(len(boxes)), link, link_seed)
                    total_bits += result.transmitted_bits
                    if not result.frame_success:
                        continue
                    reliability = float(1.0 / (1.0 + np.exp(-(float(reporting["ebn0_db_per_node"][node]) - 3.0) / 2.0)))
                    age_s = lag_steps[node] * float(robust["temporal_step_s"])
                    received.append(NodeOccupancyReport(occupancy, float(sensing["snr_db_per_node"][node]), confidence, reliability, age_s))
                    if scenario["id"] == "baseline_4_nodes":
                        actual_quality = report_quality(occupancy, truth_occ, int(protocol["demand_channels"]))
                        holdout_raw_confidence_errors.append(
                            (float(raw_confidence[observation_index, node]) - actual_quality) ** 2
                        )
                        holdout_calibrated_confidence_errors.append((confidence - actual_quality) ** 2)
                for method in robust["methods"]:
                    estimate = np.zeros(int(protocol["n_channels"]), dtype=np.float32) if not received else fuse_occupancy(
                        received,
                        str(method),
                        float(protocol["occupancy_vote_threshold"]),
                        float(protocol["freshness_tau_s"]),
                    )
                    estimates[str(method)].append(estimate)
            for method, values in estimates.items():
                rows.append({
                    "scenario": str(scenario["id"]),
                    "method": method,
                    "repeat": repeat,
                    **evaluate_estimates(np.stack(values), truths, protocol),
                    "mean_transmitted_bits": float(total_bits / len(scenes)),
                })

    calibration["holdout_raw_mse"] = float(np.mean(holdout_raw_confidence_errors))
    calibration["holdout_calibrated_mse"] = float(np.mean(holdout_calibrated_confidence_errors))
    summary = aggregate(rows)
    deltas = paired_deltas(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "robustness_trials.csv", rows)
    write_csv(args.out_dir / "robustness_summary.csv", summary)
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    (args.out_dir / "stage3_robustness_report.md").write_text(
        make_report(summary, deltas, calibration, len(scenes), repeats), encoding="utf-8"
    )
    result = {
        "config_sha256": sha256_file(args.config),
        "holdout_registry_sha256": sha256_file(PROJECT_DIR / "configs" / "stage3_holdout_registry.json"),
        "validation_scene_ids_sha256": sha256_strings(val_ids),
        "holdout_scene_ids_sha256": sha256_strings(scene_ids),
        "calibrator": {"score_knots": calibrator.score_knots.tolist(), "quality_knots": calibrator.quality_knots.tolist()},
        "calibration": calibration,
        "summary": summary,
        "paired_deltas": deltas,
        "environment": environment_snapshot(["numpy", "torch"]),
        "claim_boundary": config["claim_boundary"],
    }
    (args.out_dir / "stage3_robustness_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "stage3_robustness_report.md")


if __name__ == "__main__":
    main()
