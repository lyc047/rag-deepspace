"""Final stage-2 evidence: stronger PSD baselines and full-split robustness."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.bit_budget import packet_header_bits
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.psd_baselines import cfar_occupancy_summary, high_resolution_channel_psd, reduce_high_resolution_psd
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_classical_baselines import evaluate_resource_estimates, quantize_unit_interval
from evaluate_stage2_task_link import run_repeat, task_metrics, truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from run_stage2_full_test import load_or_create_features


def make_psd_features(frames, n_channels: int, bins: int, axis: str) -> np.ndarray:
    return np.stack([
        high_resolution_channel_psd(np.asarray(Image.open(frame.image_path).convert("L")), n_channels, bins, axis)
        for frame in frames
    ])


def score_psd(features: np.ndarray, n_channels: int, bins: int, mode: str, threshold_sigma: float) -> np.ndarray:
    if mode == "cfar":
        return np.stack([cfar_occupancy_summary(row, n_channels, bins, threshold_sigma) for row in features])
    return np.stack([reduce_high_resolution_psd(row, n_channels, bins, mode) for row in features])


def select_psd(val_features: dict[int, np.ndarray], val_truths, task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for candidate in candidates:
        candidate = {"threshold_sigma": None, **candidate}
        threshold_sigma = 3.0 if candidate["threshold_sigma"] is None else float(candidate["threshold_sigma"])
        scores = score_psd(val_features[int(candidate["bins_per_channel"])], int(task["n_channels"]), int(candidate["bins_per_channel"]), candidate["mode"], threshold_sigma)
        pressure = int(task["observations_per_pressure_decision"])
        usable = len(scores) - len(scores) % pressure
        # The task is one decision per four observations; use the same
        # equal-weight soft fusion as the link experiment but without a link
        # loss during representation-only validation selection.
        fused = scores[:usable].reshape(-1, pressure, scores.shape[1]).mean(axis=1)
        metrics = evaluate_resource_estimates(fused, val_truths[:usable], task)
        rows.append({**candidate, **metrics})
    # Validation-only lexicographic selection: task risk, then clean rate,
    # then lower source bits.  This cannot inspect test outcomes.
    selected = min(rows, key=lambda row: (row["mean_occupancy_regret"], -row["clean_resource_rate"], row["bins_per_channel"] * row["quantization_bits"]))
    return {"selected": selected, "candidates": rows}


def run_psd_link(features: np.ndarray, truths, task: dict[str, Any], link, selected: dict[str, Any], seed: int) -> dict[str, Any]:
    pressure, n_channels = int(task["observations_per_pressure_decision"]), int(task["n_channels"])
    bins, qbits, mode = int(selected["bins_per_channel"]), int(selected["quantization_bits"]), str(selected["mode"])
    # CFAR emits a four-value occupancy summary; high-resolution alternatives
    # transmit every sub-bin and only reduce after fusion.
    threshold_sigma = 3.0 if selected.get("threshold_sigma") is None else float(selected["threshold_sigma"])
    source = score_psd(features, n_channels, bins, mode, threshold_sigma) if mode == "cfar" else features
    application_bits = packet_header_bits() + source.shape[1] * qbits
    usable = len(source) - len(source) % pressure
    received, bits, durations, ratios, exhausted = [], [], [], [], []
    for index, value in enumerate(source[:usable]):
        result = transmit_payload_analytic(application_bits, link, seed + index)
        received.append(quantize_unit_interval(value, qbits) if result.frame_success else None)
        bits.append(result.transmitted_bits); durations.append(result.duration_s)
        ratios.append(result.payload_delivery_ratio); exhausted.append(float(result.latency_budget_exhausted))
    estimates, decision_latencies = [], []
    for start in range(0, usable, pressure):
        values = [row for row in received[start:start + pressure] if row is not None]
        fused = np.mean(values, axis=0) if values else np.zeros(source.shape[1], dtype=np.float32)
        estimates.append(fused if mode == "cfar" else reduce_high_resolution_psd(fused, n_channels, bins, mode))
        decision_latencies.append(max(durations[start:start + pressure]))
    metrics = evaluate_resource_estimates(np.stack(estimates), truths[:usable], task)
    return {"f1": None, **metrics, "mean_transmitted_bits": float(np.sum(bits) / (usable // pressure)), "mean_link_latency_s": float(np.mean(decision_latencies)), "mean_payload_delivery_ratio": float(np.mean(ratios)), "latency_budget_exhaustion_rate": float(np.mean(exhausted))}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["clean_resource_rate", "mean_occupancy_regret", "mean_transmitted_bits", "mean_link_latency_s", "mean_payload_delivery_ratio"]
    output = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        for scheme in sorted({str(row["scheme"]) for row in rows if row["condition"] == condition}):
            group = [row for row in rows if row["condition"] == condition and row["scheme"] == scheme]
            merged: dict[str, Any] = {"condition": condition, "scheme": scheme, "repeat_count": len(group)}
            f1 = [float(row["f1"]) for row in group if row["f1"] is not None]
            merged["f1_mean"] = float(np.mean(f1)) if f1 else None
            for metric in metrics:
                for name, value in mean_std_ci([float(row[metric]) for row in group]).items(): merged[f"{metric}_{name}"] = value
            output.append(merged)
    return output


def report(selection: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    lines = ["# Stage 2 Completion: Strong PSD Baseline and Full-Split Robustness", "", "## Validation-only PSD selection", "", f"Selected PSD: `{json.dumps(selection['selected'], ensure_ascii=False)}`. Candidate resolution/reducer choices were selected on validation data only; all task-link rows below use the frozen choice.", "", "## Full frozen-test results", "", "| Condition | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Latency |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        f1 = "n/a" if row["f1_mean"] is None else f"{row['f1_mean']:.4f}"
        lines.append(f"| {row['condition']} | {row['scheme']} | {f1} | {row['clean_resource_rate_mean']:.4f} [{row['clean_resource_rate_ci95_low']:.4f}, {row['clean_resource_rate_ci95_high']:.4f}] | {row['mean_occupancy_regret_mean']:.6f} [{row['mean_occupancy_regret_ci95_low']:.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | {row['mean_transmitted_bits_mean']:.1f} | {1000*row['mean_link_latency_s_mean']:.2f} ms |")
    lines.extend(["", "## Claim boundary", "", "- PSD remains normalized spectrogram intensity, not calibrated dBm receiver power.", "- Rayleigh/Rician use ideal receiver equalization; burst uses a declared Gilbert--Elliott packet-attempt model.", "- Four unrelated frames form a pressure decision, so these are H1/H4 results rather than same-scene multi-UAV H3 evidence.", "- RadDet remains synthetic radar data."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete stage 2 on the frozen split.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "completion")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8")); assert_valid_experiment_metadata(config)
    settings, task0 = config["stage2_completion"], config["task_validation"]
    task = {**task0, "observations_per_pressure_decision": settings["observations_per_pressure_decision"]}
    val_frames = iter_raddet_frames(args.root, settings["selection_split"], read_metadata=False, default_sequence_length=1_000_000, max_frames=int(settings["selection_max_frames"]))
    test_frames = iter_raddet_frames(args.root, settings["test_split"], read_metadata=False, default_sequence_length=1_000_000, max_frames=None)
    val_truths, truths = [truth_boxes(frame) for frame in val_frames], [truth_boxes(frame) for frame in test_frames]
    candidate_bins = sorted({int(row["bins_per_channel"]) for row in settings["psd_candidates"]})
    val_features = {bins: make_psd_features(val_frames, int(task["n_channels"]), bins, str(task["channel_axis"])) for bins in candidate_bins}
    selection = select_psd(val_features, val_truths, task, settings["psd_candidates"]); chosen = selection["selected"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    psd_cache = args.out_dir / f"highres_psd_{int(chosen['bins_per_channel'])}bin.npy"
    if psd_cache.exists():
        test_features = np.load(psd_cache)
        if len(test_features) != len(test_frames):
            raise ValueError("high-resolution PSD cache does not match frozen test-frame count")
        print(f"loaded {psd_cache}", flush=True)
    else:
        test_features = make_psd_features(test_frames, int(task["n_channels"]), int(chosen["bins_per_channel"]), str(task["channel_axis"]))
        np.save(psd_cache, test_features)
        print(f"wrote {psd_cache}", flush=True)
    # Reuse the existing detector cache for semantic recovery.
    import torch
    model_info = json.loads(args.mask_result.read_text(encoding="utf-8")); model = TinyOccupancyCNN(channels=16).to(args.device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=args.device)); model.eval()
    predictions, _ = load_or_create_features(test_frames, model, float(model_info["threshold"]), int(model_info["min_cells"]), args.device, int(config["full_test_validation"]["prediction_batch_size"]), args.out_dir / "semantic_feature_cache.json", True)
    rows = []
    for cindex, condition in enumerate(settings["conditions"]):
        link = replace(build_link_config(config, float(condition["ebn0_db"])), channel=condition["channel"], latency_budget_s=float(condition["latency_budget_s"]))
        for sindex, scheme in enumerate(["hard_semantic", "selected_strong_psd"]):
            for repeat in range(int(settings["repeat_count"])):
                seed = int(config["seed"]) + 50_000_000 + cindex * 100_000 + sindex * 10_000 + repeat * 1_000
                print(f"{condition['name']} {scheme} repeat={repeat+1}", flush=True)
                row = run_repeat("hard_semantic", test_frames, predictions, truths, None, float(model_info["threshold"]), int(model_info["min_cells"]), float(model_info["iou_threshold"]), task, link, seed, args.device) if scheme == "hard_semantic" else run_psd_link(test_features, truths, task, link, chosen, seed)
                row.update({"condition": condition["name"], "scheme": scheme, "seed": seed}); rows.append(row)
    summary = aggregate(rows); args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "psd_validation_selection.csv", selection["candidates"]); write_csv(args.out_dir / "completion_trials.csv", rows); write_csv(args.out_dir / "completion_summary.csv", summary)
    (args.out_dir / "stage2_completion_report.md").write_text(report(selection, summary), encoding="utf-8")
    (args.out_dir / "stage2_completion_result.json").write_text(json.dumps({"config_sha256": sha256_file(args.config), "selection": selection, "summary": summary, "test_frame_ids_sha256": sha256_strings(frame.stem for frame in test_frames), "environment": environment_snapshot(["numpy", "torch"]), "claim_boundary": {"full_frozen_test": True, "strong_psd_validation_only_selection": True, "burst_model": "Gilbert-Elliott", "calibrated_psd_dbm": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
