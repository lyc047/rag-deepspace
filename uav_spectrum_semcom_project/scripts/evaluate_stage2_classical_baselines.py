from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_task_link import infer_arrays, run_repeat, task_metrics, truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from simulate_resource_optimization import all_channel_occupancy, block_occupancy


RESOURCE_METRICS = [
    "clean_resource_rate",
    "mean_occupancy_regret",
    "mean_selected_occupancy",
    "oracle_equivalent_rate",
    "mean_transmitted_bits",
    "mean_link_latency_s",
    "mean_payload_delivery_ratio",
    "latency_budget_exhaustion_rate",
]


def channel_energy(image: np.ndarray, n_channels: int, axis: str) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32) / 255.0
    dimension = 0 if axis == "y" else 1
    parts = np.array_split(value, n_channels, axis=dimension)
    return np.asarray([float(np.mean(part)) for part in parts], dtype=np.float32)


def quantize_unit_interval(values: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0 or bits > 16:
        raise ValueError("quantization bits must be in [1, 16]")
    levels = (1 << bits) - 1
    indices = np.rint(np.clip(values, 0.0, 1.0) * levels)
    return (indices / levels).astype(np.float32)


def grouped_truth_resources(truths: list[list[np.ndarray]], task: dict[str, Any]) -> np.ndarray:
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(truths) - len(truths) % pressure
    grouped = [
        [box for frame in truths[start : start + pressure] for box in frame]
        for start in range(0, usable, pressure)
    ]
    channels = all_channel_occupancy(grouped, int(task["n_channels"]), str(task["channel_axis"]))
    return block_occupancy(channels, int(task["demand_channels"]))


def evaluate_resource_estimates(
    estimated_channels: np.ndarray,
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
) -> dict[str, float]:
    truth_resources = grouped_truth_resources(truths, task)
    sensed_resources = block_occupancy(estimated_channels, int(task["demand_channels"]))
    chosen = np.argmin(sensed_resources, axis=1)
    oracle = np.argmin(truth_resources, axis=1)
    indices = np.arange(len(chosen))
    chosen_occupancy = truth_resources[indices, chosen]
    oracle_occupancy = truth_resources[indices, oracle]
    return {
        "clean_resource_rate": float(np.mean(chosen_occupancy <= float(task["clean_threshold"]))),
        "mean_occupancy_regret": float(np.mean(chosen_occupancy - oracle_occupancy)),
        "mean_selected_occupancy": float(np.mean(chosen_occupancy)),
        "oracle_equivalent_rate": float(np.mean(np.abs(chosen_occupancy - oracle_occupancy) <= 1e-9)),
    }


def run_quantized_psd(
    arrays: list[np.ndarray],
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
    link,
    quantization_bits: int,
    seed: int,
) -> dict[str, Any]:
    pressure = int(task["observations_per_pressure_decision"])
    n_channels = int(task["n_channels"])
    delivered: list[np.ndarray | None] = []
    transmitted_bits: list[int] = []
    durations: list[float] = []
    payload_ratios: list[float] = []
    exhausted: list[float] = []
    application_bits = packet_header_bits() + n_channels * quantization_bits
    for index, image in enumerate(arrays):
        energy = quantize_unit_interval(channel_energy(image, n_channels, str(task["channel_axis"])), quantization_bits)
        result = transmit_payload_analytic(application_bits, link, seed + index)
        delivered.append(energy if result.frame_success else None)
        transmitted_bits.append(result.transmitted_bits)
        durations.append(result.duration_s)
        payload_ratios.append(result.payload_delivery_ratio)
        exhausted.append(float(result.latency_budget_exhausted))

    usable = len(arrays) - len(arrays) % pressure
    estimates: list[np.ndarray] = []
    decision_latencies: list[float] = []
    for start in range(0, usable, pressure):
        received = [value for value in delivered[start : start + pressure] if value is not None]
        estimates.append(np.mean(received, axis=0) if received else np.zeros(n_channels, dtype=np.float32))
        decision_latencies.append(max(durations[start : start + pressure]))
    metrics = evaluate_resource_estimates(np.stack(estimates), truths[:usable], task)
    decisions = usable // pressure
    return {
        "f1": None,
        **metrics,
        "mean_transmitted_bits": float(np.sum(transmitted_bits[:usable]) / decisions),
        "mean_link_latency_s": float(np.mean(decision_latencies)),
        "mean_payload_delivery_ratio": float(np.mean(payload_ratios[:usable])),
        "latency_budget_exhaustion_rate": float(np.mean(exhausted[:usable])),
    }


def transmit_tiled_image(
    image: np.ndarray,
    link,
    tile_size: int,
    tile_header_bits: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    height, width = image.shape
    tiles = [(y, x) for y in range(0, height, tile_size) for x in range(0, width, tile_size)]
    rng.shuffle(tiles)
    restored = np.zeros_like(image)
    recovered_mask = np.zeros_like(image, dtype=bool)
    total_bits = 0
    total_duration = 0.0
    budget_exhausted = False
    recovered_tiles = 0
    for tile_index, (y, x) in enumerate(tiles):
        remaining = None if link.latency_budget_s is None else link.latency_budget_s - total_duration
        if remaining is not None and remaining <= 0:
            budget_exhausted = True
            break
        tile = image[y : y + tile_size, x : x + tile_size]
        tile_link = replace(link, latency_budget_s=remaining)
        result = transmit_payload_analytic(tile.size * 8 + tile_header_bits, tile_link, seed + tile_index + 1)
        total_bits += result.transmitted_bits
        total_duration += result.duration_s
        budget_exhausted = budget_exhausted or result.latency_budget_exhausted
        if result.frame_success:
            restored[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
            recovered_mask[y : y + tile.shape[0], x : x + tile.shape[1]] = True
            recovered_tiles += 1
        if result.latency_budget_exhausted:
            break
    fill = int(np.rint(np.mean(restored[recovered_mask]))) if recovered_mask.any() else 0
    restored[~recovered_mask] = fill
    return restored, {
        "transmitted_bits": float(total_bits),
        "duration_s": total_duration,
        "payload_delivery_ratio": recovered_tiles / len(tiles),
        "latency_budget_exhausted": float(budget_exhausted),
    }


def run_tile_baseline(
    arrays: list[np.ndarray],
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
    model,
    threshold: float,
    min_cells: int,
    iou_threshold: float,
    link,
    tile_size: int,
    tile_header_bits: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    restored: list[np.ndarray] = []
    link_rows: list[dict[str, float]] = []
    for index, image in enumerate(arrays):
        value, link_row = transmit_tiled_image(image, link, tile_size, tile_header_bits, seed + index * 1000)
        restored.append(value)
        link_rows.append(link_row)
    predictions = infer_arrays(restored, model, threshold, min_cells, device)
    metrics = task_metrics(predictions, truths, task, iou_threshold)
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(arrays) - len(arrays) % pressure
    decisions = usable // pressure
    decision_latencies = [
        max(row["duration_s"] for row in link_rows[start : start + pressure])
        for start in range(0, usable, pressure)
    ]
    return {
        "f1": metrics["f1"],
        **{key: metrics[key] for key in ["clean_resource_rate", "mean_occupancy_regret", "mean_selected_occupancy", "oracle_equivalent_rate"]},
        "mean_transmitted_bits": float(sum(row["transmitted_bits"] for row in link_rows[:usable]) / decisions),
        "mean_link_latency_s": float(np.mean(decision_latencies)),
        "mean_payload_delivery_ratio": float(np.mean([row["payload_delivery_ratio"] for row in link_rows[:usable]])),
        "latency_budget_exhaustion_rate": float(np.mean([row["latency_budget_exhausted"] for row in link_rows[:usable]])),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["ebn0_db"]), str(row["scheme"]))].append(row)
    output: list[dict[str, Any]] = []
    for (ebn0_db, scheme), items in sorted(grouped.items()):
        merged: dict[str, Any] = {"ebn0_db": ebn0_db, "scheme": scheme, "repeat_count": len(items)}
        f1_values = [float(item["f1"]) for item in items if item["f1"] is not None]
        if f1_values:
            for stat, value in mean_std_ci(f1_values).items():
                merged[f"f1_{stat}"] = value
        else:
            for stat in ["mean", "std", "ci95_low", "ci95_high"]:
                merged[f"f1_{stat}"] = None
        for metric in RESOURCE_METRICS:
            for stat, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{stat}"] = value
        output.append(merged)
    return output


def make_report(baseline: dict[str, float], summary: list[dict[str, Any]], config: dict[str, Any]) -> str:
    settings = config["classical_baseline_validation"]
    lines = [
        "# Stage 2 Classical Soft-Information and Tile Baselines",
        "",
        "## Protocol",
        "",
        "This experiment adds two missing fair baselines: quantized per-channel PSD/energy reports and independently decodable 16x16 uint8 spectrogram tiles. The frozen detector, source frames, resource task, packetization, CRC, Hamming(7,4), QPSK, AWGN, ARQ, and 0.25 s deadline are unchanged.",
        "",
        f"No-link detector reference: F1={baseline['f1']:.4f}, clean rate={baseline['clean_resource_rate']:.4f}, regret={baseline['mean_occupancy_regret']:.6f}.",
        "",
        "PSD baselines average quantized channel energy from successfully received observations and select the minimum-energy contiguous block. They do not produce boxes, so detection F1 is not applicable. Tile transmission uses a seeded random tile order and a 32-bit tile header; only completely decoded tiles are restored before running the frozen detector.",
        "",
        "## Results",
        "",
        "| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency | Payload/tile delivery |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        f1 = "n/a" if row["f1_mean"] is None else f"{row['f1_mean']:.4f}"
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['scheme']} | {f1} | "
            f"{row['clean_resource_rate_mean']:.4f} [{max(0.0,row['clean_resource_rate_ci95_low']):.4f}, {min(1.0,row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} [{max(0.0,row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {1000*row['mean_link_latency_s_mean']:.2f} ms | "
            f"{row['mean_payload_delivery_ratio_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- PSD energy is derived from RadDet max-hold grayscale spectrograms, not calibrated receiver power in dBm; it is a classical soft-information proxy.",
            "- PSD uses only four candidate-channel values. A later sweep must study frequency resolution and quantizer bits under matched total budgets.",
            "- Independent tiles remove the unfair whole-image gate but add explicit per-tile metadata and require complete tile delivery.",
            "- The 100-frame unrelated-observation pressure subset tests H1/H4 only, not H3 multi-UAV cooperation.",
            "- No test-set threshold is tuned in this experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classical quantized-PSD and independently decodable tile baselines.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "classical_baselines")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    settings = config["classical_baseline_validation"]
    task = {
        **config["task_validation"],
        "split": settings["split"],
        "max_frames": settings["max_frames"],
        "observations_per_pressure_decision": settings["observations_per_pressure_decision"],
    }
    model_config = json.loads(args.mask_result.read_text(encoding="utf-8"))
    threshold = float(model_config["threshold"])
    min_cells = int(model_config["min_cells"])
    iou_threshold = float(model_config["iou_threshold"])
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    frames = iter_raddet_frames(
        args.root, task["split"], read_metadata=False, default_sequence_length=1_000_000,
        max_frames=int(task["max_frames"])
    )
    truths = [truth_boxes(frame) for frame in frames]
    arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in frames]
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()
    predictions = infer_arrays(arrays, model, threshold, min_cells, device)
    baseline = task_metrics(predictions, truths, task, iou_threshold)

    rows: list[dict[str, Any]] = []
    for ebn0_index, ebn0_db in enumerate(settings["ebn0_db_values"]):
        link = replace(build_link_config(config, float(ebn0_db)), channel=str(settings["channel"]))
        for scheme_index, scheme in enumerate(settings["schemes"]):
            for repeat_index in range(int(settings["repeat_count"])):
                seed = int(config["seed"]) + 8_000_000 + ebn0_index * 100_000 + scheme_index * 10_000 + repeat_index * 1_000
                if scheme == "hard_semantic":
                    row = run_repeat(
                        scheme, frames, predictions, truths, model, threshold, min_cells,
                        iou_threshold, task, link, seed, device
                    )
                elif scheme.startswith("quantized_psd_"):
                    bits = int(scheme.removeprefix("quantized_psd_").removesuffix("bit"))
                    row = run_quantized_psd(arrays, truths, task, link, bits, seed)
                elif scheme == "spectrogram_tile16":
                    row = run_tile_baseline(
                        arrays, truths, task, model, threshold, min_cells, iou_threshold, link,
                        int(settings["tile_size"]), int(settings["tile_header_bits"]), seed, device
                    )
                else:
                    raise ValueError(f"unknown classical baseline: {scheme}")
                row.update({"scheme": scheme, "ebn0_db": float(ebn0_db), "seed": seed})
                rows.append(row)
    summary = aggregate(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "classical_baseline_trials.csv", rows)
    write_csv(args.out_dir / "classical_baseline_summary.csv", summary)
    report_path = args.out_dir / "stage2_classical_baseline_report.md"
    report_path.write_text(make_report(baseline, summary, config), encoding="utf-8")
    result = {
        "config_id": config["config_id"],
        "config_sha256": sha256_file(args.config),
        "hypothesis_ids": ["H1", "H4"],
        "data": {
            "split": task["split"],
            "frame_count": len(frames),
            "frame_ids_sha256": sha256_strings(frame.stem for frame in frames),
        },
        "model_checkpoint_sha256": sha256_file(args.mask_checkpoint),
        "environment": environment_snapshot(["numpy", "scipy", "pillow", "torch"]),
        "baseline": baseline,
        "summary": summary,
        "claim_boundary": {
            "calibrated_psd_dbm": False,
            "full_test": False,
            "multi_uav_h3": False,
        },
    }
    (args.out_dir / "stage2_classical_baseline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
