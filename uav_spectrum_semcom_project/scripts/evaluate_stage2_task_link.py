from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.bit_budget import hard_box_packet_bits, packet_header_bits, soft_box_packet_bits
from spectrum_semcom.digital_link import transmit_payload_analytic
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import RadDetFrame, iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv
from simulate_resource_optimization import all_channel_occupancy, block_occupancy
from simulate_semantic_link import evaluate_box_lists
from train_raddet_occupancy_mask import mask_to_boxes, yolo_to_xyxy


def truth_boxes(frame: RadDetFrame) -> list[np.ndarray]:
    return [
        yolo_to_xyxy(np.asarray([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
        for box in frame.boxes
    ]


def normalize_image(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32) / 255.0
    return (value - value.mean()) / (value.std() + 1e-6)


def infer_arrays(
    arrays: list[np.ndarray],
    model,
    threshold: float,
    min_cells: int,
    device: str,
    batch_size: int = 32,
) -> list[list[np.ndarray]]:
    predictions, _ = infer_arrays_with_confidence(arrays, model, threshold, min_cells, device, batch_size)
    return predictions


def infer_arrays_with_confidence(
    arrays: list[np.ndarray],
    model,
    threshold: float,
    min_cells: int,
    device: str,
    batch_size: int = 32,
) -> tuple[list[list[np.ndarray]], list[float]]:
    """Infer boxes and an auditable label-free mask confidence per frame.

    The raw confidence is the mean foreground probability when foreground is
    detected, otherwise the mean background probability.  Stage 3 calibrates
    this raw score against occupancy-report quality on validation scenes only.
    """
    import torch

    predictions: list[list[np.ndarray]] = []
    confidences: list[float] = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            batch = np.stack([normalize_image(value) for value in arrays[start : start + batch_size]])[:, None, :, :]
            probabilities = torch.sigmoid(model(torch.tensor(batch, dtype=torch.float32, device=device)))[:, 0]
            for probability in probabilities.detach().cpu().numpy():
                binary = probability >= threshold
                predictions.append(mask_to_boxes(binary, min_cells=min_cells))
                if np.any(binary):
                    confidences.append(float(np.mean(probability[binary])))
                else:
                    confidences.append(float(np.mean(1.0 - probability)))
    return predictions, confidences


def recovered_packet_indices(result) -> set[int]:
    return {trace.packet_index for trace in result.packet_traces if trace.delivered}


def recover_semantic_boxes(
    boxes: list[np.ndarray],
    delivered_packets: set[int],
    packet_payload_bits: int,
    per_box_bits: int,
) -> list[np.ndarray]:
    application_header = packet_header_bits()
    header_last_packet = max(0, (application_header - 1) // packet_payload_bits)
    if not all(index in delivered_packets for index in range(header_last_packet + 1)):
        return []
    recovered: list[np.ndarray] = []
    for index, box in enumerate(boxes):
        start = application_header + index * per_box_bits
        end = start + per_box_bits
        first_packet = start // packet_payload_bits
        last_packet = (end - 1) // packet_payload_bits
        if all(packet in delivered_packets for packet in range(first_packet, last_packet + 1)):
            recovered.append(box)
    return recovered


def conceal_fragmented_spectrogram(frame: RadDetFrame, result) -> np.ndarray:
    original = np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8)
    flat = original.reshape(-1)
    recovered = np.zeros(len(flat), dtype=bool)
    packet_payload_bytes = result.packet_traces[0].application_bits // 8 if result.packet_traces else 0
    for trace in result.packet_traces:
        if not trace.delivered:
            continue
        start = trace.packet_index * packet_payload_bytes
        stop = min(start + trace.application_bits // 8, len(flat))
        recovered[start:stop] = True
    if recovered.any():
        fill = int(np.rint(np.mean(flat[recovered])))
    else:
        fill = 0
    output = np.full_like(flat, fill)
    output[recovered] = flat[recovered]
    return output.reshape(original.shape)


def task_metrics(
    predictions: list[list[np.ndarray]],
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
    iou_threshold: float,
) -> dict[str, float]:
    detection = evaluate_box_lists(predictions, truths, iou_threshold)
    pressure = int(task.get("observations_per_pressure_decision", 1))
    usable = len(truths) - len(truths) % pressure
    grouped_predictions = [
        [box for frame in predictions[start : start + pressure] for box in frame]
        for start in range(0, usable, pressure)
    ]
    grouped_truths = [
        [box for frame in truths[start : start + pressure] for box in frame]
        for start in range(0, usable, pressure)
    ]
    truth_channels = all_channel_occupancy(grouped_truths, int(task["n_channels"]), str(task["channel_axis"]))
    truth_resources = block_occupancy(truth_channels, int(task["demand_channels"]))
    sensed_channels = all_channel_occupancy(grouped_predictions, int(task["n_channels"]), str(task["channel_axis"]))
    sensed_resources = block_occupancy(sensed_channels, int(task["demand_channels"]))
    chosen = np.argmin(sensed_resources, axis=1)
    oracle = np.argmin(truth_resources, axis=1)
    indices = np.arange(len(grouped_truths))
    chosen_occupancy = truth_resources[indices, chosen]
    oracle_occupancy = truth_resources[indices, oracle]
    return {
        **detection,
        "clean_resource_rate": float(np.mean(chosen_occupancy <= float(task["clean_threshold"]))),
        "mean_occupancy_regret": float(np.mean(chosen_occupancy - oracle_occupancy)),
        "mean_selected_occupancy": float(np.mean(chosen_occupancy)),
        "oracle_equivalent_rate": float(np.mean(np.abs(chosen_occupancy - oracle_occupancy) <= 1e-9)),
    }


def run_repeat(
    scheme: str,
    frames: list[RadDetFrame],
    original_predictions: list[list[np.ndarray]],
    truths: list[list[np.ndarray]],
    model,
    threshold: float,
    min_cells: int,
    iou_threshold: float,
    task: dict[str, Any],
    link,
    seed: int,
    device: str,
) -> dict[str, float | int | str]:
    delivered_predictions: list[list[np.ndarray]] = []
    fragmented_arrays: list[np.ndarray] = []
    total_tx_bits = 0
    total_duration = 0.0
    total_delivered_ratio = 0.0
    budget_exhausted = 0
    packet_payload = link.packet.payload_bits
    per_observation_durations: list[float] = []

    for index, (frame, boxes) in enumerate(zip(frames, original_predictions)):
        frame_seed = seed + index
        if scheme == "hard_semantic":
            application_bits = hard_box_packet_bits(len(boxes))
            result = transmit_payload_analytic(application_bits, link, frame_seed)
            delivered_predictions.append(
                recover_semantic_boxes(boxes, recovered_packet_indices(result), packet_payload, 8 + 4 * 16)
            )
        elif scheme == "quantized_soft_semantic":
            application_bits = soft_box_packet_bits(len(boxes), n_classes=11)
            result = transmit_payload_analytic(application_bits, link, frame_seed)
            delivered_predictions.append(
                recover_semantic_boxes(boxes, recovered_packet_indices(result), packet_payload, 4 * 16 + 8 + 11 * 8)
            )
        elif scheme == "compressed_spectrogram_png":
            application_bits = frame.image_compressed_bits
            result = transmit_payload_analytic(application_bits, link, frame_seed)
            # PNG is not independently decodable from arbitrary missing fragments.
            delivered_predictions.append(boxes if result.frame_success else [])
        elif scheme == "spectrogram_uint8_fragment":
            application_bits = 128 * 128 * 8
            result = transmit_payload_analytic(application_bits, link, frame_seed)
            fragmented_arrays.append(conceal_fragmented_spectrogram(frame, result))
        else:
            raise ValueError(f"unknown task-validation scheme: {scheme}")
        total_tx_bits += result.transmitted_bits
        total_duration += result.duration_s
        per_observation_durations.append(result.duration_s)
        total_delivered_ratio += result.payload_delivery_ratio
        budget_exhausted += int(result.latency_budget_exhausted)

    if scheme == "spectrogram_uint8_fragment":
        delivered_predictions = infer_arrays(fragmented_arrays, model, threshold, min_cells, device)
    metrics = task_metrics(delivered_predictions, truths, task, iou_threshold)
    n = len(frames)
    pressure = int(task.get("observations_per_pressure_decision", 1))
    usable = n - n % pressure
    decision_count = max(1, usable // pressure)
    parallel_decision_latencies = [
        max(per_observation_durations[start : start + pressure])
        for start in range(0, usable, pressure)
    ]
    return {
        "scheme": scheme,
        "seed": seed,
        "frames": n,
        **metrics,
        "mean_transmitted_bits": total_tx_bits / decision_count,
        "mean_link_latency_s": float(np.mean(parallel_decision_latencies)),
        "mean_payload_delivery_ratio": total_delivered_ratio / n,
        "latency_budget_exhaustion_rate": budget_exhausted / n,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scheme"]), float(row["ebn0_db"]))].append(row)
    metrics = [
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "clean_resource_rate",
        "mean_occupancy_regret",
        "mean_selected_occupancy",
        "oracle_equivalent_rate",
        "mean_transmitted_bits",
        "mean_link_latency_s",
        "mean_payload_delivery_ratio",
        "latency_budget_exhaustion_rate",
    ]
    output: list[dict[str, Any]] = []
    for (scheme, ebn0_db), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        row: dict[str, Any] = {"scheme": scheme, "ebn0_db": ebn0_db, "repeat_count": len(items)}
        for metric in metrics:
            for name, value in mean_std_ci([float(item[metric]) for item in items]).items():
                row[f"{metric}_{name}"] = value
        output.append(row)
    return output


def make_report(config: dict[str, Any], baseline: dict[str, float], summary: list[dict[str, Any]]) -> str:
    task = config["task_validation"]
    lines = [
        "# Stage 2 Task-Level Digital Link Validation",
        "",
        "## Scope",
        "",
        "This experiment connects the fair digital link to the frozen RadDet occupancy detector and a controlled resource-pressure task. All schemes share packetization, CRC, Hamming(7,4), QPSK, AWGN, ARQ, symbol rate, and the 0.25 s decision deadline.",
        "",
        "The uint8 spectrogram baseline uses independently delivered row-major fragments; missing bytes are concealed with the mean of received bytes before the frozen detector runs. PNG requires complete delivery because arbitrary PNG fragments are not independently decodable. Raw IQ is excluded from task metrics because the local RadDet copy has no raw IQ samples.",
        "",
        "## No-link frozen detector reference",
        "",
        f"- Frames: {task['max_frames']} from frozen `{task['split']}` split.",
        f"- Pressure decisions: {task['max_frames'] // task['observations_per_pressure_decision']}, each formed from {task['observations_per_pressure_decision']} unrelated observations; this is not spatial cooperation evidence.",
        f"- Detection F1: {baseline['f1']:.4f}; precision: {baseline['precision']:.4f}; recall: {baseline['recall']:.4f}.",
        f"- Clean resource rate: {baseline['clean_resource_rate']:.4f}; mean occupancy regret: {baseline['mean_occupancy_regret']:.6f}.",
        "",
        "## Fair-link task results",
        "",
        "| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | Tx bits/decision | Parallel-link latency (ms) | Delivered payload |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['scheme']} | {row['f1_mean']:.4f} | "
            f"{row['clean_resource_rate_mean']:.4f} [{max(0.0, row['clean_resource_rate_ci95_low']):.4f}, {min(1.0, row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} [{max(0.0, row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {1000 * row['mean_link_latency_s_mean']:.2f} | "
            f"{row['mean_payload_delivery_ratio_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- This is a controlled H4/H1 pressure validation. Aggregated observations are unrelated and therefore provide no evidence for multi-UAV spatial cooperation H3.",
            "- The detector and threshold are frozen; no test-set tuning is performed.",
            "- Spectrogram fragment concealment is an explicit baseline choice and must be ablated against tiling/erasure-aware training before the final thesis.",
            "- The reported confidence intervals quantify link Monte Carlo variation on the fixed frame subset; a final thesis experiment must repeat across a larger frozen test set and additional channel models.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate task performance through the stage-2 fair digital link.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "task_link")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    task = config["task_validation"]
    model_config = json.loads(args.mask_result.read_text(encoding="utf-8"))
    threshold = float(model_config["threshold"])
    min_cells = int(model_config["min_cells"])
    iou_threshold = float(model_config["iou_threshold"])
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    frames = iter_raddet_frames(
        args.root,
        str(task["split"]),
        read_metadata=False,
        default_sequence_length=1_000_000,
        max_frames=int(task["max_frames"]),
    )
    truths = [truth_boxes(frame) for frame in frames]
    original_arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in frames]
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()
    original_predictions = infer_arrays(original_arrays, model, threshold, min_cells, device)
    baseline = task_metrics(original_predictions, truths, task, iou_threshold)

    rows: list[dict[str, Any]] = []
    for ebn0_index, ebn0_db in enumerate(task["ebn0_db_values"]):
        link = build_link_config(config, float(ebn0_db))
        for scheme_index, scheme in enumerate(task["schemes"]):
            for repeat in range(int(task["repeat_count"])):
                seed = int(config["seed"]) + ebn0_index * 100_000 + scheme_index * 10_000 + repeat * 1_000
                row = run_repeat(
                    str(scheme),
                    frames,
                    original_predictions,
                    truths,
                    model,
                    threshold,
                    min_cells,
                    iou_threshold,
                    task,
                    link,
                    seed,
                    device,
                )
                row["ebn0_db"] = float(ebn0_db)
                rows.append(row)
    summary = aggregate(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "task_link_trials.csv", rows)
    write_csv(args.out_dir / "task_link_summary.csv", summary)
    (args.out_dir / "stage2_task_link_report.md").write_text(make_report(config, baseline, summary), encoding="utf-8")
    result = {
        "config_id": config["config_id"],
        "config_sha256": sha256_file(args.config),
        "hypothesis_ids": ["H1", "H4"],
        "data": {
            "split": task["split"],
            "frame_count": len(frames),
            "frame_ids_sha256": sha256_strings(frame.stem for frame in frames),
            "dataset_registry_sha256": sha256_file(PROJECT_DIR / "configs" / "dataset_registry.json"),
        },
        "model": {
            "checkpoint": args.mask_checkpoint.relative_to(PROJECT_DIR).as_posix(),
            "checkpoint_sha256": sha256_file(args.mask_checkpoint),
            "result_config_sha256": sha256_file(args.mask_result),
        },
        "environment": environment_snapshot(["numpy", "scipy", "pillow", "torch", "pytest"]),
        "baseline": baseline,
        "summary": summary,
        "artifacts": {
            "trials_csv": "results/stage2/task_link/task_link_trials.csv",
            "summary_csv": "results/stage2/task_link/task_link_summary.csv",
            "report": "results/stage2/task_link/stage2_task_link_report.md",
        },
        "claim_boundary": {
            "multi_uav_h3": False,
            "raw_iq_task_baseline": False,
            "test_tuning": False,
        },
    }
    (args.out_dir / "stage2_task_link_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'stage2_task_link_report.md'}")


if __name__ == "__main__":
    main()
