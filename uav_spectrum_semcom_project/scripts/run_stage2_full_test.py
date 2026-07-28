from __future__ import annotations

import argparse
import json
import sys
import time
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
from evaluate_stage2_classical_baselines import (
    channel_energy,
    evaluate_resource_estimates,
    quantize_unit_interval,
)
from evaluate_stage2_task_link import infer_arrays, run_repeat, task_metrics, truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv


def load_or_create_features(
    frames,
    model,
    threshold: float,
    min_cells: int,
    device: str,
    batch_size: int,
    cache_path: Path,
    use_cache: bool,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    frame_hash = sha256_strings(frame.stem for frame in frames)
    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("frame_ids_sha256") == frame_hash and len(data.get("predictions", [])) == len(frames):
            predictions = [[np.asarray(box, dtype=np.float32) for box in item] for item in data["predictions"]]
            energies = np.asarray(data["channel_energies"], dtype=np.float32)
            print(f"loaded frozen feature cache for {len(frames)} frames", flush=True)
            return predictions, energies

    predictions: list[list[np.ndarray]] = []
    energy_rows: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in batch_frames]
        predictions.extend(infer_arrays(arrays, model, threshold, min_cells, device, batch_size=batch_size))
        energy_rows.extend(channel_energy(array, 4, "y") for array in arrays)
        if start == 0 or (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(frames):
            print(f"feature extraction {min(start + batch_size, len(frames))}/{len(frames)}", flush=True)
    energies = np.stack(energy_rows)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frame_ids_sha256": frame_hash,
            "predictions": [[box.tolist() for box in item] for item in predictions],
            "channel_energies": energies.tolist(),
            "duration_s": time.perf_counter() - started,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {cache_path}", flush=True)
    return predictions, energies


def run_psd_from_energies(
    energies: np.ndarray,
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
    link,
    quantization_bits: int,
    seed: int,
) -> dict[str, Any]:
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(energies) - len(energies) % pressure
    quantized = [quantize_unit_interval(row, quantization_bits) for row in energies[:usable]]
    application_bits = packet_header_bits() + int(task["n_channels"]) * quantization_bits
    received: list[np.ndarray | None] = []
    bits: list[int] = []
    durations: list[float] = []
    ratios: list[float] = []
    exhausted: list[float] = []
    for index, value in enumerate(quantized):
        result = transmit_payload_analytic(application_bits, link, seed + index)
        received.append(value if result.frame_success else None)
        bits.append(result.transmitted_bits)
        durations.append(result.duration_s)
        ratios.append(result.payload_delivery_ratio)
        exhausted.append(float(result.latency_budget_exhausted))
    estimates: list[np.ndarray] = []
    decision_latencies: list[float] = []
    for start in range(0, usable, pressure):
        available = [value for value in received[start : start + pressure] if value is not None]
        estimates.append(
            np.mean(available, axis=0) if available else np.zeros(int(task["n_channels"]), dtype=np.float32)
        )
        decision_latencies.append(max(durations[start : start + pressure]))
    metrics = evaluate_resource_estimates(np.stack(estimates), truths[:usable], task)
    decisions = usable // pressure
    return {
        "f1": None,
        **metrics,
        "mean_transmitted_bits": float(np.sum(bits) / decisions),
        "mean_link_latency_s": float(np.mean(decision_latencies)),
        "mean_payload_delivery_ratio": float(np.mean(ratios)),
        "latency_budget_exhaustion_rate": float(np.mean(exhausted)),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        "clean_resource_rate",
        "mean_occupancy_regret",
        "mean_selected_occupancy",
        "oracle_equivalent_rate",
        "mean_transmitted_bits",
        "mean_link_latency_s",
        "mean_payload_delivery_ratio",
        "latency_budget_exhaustion_rate",
    ]
    for ebn0_db in sorted({float(row["ebn0_db"]) for row in rows}):
        for scheme in sorted({str(row["scheme"]) for row in rows}):
            items = [
                row for row in rows
                if float(row["ebn0_db"]) == ebn0_db and str(row["scheme"]) == scheme
            ]
            if not items:
                continue
            merged: dict[str, Any] = {"ebn0_db": ebn0_db, "scheme": scheme, "repeat_count": len(items)}
            f1_values = [float(item["f1"]) for item in items if item["f1"] is not None]
            merged["f1_mean"] = float(np.mean(f1_values)) if f1_values else None
            for metric in metrics:
                for stat, value in mean_std_ci([float(item[metric]) for item in items]).items():
                    merged[f"{metric}_{stat}"] = value
            output.append(merged)
    return output


def make_report(
    settings: dict[str, Any],
    selected_psd_bits: int,
    frame_count: int,
    baseline: dict[str, float],
    summary: list[dict[str, Any]],
) -> str:
    usable = frame_count - frame_count % int(settings["observations_per_pressure_decision"])
    lines = [
        "# Stage 2 Full Frozen Test-Split Validation",
        "",
        "## Protocol",
        "",
        f"All {frame_count} frames in the frozen local `{settings['split']}` directory are used; {usable} frames form {usable // settings['observations_per_pressure_decision']} complete four-observation pressure decisions and the remainder is excluded deterministically.",
        "",
        f"The PSD quantizer is frozen at {selected_psd_bits} bit/value from the independent validation-split selection. The test split does not select detector thresholds, quantizer resolution, FEC, ARQ, packet size, channel conditions, or latency budget.",
        "",
        f"No-link detector reference: F1={baseline['f1']:.4f}, clean rate={baseline['clean_resource_rate']:.4f}, regret={baseline['mean_occupancy_regret']:.6f}.",
        "",
        "## Full-split fair-link results",
        "",
        "| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency | Payload delivered |",
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
            "## Claim boundary",
            "",
            "- The local directory named `test` contains 20,001 frames. Stage 1 froze this observed mapping but did not verify the upstream provenance of the nonstandard train/val/test counts; the directory is therefore not relabeled.",
            "- Four unrelated frames form a resource-pressure decision. This is full-split H1/H4 evidence, not H3 spatially correlated multi-UAV evidence.",
            "- RadDet is synthetic wideband radar data, not real UAV RF capture.",
            "- PSD is normalized grayscale energy rather than calibrated dBm power.",
            "- Hamming(7,4) is an auditable coding baseline rather than a 5G-grade channel code.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-2 main comparison on the complete frozen test directory.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "full_test")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    settings = config["full_test_validation"]
    task = {
        **config["task_validation"],
        "split": settings["split"],
        "observations_per_pressure_decision": settings["observations_per_pressure_decision"],
    }
    psd_result_path = PROJECT_DIR / settings["psd_selection_result"]
    psd_selection = json.loads(psd_result_path.read_text(encoding="utf-8"))
    selected_psd_bits = int(psd_selection["selected_bits"])
    frames = iter_raddet_frames(
        args.root, settings["split"], read_metadata=False, default_sequence_length=1_000_000,
        max_frames=None if settings["use_all_frames"] else int(settings["expected_frame_count"])
    )
    if len(frames) != int(settings["expected_frame_count"]):
        raise ValueError(f"frozen split count changed: expected {settings['expected_frame_count']}, found {len(frames)}")
    truths = [truth_boxes(frame) for frame in frames]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_config = json.loads(args.mask_result.read_text(encoding="utf-8"))
    threshold = float(model_config["threshold"])
    min_cells = int(model_config["min_cells"])
    iou_threshold = float(model_config["iou_threshold"])
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()
    cache_path = args.out_dir / "frozen_feature_cache.json"
    predictions, energies = load_or_create_features(
        frames, model, threshold, min_cells, device, int(settings["prediction_batch_size"]),
        cache_path, bool(settings["cache_predictions"])
    )
    baseline = task_metrics(predictions, truths, task, iou_threshold)

    rows: list[dict[str, Any]] = []
    for ebn0_index, ebn0_db in enumerate(settings["ebn0_db_values"]):
        link = build_link_config(config, float(ebn0_db))
        for scheme_index, scheme in enumerate(settings["schemes"]):
            for repeat_index in range(int(settings["repeat_count"])):
                seed = int(config["seed"]) + 20_000_000 + ebn0_index * 100_000 + scheme_index * 10_000 + repeat_index * 1_000
                print(f"running {scheme}, Eb/N0={ebn0_db}, repeat={repeat_index + 1}/{settings['repeat_count']}", flush=True)
                if scheme == "hard_semantic":
                    row = run_repeat(
                        scheme, frames, predictions, truths, None, threshold, min_cells,
                        iou_threshold, task, link, seed, device
                    )
                elif scheme == "frozen_quantized_psd":
                    row = run_psd_from_energies(energies, truths, task, link, selected_psd_bits, seed)
                else:
                    raise ValueError(f"unknown full-test scheme: {scheme}")
                row.update({"scheme": scheme, "ebn0_db": float(ebn0_db), "seed": seed})
                rows.append(row)
    summary = aggregate(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "full_test_trials.csv", rows)
    write_csv(args.out_dir / "full_test_summary.csv", summary)
    report_path = args.out_dir / "stage2_full_test_report.md"
    report_path.write_text(
        make_report(settings, selected_psd_bits, len(frames), baseline, summary), encoding="utf-8"
    )
    result = {
        "config_id": config["config_id"],
        "config_sha256": sha256_file(args.config),
        "hypothesis_ids": ["H1", "H4"],
        "selected_psd_bits": selected_psd_bits,
        "psd_selection_result_sha256": sha256_file(psd_result_path),
        "data": {
            "split": settings["split"],
            "frame_count": len(frames),
            "frame_ids_sha256": sha256_strings(frame.stem for frame in frames),
        },
        "model_checkpoint_sha256": sha256_file(args.mask_checkpoint),
        "environment": environment_snapshot(["numpy", "scipy", "pillow", "torch"]),
        "baseline": baseline,
        "summary": summary,
        "claim_boundary": {
            "complete_frozen_local_test_directory": True,
            "upstream_split_provenance_verified": False,
            "multi_uav_h3": False,
            "real_uav_rf": False,
        },
    }
    (args.out_dir / "stage2_full_test_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
