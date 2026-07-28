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

from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_classical_baselines import (
    channel_energy,
    evaluate_resource_estimates,
    quantize_unit_interval,
    run_quantized_psd,
)
from evaluate_stage2_task_link import truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv


def load_split(root: Path, split: str, max_frames: int):
    frames = iter_raddet_frames(
        root, split, read_metadata=False, default_sequence_length=1_000_000, max_frames=max_frames
    )
    arrays = [np.asarray(Image.open(frame.image_path).convert("L"), dtype=np.uint8) for frame in frames]
    truths = [truth_boxes(frame) for frame in frames]
    return frames, arrays, truths


def direct_psd_metrics(
    arrays: list[np.ndarray],
    truths: list[list[np.ndarray]],
    task: dict[str, Any],
    bits: int,
) -> dict[str, float]:
    pressure = int(task["observations_per_pressure_decision"])
    usable = len(arrays) - len(arrays) % pressure
    node_values = [
        quantize_unit_interval(
            channel_energy(image, int(task["n_channels"]), str(task["channel_axis"])), bits
        )
        for image in arrays[:usable]
    ]
    grouped = np.stack(
        [np.mean(node_values[start : start + pressure], axis=0) for start in range(0, usable, pressure)]
    )
    return evaluate_resource_estimates(grouped, truths[:usable], task)


def aggregate_link_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ebn0_db in sorted({float(row["ebn0_db"]) for row in rows}):
        items = [row for row in rows if float(row["ebn0_db"]) == ebn0_db]
        merged: dict[str, Any] = {"ebn0_db": ebn0_db, "repeat_count": len(items)}
        for metric in [
            "clean_resource_rate",
            "mean_occupancy_regret",
            "mean_transmitted_bits",
            "mean_link_latency_s",
            "mean_payload_delivery_ratio",
        ]:
            for stat, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{stat}"] = value
        output.append(merged)
    return output


def make_report(
    config: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    selected_bits: int,
    test_direct: dict[str, float],
    link_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Stage 2 PSD Quantizer Selection",
        "",
        "## Selection protocol",
        "",
        "Quantizer resolution is selected only on the frozen validation split. The rule is lexicographic: minimum occupancy regret, then maximum clean rate, then minimum bit width. The selected width is subsequently frozen and evaluated once on the test split and through the fair digital reporting link.",
        "",
        "The PSD values are four normalized mean grayscale energies from RadDet max-hold spectrograms. They are a soft-information proxy, not calibrated receiver power in dBm.",
        "",
        "## Validation selection",
        "",
        "| Quantizer bits/value | Clean rate | Regret |",
        "|---:|---:|---:|",
    ]
    for row in selection_rows:
        lines.append(f"| {row['bits']} | {row['clean_resource_rate']:.4f} | {row['mean_occupancy_regret']:.6f} |")
    lines.extend(
        [
            "",
            f"Selected resolution: **{selected_bits} bit/value**.",
            "",
            "## Frozen test result without reporting errors",
            "",
            f"- Clean rate: {test_direct['clean_resource_rate']:.4f}.",
            f"- Mean occupancy regret: {test_direct['mean_occupancy_regret']:.6f}.",
            "",
            "## Frozen quantizer through the digital link",
            "",
            "| Eb/N0 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Link latency | Payload delivered |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in link_summary:
        lines.append(
            f"| {row['ebn0_db']:.1f} | {row['clean_resource_rate_mean']:.4f} "
            f"[{max(0.0,row['clean_resource_rate_ci95_low']):.4f}, {min(1.0,row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} "
            f"[{max(0.0,row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {1000*row['mean_link_latency_s_mean']:.2f} ms | "
            f"{row['mean_payload_delivery_ratio_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Candidate bit widths and the selection rule were declared before this run; test results do not alter the selected width.",
            "- The 100-frame validation and test subsets are diagnostic. Final thesis evidence requires the complete frozen splits.",
            "- The quantizer comparison isolates representation resolution; it does not tune energy thresholds or frequency aggregation on the test set.",
            "- Selection on grayscale energy may not transfer to calibrated IQ/PSD measurements and must be repeated when real receiver power is available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PSD quantization only on validation data, then freeze it for test.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "psd_quantizer")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    settings = config["psd_quantizer_selection"]
    task = {
        **config["task_validation"],
        "observations_per_pressure_decision": config["classical_baseline_validation"]["observations_per_pressure_decision"],
    }
    val_frames, val_arrays, val_truths = load_split(
        args.root, settings["selection_split"], int(settings["max_frames_per_split"])
    )
    test_frames, test_arrays, test_truths = load_split(
        args.root, settings["test_split"], int(settings["max_frames_per_split"])
    )
    selection_rows = []
    for bits in settings["candidate_bits"]:
        selection_rows.append({"bits": int(bits), **direct_psd_metrics(val_arrays, val_truths, task, int(bits))})
    selected = min(
        selection_rows,
        key=lambda row: (row["mean_occupancy_regret"], -row["clean_resource_rate"], row["bits"]),
    )
    selected_bits = int(selected["bits"])
    test_direct = direct_psd_metrics(test_arrays, test_truths, task, selected_bits)

    link_rows: list[dict[str, Any]] = []
    for ebn0_index, ebn0_db in enumerate(settings["reporting_channel_ebn0_db_values"]):
        link = build_link_config(config, float(ebn0_db))
        for repeat_index in range(int(settings["repeat_count"])):
            seed = int(config["seed"]) + 12_000_000 + ebn0_index * 100_000 + repeat_index * 1_000
            row = run_quantized_psd(test_arrays, test_truths, task, link, selected_bits, seed)
            row.update({"ebn0_db": float(ebn0_db), "seed": seed, "selected_bits": selected_bits})
            link_rows.append(row)
    link_summary = aggregate_link_rows(link_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "psd_quantizer_selection.csv", selection_rows)
    write_csv(args.out_dir / "psd_quantizer_link_trials.csv", link_rows)
    write_csv(args.out_dir / "psd_quantizer_link_summary.csv", link_summary)
    report_path = args.out_dir / "stage2_psd_quantizer_report.md"
    report_path.write_text(
        make_report(config, selection_rows, selected_bits, test_direct, link_summary), encoding="utf-8"
    )
    result = {
        "config_id": config["config_id"],
        "config_sha256": sha256_file(args.config),
        "selection_rule": settings["selection_rule"],
        "selected_bits": selected_bits,
        "selection_rows": selection_rows,
        "test_direct": test_direct,
        "link_summary": link_summary,
        "data": {
            "selection_split": settings["selection_split"],
            "selection_frame_ids_sha256": sha256_strings(frame.stem for frame in val_frames),
            "test_split": settings["test_split"],
            "test_frame_ids_sha256": sha256_strings(frame.stem for frame in test_frames),
        },
        "environment": environment_snapshot(["numpy", "scipy", "pillow"]),
        "claim_boundary": {"full_split": False, "calibrated_psd_dbm": False, "test_tuning": False},
    }
    (args.out_dir / "stage2_psd_quantizer_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"selected {selected_bits} bit/value")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
