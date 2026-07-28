from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in [PROJECT_DIR / "src", PROJECT_DIR / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.research_scope import assert_valid_experiment_metadata
from evaluate_stage2_task_link import infer_arrays, run_repeat, task_metrics, truth_boxes
from run_stage2_digital_link import build_link_config, mean_std_ci, write_csv


METRICS = [
    "f1",
    "clean_resource_rate",
    "mean_occupancy_regret",
    "mean_transmitted_bits",
    "mean_link_latency_s",
    "mean_payload_delivery_ratio",
    "latency_budget_exhaustion_rate",
]


def aggregate(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        merged: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        merged["repeat_count"] = len(items)
        for metric in METRICS:
            for stat, value in mean_std_ci([float(item[metric]) for item in items]).items():
                merged[f"{metric}_{stat}"] = value
        output.append(merged)
    return output


def run_robustness(
    config: dict[str, Any],
    task: dict[str, Any],
    frames,
    predictions,
    truths,
    model,
    threshold: float,
    min_cells: int,
    iou_threshold: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["stage2_enhancement"]["robustness"]
    rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(settings["channel_models"]):
        for ebn0_index, ebn0_db in enumerate(settings["ebn0_db_values"]):
            base_link = build_link_config(config, float(ebn0_db))
            link = replace(base_link, channel=str(channel))
            for scheme_index, scheme in enumerate(settings["schemes"]):
                for repeat_index in range(int(config["stage2_enhancement"]["repeat_count"])):
                    seed = (
                        int(config["seed"])
                        + channel_index * 1_000_000
                        + ebn0_index * 100_000
                        + scheme_index * 10_000
                        + repeat_index * 1_000
                    )
                    row = run_repeat(
                        str(scheme), frames, predictions, truths, model, threshold, min_cells,
                        iou_threshold, task, link, seed, device
                    )
                    row.update({"channel": channel, "ebn0_db": float(ebn0_db)})
                    rows.append(row)
    return rows, aggregate(rows, ["channel", "ebn0_db", "scheme"])


def run_pareto(
    config: dict[str, Any],
    task: dict[str, Any],
    frames,
    predictions,
    truths,
    model,
    threshold: float,
    min_cells: int,
    iou_threshold: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["stage2_enhancement"]["pareto"]
    rows: list[dict[str, Any]] = []
    for budget_index, budget_s in enumerate(settings["latency_budget_s_values"]):
        base_link = build_link_config(config, float(settings["ebn0_db"]))
        link = replace(base_link, channel=str(settings["channel"]), latency_budget_s=float(budget_s))
        for scheme_index, scheme in enumerate(settings["schemes"]):
            for repeat_index in range(int(settings["repeat_count"])):
                seed = int(config["seed"]) + 5_000_000 + budget_index * 100_000 + scheme_index * 10_000 + repeat_index * 1_000
                row = run_repeat(
                    str(scheme), frames, predictions, truths, model, threshold, min_cells,
                    iou_threshold, task, link, seed, device
                )
                row.update(
                    {
                        "channel": settings["channel"],
                        "ebn0_db": float(settings["ebn0_db"]),
                        "latency_budget_s": float(budget_s),
                    }
                )
                rows.append(row)
    return rows, aggregate(rows, ["latency_budget_s", "scheme"])


def plot_pareto(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    schemes = sorted({str(row["scheme"]) for row in summary})
    for scheme in schemes:
        rows = sorted(
            [row for row in summary if row["scheme"] == scheme],
            key=lambda row: float(row["latency_budget_s"]),
        )
        bits = [float(row["mean_transmitted_bits_mean"]) / 1000.0 for row in rows]
        regret = [float(row["mean_occupancy_regret_mean"]) for row in rows]
        clean = [float(row["clean_resource_rate_mean"]) for row in rows]
        axes[0].plot(bits, regret, marker="o", label=scheme)
        axes[1].plot(bits, clean, marker="o", label=scheme)
        for x, y, row in zip(bits, regret, rows):
            axes[0].annotate(f"{1000*float(row['latency_budget_s']):.0f}ms", (x, y), fontsize=7)
    axes[0].set_xlabel("Mean transmitted bits / decision (kbit)")
    axes[0].set_ylabel("Mean occupancy regret")
    axes[0].set_title("Rate-regret trade-off")
    axes[1].set_xlabel("Mean transmitted bits / decision (kbit)")
    axes[1].set_ylabel("Clean resource rate")
    axes[1].set_title("Rate-clean-rate trade-off")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_report(
    config: dict[str, Any],
    baseline: dict[str, float],
    robustness: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
) -> str:
    lines = [
        "# Stage 2 Enhanced Robustness and Pareto Report",
        "",
        "## Protocol",
        "",
        "The frozen detector, source frames, packetization, CRC, FEC, modulation, ARQ, resource task, and random-seed policy are unchanged. Robustness varies only the reporting channel model and Eb/N0. Pareto experiments vary only the shared latency budget under AWGN at 6 dB.",
        "",
        f"No-link reference: F1={baseline['f1']:.4f}, clean rate={baseline['clean_resource_rate']:.4f}, regret={baseline['mean_occupancy_regret']:.6f}.",
        "",
        "## Reporting-channel robustness",
        "",
        "| Channel | Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Payload delivered |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in robustness:
        lines.append(
            f"| {row['channel']} | {float(row['ebn0_db']):.1f} | {row['scheme']} | {row['f1_mean']:.4f} | "
            f"{row['clean_resource_rate_mean']:.4f} [{max(0.0,row['clean_resource_rate_ci95_low']):.4f}, {min(1.0,row['clean_resource_rate_ci95_high']):.4f}] | "
            f"{row['mean_occupancy_regret_mean']:.6f} [{max(0.0,row['mean_occupancy_regret_ci95_low']):.6f}, {row['mean_occupancy_regret_ci95_high']:.6f}] | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {row['mean_payload_delivery_ratio_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shared-latency-budget Pareto sweep",
            "",
            "| Budget | Scheme | F1 | Clean rate | Regret | bit/decision | Link latency | Budget exhaustion |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pareto:
        lines.append(
            f"| {1000*float(row['latency_budget_s']):.0f} ms | {row['scheme']} | {row['f1_mean']:.4f} | "
            f"{row['clean_resource_rate_mean']:.4f} | {row['mean_occupancy_regret_mean']:.6f} | "
            f"{row['mean_transmitted_bits_mean']:.1f} | {1000*row['mean_link_latency_s_mean']:.2f} ms | "
            f"{row['latency_budget_exhaustion_rate_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- The 100-frame subset is a diagnostic enhancement experiment, not the final full-test thesis table.",
            "- Rayleigh/Rician are independent block-fading reporting channels with perfect receiver-side equalization; burst errors and channel-estimation error remain future work.",
            "- Unrelated four-observation pressure aggregation tests H1/H4 only and does not establish H3 spatial cooperation.",
            "- Pareto points are produced by changing a shared deadline, not by granting method-specific coding or retransmission rules.",
            "- PNG and row-major spectrogram fragmentation represent explicit codec/recovery baselines; future tile and erasure-aware baselines are still required.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-2 reporting-channel robustness and rate-task Pareto sweeps.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "enhanced")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_experiment_metadata(config)
    enhancement = config["stage2_enhancement"]
    task = {
        **config["task_validation"],
        "split": enhancement["split"],
        "max_frames": enhancement["max_frames"],
        "observations_per_pressure_decision": enhancement["observations_per_pressure_decision"],
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

    robustness_rows, robustness_summary = run_robustness(
        config, task, frames, predictions, truths, model, threshold, min_cells, iou_threshold, device
    )
    pareto_rows, pareto_summary = run_pareto(
        config, task, frames, predictions, truths, model, threshold, min_cells, iou_threshold, device
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "robustness_trials.csv", robustness_rows)
    write_csv(args.out_dir / "robustness_summary.csv", robustness_summary)
    write_csv(args.out_dir / "pareto_trials.csv", pareto_rows)
    write_csv(args.out_dir / "pareto_summary.csv", pareto_summary)
    plot_pareto(pareto_summary, args.out_dir / "stage2_rate_task_pareto.png")
    report_path = args.out_dir / "stage2_enhanced_report.md"
    report_path.write_text(make_report(config, baseline, robustness_summary, pareto_summary), encoding="utf-8")
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
        "environment": environment_snapshot(["numpy", "scipy", "matplotlib", "pillow", "torch"]),
        "baseline": baseline,
        "robustness_summary": robustness_summary,
        "pareto_summary": pareto_summary,
        "claim_boundary": {"full_test": False, "multi_uav_h3": False, "burst_error_model": False},
    }
    (args.out_dir / "stage2_enhanced_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
