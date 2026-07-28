"""Audit Gate A resource-task saturation without accessing a final holdout."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectrum_semcom.raddet import RadDetFrame, iter_raddet_frames
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file, sha256_strings
from spectrum_semcom.resource_losses import resource_task_diagnostics
from spectrum_semcom.resource_task_audit import select_non_saturated_configuration
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol
from simulate_resource_optimization import frame_channel_occupancy


def normalized_boxes(frames: list[RadDetFrame]) -> list[np.ndarray]:
    boxes: list[np.ndarray] = []
    for frame in frames:
        for box in frame.boxes:
            boxes.append(
                np.asarray(
                    [
                        box.x_center - box.width / 2,
                        box.y_center - box.height / 2,
                        box.x_center + box.width / 2,
                        box.y_center + box.height / 2,
                    ],
                    dtype=np.float32,
                )
            )
    return boxes


def grouped_frames(frames: list[RadDetFrame], sources_per_scene: int, max_scenes: int) -> list[list[RadDetFrame]]:
    count = min(int(max_scenes), len(frames) // int(sources_per_scene))
    return [frames[index * sources_per_scene : (index + 1) * sources_per_scene] for index in range(count)]


def make_report(rows: list[dict], selection: dict | None, settings: dict) -> str:
    lines = [
        "# Stage 4 Gate A Resource-Task Difficulty Audit",
        "",
        "This is a train/validation-only H2 pressure-task diagnostic. It does not use or create the final holdout and cannot prove H3.",
        "",
        "| Split role | Sources/scene | Channels | Demand | Scenes | Oracle clean | Random-oracle gap | Oracle margin | Ambiguous rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split_role']} | {row['sources_per_scene']} | {row['n_channels']} | {row['demand_channels']} | "
            f"{row['scene_count']} | {row['oracle_clean_rate']:.4f} | {row['mean_random_oracle_gap']:.6f} | "
            f"{row['mean_oracle_margin']:.6f} | {row['ambiguous_oracle_rate']:.4f} |"
        )
    lines.extend(["", "## Preregistered selection", ""])
    if selection is None:
        lines.append("No candidate met the frozen non-saturation filters; Gate A training must not start on this task definition.")
    else:
        lines.append(
            f"Validation selected sources/scene={selection['sources_per_scene']}, channels={selection['n_channels']}, "
            f"demand={selection['demand_channels']}, random-oracle gap={selection['mean_random_oracle_gap']:.6f}."
        )
    lines.extend(
        [
            "",
            "Selection uses truth-only task geometry, never detector predictions or proposed-method performance.",
            f"Rule: {settings['selection_rule']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage4_protocol.json")
    parser.add_argument(
        "--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2"
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage4" / "task_difficulty_audit")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert_valid_stage4_protocol(config)
    settings = config["task_difficulty_audit"]
    max_sources = max(int(value) for value in settings["sources_per_scene_candidates"])
    rows: list[dict] = []
    source_ids: dict[str, list[str]] = {}
    for split_role, source_split in settings["source_split_map"].items():
        max_scenes = int(settings["max_scenes"][split_role])
        frames = iter_raddet_frames(
            args.root,
            source_split,
            read_metadata=False,
            max_frames=max_scenes * max_sources,
        )
        source_ids[split_role] = [frame.stem for frame in frames]
        for sources_per_scene in settings["sources_per_scene_candidates"]:
            scenes = grouped_frames(frames, int(sources_per_scene), max_scenes)
            for n_channels in settings["n_channels_candidates"]:
                channel_truth = torch.as_tensor(
                    np.stack(
                        [
                            frame_channel_occupancy(
                                normalized_boxes(scene), int(n_channels), str(settings["channel_axis"])
                            )
                            for scene in scenes
                        ]
                    ),
                    dtype=torch.float32,
                )
                for demand in settings["demand_channels_candidates"]:
                    if int(demand) > int(n_channels):
                        continue
                    diagnostic = resource_task_diagnostics(
                        channel_truth,
                        int(demand),
                        float(settings["clean_threshold"]),
                    )
                    rows.append(
                        {
                            "split_role": split_role,
                            "source_split": source_split,
                            "sources_per_scene": int(sources_per_scene),
                            "n_channels": int(n_channels),
                            "demand_channels": int(demand),
                            "scene_count": len(scenes),
                            **diagnostic.__dict__,
                        }
                    )

    clean_range = tuple(float(value) for value in settings["oracle_clean_rate_range"])
    selection = select_non_saturated_configuration(
        rows,
        selection_split=str(settings["selection_split"]),
        minimum_mean_random_oracle_gap=float(settings["minimum_mean_random_oracle_gap"]),
        maximum_ambiguous_oracle_rate=float(settings["maximum_ambiguous_oracle_rate"]),
        oracle_clean_rate_range=(clean_range[0], clean_range[1]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.out_dir / "task_difficulty_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment_id": "stage4_gate_a_task_difficulty_audit_v1",
        "config_sha256": sha256_file(args.config),
        "source_frame_ids_sha256": {key: sha256_strings(value) for key, value in source_ids.items()},
        "settings": settings,
        "selection": selection,
        "rows": rows,
        "environment": environment_snapshot(["numpy", "torch"]),
        "final_holdout_accessed": False,
        "claim_boundary": settings["claim_boundary"],
    }
    (args.out_dir / "stage4_task_difficulty_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "stage4_task_difficulty_report.md").write_text(
        make_report(rows, selection, settings), encoding="utf-8"
    )
    print(args.out_dir / "stage4_task_difficulty_report.md")


if __name__ == "__main__":
    main()
