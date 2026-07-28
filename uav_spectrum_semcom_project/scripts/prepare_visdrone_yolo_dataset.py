from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.visdrone import VISDRONE_CLASS_NAMES, VisDroneFrame, iter_visdrone_frames


PRIORITY_VISDRONE_CLASSES = [1, 2, 4, 5, 6, 9, 10]
ALL_DETECTION_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def class_ids(mode: str) -> list[int]:
    if mode == "priority":
        return PRIORITY_VISDRONE_CLASSES
    if mode == "all10":
        return ALL_DETECTION_CLASSES
    raise ValueError(f"Unknown class mode: {mode}")


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    raise ValueError(mode)


def frame_to_yolo_lines(frame: VisDroneFrame, mapping: dict[int, int], min_box_area: float) -> list[str]:
    lines: list[str] = []
    image_area = max(1.0, float(frame.width * frame.height))
    for box in frame.boxes:
        if box.category not in mapping:
            continue
        if max(0.0, box.width) * max(0.0, box.height) / image_area < min_box_area:
            continue
        x_center = (box.x + box.width / 2.0) / frame.width
        y_center = (box.y + box.height / 2.0) / frame.height
        width = box.width / frame.width
        height = box.height / frame.height
        values = [
            mapping[box.category],
            np.clip(x_center, 0.0, 1.0),
            np.clip(y_center, 0.0, 1.0),
            np.clip(width, 0.0, 1.0),
            np.clip(height, 0.0, 1.0),
        ]
        lines.append(f"{int(values[0])} {values[1]:.8f} {values[2]:.8f} {values[3]:.8f} {values[4]:.8f}")
    return lines


def split_frames(frames: list[VisDroneFrame], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[VisDroneFrame]]:
    rng = random.Random(seed)
    shuffled = frames[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_yaml(path: Path, names: list[str]) -> None:
    lines = [
        f"path: {path.parent.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_split(frames: list[VisDroneFrame], mapping: dict[int, int], min_box_area: float) -> dict[str, float | int]:
    boxes_per_frame = []
    class_counts = {VISDRONE_CLASS_NAMES[src_id]: 0 for src_id in mapping}
    for frame in frames:
        count = 0
        image_area = max(1.0, float(frame.width * frame.height))
        for box in frame.boxes:
            if box.category not in mapping:
                continue
            if max(0.0, box.width) * max(0.0, box.height) / image_area < min_box_area:
                continue
            count += 1
            class_counts[VISDRONE_CLASS_NAMES[box.category]] += 1
        boxes_per_frame.append(count)
    return {
        "frames": len(frames),
        "boxes": int(sum(boxes_per_frame)),
        "mean_boxes_per_frame": float(np.mean(boxes_per_frame)) if boxes_per_frame else 0.0,
        "nonempty_frame_ratio": float(np.mean([x > 0 for x in boxes_per_frame])) if boxes_per_frame else 0.0,
        "class_counts": class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to a YOLO-format task-oriented UAV visual semantic dataset.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "data" / "processed" / "visdrone_yolo_priority")
    parser.add_argument("--class-mode", choices=["priority", "all10"], default="priority")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--min-box-area", type=float, default=0.0)
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ids = class_ids(args.class_mode)
    mapping = {src_id: idx for idx, src_id in enumerate(ids)}
    names = [VISDRONE_CLASS_NAMES[src_id] for src_id in ids]
    frames = iter_visdrone_frames(args.root, max_frames=args.max_frames)
    splits = split_frames(frames, args.seed, args.train_ratio, args.val_ratio)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    link_stats: dict[str, int] = {}
    for split, split_frames_list in splits.items():
        image_dir = args.out_dir / "images" / split
        label_dir = args.out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for frame in split_frames_list:
            dst_image = image_dir / frame.image_path.name
            action = safe_link_or_copy(frame.image_path, dst_image, args.link_mode)
            link_stats[action] = link_stats.get(action, 0) + 1
            lines = frame_to_yolo_lines(frame, mapping, args.min_box_area)
            (label_dir / f"{frame.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = args.out_dir / "visdrone_uav_semantic.yaml"
    write_yaml(yaml_path, names)
    train_cmd_path = args.out_dir / "train_yolov8n_2gb_safe.ps1"
    train_cmd_path.write_text(
        "\n".join(
            [
                "# 2GB 显存安全优先配置；需要先安装 ultralytics。",
                "# 当前环境已验证：imgsz=256, batch=1, epochs=1 可在 NVIDIA MX450 2GB 上跑通。",
                "# 注意：Ultralytics 在中文路径下保存目录可能出错，因此默认输出到 C:\\yolo_visdrone_runs。",
                "$env:KMP_DUPLICATE_LIB_OK='TRUE'",
                "$env:OMP_NUM_THREADS='1'",
                "$PY='D:\\anaconda\\envs\\pytorch\\python.exe'",
                "& $PY uav_spectrum_semcom_project\\scripts\\train_visdrone_yolov8n_smoke.py --imgsz 256 --batch 1 --epochs 1 --device 0 --project C:\\yolo_visdrone_runs --name yolov8n_priority_smoke_2gb",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source_root": str(args.root),
        "out_dir": str(args.out_dir),
        "class_mode": args.class_mode,
        "class_mapping": {VISDRONE_CLASS_NAMES[src_id]: dst_id for src_id, dst_id in mapping.items()},
        "yaml": str(yaml_path),
        "train_command": str(train_cmd_path),
        "link_stats": link_stats,
        "splits": {split: summarize_split(items, mapping, args.min_box_area) for split, items in splits.items()},
        "notes": [
            "This dataset is task-oriented: the default priority mode keeps pedestrian, people, car, van, truck, bus, and motor.",
            "Images are hard-linked by default when possible, so the processed YOLO folder does not duplicate image bytes on the same drive.",
            "The generated PowerShell training command is intentionally conservative for a 2GB GPU.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VisDrone YOLO Dataset Preparation",
        "",
        f"- Source: `{args.root}`",
        f"- Output: `{args.out_dir}`",
        f"- Class mode: `{args.class_mode}`",
        f"- Classes: {', '.join(names)}",
        f"- YAML: `{yaml_path}`",
        f"- 2GB-safe train command: `{train_cmd_path}`",
        "",
        "## Split summary",
        "",
        "| Split | Frames | Boxes | Mean boxes/frame | Nonempty ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, stat in summary["splits"].items():
        lines.append(
            f"| {split} | {stat['frames']} | {stat['boxes']} | {stat['mean_boxes_per_frame']:.2f} | {stat['nonempty_frame_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Why this matters",
            "",
            "This prepares a replaceable visual semantic front-end. The current ROI-mask baseline can run the closed loop, but it is weak. "
            "Once a lightweight YOLO detector is trained, its object boxes, classes, and confidence scores can replace ROI-mask semantics without changing the spectrum/DQN closed-loop evaluation.",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {yaml_path}")
    print(f"wrote {train_cmd_path}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
