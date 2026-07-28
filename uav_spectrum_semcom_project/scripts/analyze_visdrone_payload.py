from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.visdrone import VISDRONE_CLASS_NAMES, iter_visdrone_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze UAV visual semantic payloads on VisDrone.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_visdrone_frames(args.root, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError(f"No VisDrone frames found under {args.root}")

    class_counts: Counter[int] = Counter()
    for frame in frames:
        class_counts.update(box.category for box in frame.boxes)

    rows = [
        {
            "stem": frame.stem,
            "width": frame.width,
            "height": frame.height,
            "n_boxes": len(frame.boxes),
            "jpeg_bits": frame.jpeg_bits,
            "raw_rgb_8bit_bits": frame.raw_rgb_8bit_bits,
            "semantic_box_bits": frame.semantic_box_bits,
            "jpeg_to_semantic_ratio": frame.jpeg_bits / max(frame.semantic_box_bits, 1),
            "raw_rgb_to_semantic_ratio": frame.raw_rgb_8bit_bits / max(frame.semantic_box_bits, 1),
        }
        for frame in frames
    ]

    mean_jpeg = float(np.mean([frame.jpeg_bits for frame in frames]))
    mean_raw = float(np.mean([frame.raw_rgb_8bit_bits for frame in frames]))
    mean_sem = float(np.mean([frame.semantic_box_bits for frame in frames]))
    summary = {
        "dataset": "VisDrone2019-DET-val",
        "root": str(args.root),
        "n_frames": len(frames),
        "mean_width": float(np.mean([frame.width for frame in frames])),
        "mean_height": float(np.mean([frame.height for frame in frames])),
        "mean_boxes_per_frame": float(np.mean([len(frame.boxes) for frame in frames])),
        "median_boxes_per_frame": float(np.median([len(frame.boxes) for frame in frames])),
        "nonempty_frame_ratio": float(np.mean([len(frame.boxes) > 0 for frame in frames])),
        "mean_jpeg_bits_per_frame": mean_jpeg,
        "mean_raw_rgb_8bit_bits_per_frame": mean_raw,
        "mean_semantic_box_bits_per_frame": mean_sem,
        "jpeg_to_semantic_ratio": mean_jpeg / max(mean_sem, 1),
        "raw_rgb_to_semantic_ratio": mean_raw / max(mean_sem, 1),
        "class_counts": {
            VISDRONE_CLASS_NAMES.get(idx, f"class_{idx}"): count for idx, count in sorted(class_counts.items())
        },
        "notes": [
            "Visual semantics are represented as class labels plus bounding boxes.",
            "Category 0 ignored regions are excluded.",
            "This is a payload/statistics analysis using ground-truth annotations, not a trained visual detector yet.",
        ],
    }

    csv_path = args.out_dir / "visdrone_payload_summary.csv"
    json_path = args.out_dir / "visdrone_payload_summary.json"
    md_path = args.out_dir / "visdrone_payload_report.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VisDrone Visual Semantic Payload Report",
        "",
        "## Dataset",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Frames: {summary['n_frames']}",
        f"- Mean image size: {summary['mean_width']:.1f} × {summary['mean_height']:.1f}",
        f"- Mean boxes/frame: {summary['mean_boxes_per_frame']:.2f}",
        f"- Median boxes/frame: {summary['median_boxes_per_frame']:.1f}",
        f"- Nonempty frame ratio: {summary['nonempty_frame_ratio']:.3f}",
        "",
        "## Payload comparison",
        "",
        "| Payload | Mean bits/frame | Ratio vs visual semantics |",
        "|---|---:|---:|",
        f"| JPEG image | {mean_jpeg:.1f} | {summary['jpeg_to_semantic_ratio']:.1f}x |",
        f"| Raw RGB 8-bit image | {mean_raw:.1f} | {summary['raw_rgb_to_semantic_ratio']:.1f}x |",
        f"| Visual semantic boxes | {mean_sem:.1f} | 1.0x |",
        "",
        "## Class distribution",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for name, count in summary["class_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report establishes the visual side of the later multimodal semantic system. "
            "For UAV visual monitoring tasks, transmitting object-level semantics can be much smaller than forwarding JPEG or raw RGB images, but this only preserves task-level information rather than reconstructable imagery.",
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
