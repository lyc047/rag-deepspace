from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import iter_raddet_frames
from train_raddet_occupancy_mask import RadDetMaskDataset, collate, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RadDet occupancy-mask detector by SNR.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--result-json", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    previous = json.loads(args.result_json.read_text(encoding="utf-8"))
    threshold = float(previous["threshold"])
    iou_threshold = float(previous["iou_threshold"])
    min_cells = int(previous["min_cells"])
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    frames = iter_raddet_frames(
        args.root,
        args.split,
        read_metadata=True,
        default_sequence_length=1_000_000,
        max_frames=args.max_frames,
    )
    groups = defaultdict(list)
    for frame in frames:
        groups[frame.snr_db].append(frame)

    model = TinyOccupancyCNN(channels=16).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    rows = []
    for snr in sorted(groups):
        loader = DataLoader(RadDetMaskDataset(groups[snr]), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        metrics = evaluate(model, loader, device, threshold, iou_threshold, min_cells)
        row = {"snr_db": snr, "n_frames": len(groups[snr]), **metrics}
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "raddet_occupancy_mask_by_snr.csv"
    json_path = args.out_dir / "raddet_occupancy_mask_by_snr.json"
    md_path = args.out_dir / "raddet_occupancy_mask_by_snr.md"

    fieldnames = [
        "snr_db",
        "n_frames",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "tp",
        "fp",
        "fn",
        "mean_semantic_bits_per_frame",
        "mean_raw_spectrogram_8bit_bits_per_frame",
        "mean_raw_iq_12bit_bits_per_frame",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": previous["dataset"],
        "model": previous["model"],
        "split": args.split,
        "max_frames": args.max_frames,
        "threshold": threshold,
        "iou_threshold": iou_threshold,
        "min_cells": min_cells,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RadDet Occupancy-Mask SNR Evaluation",
        "",
        f"- split: {args.split}",
        f"- frames: {len(frames)}",
        f"- selected threshold: {threshold:.2f}",
        f"- IoU threshold: {iou_threshold:.2f}",
        "",
        "| SNR (dB) | Frames | Precision | Recall | F1 | Mean IoU | Semantic bits/frame |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['snr_db']:.0f} | {row['n_frames']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['mean_iou']:.4f} | "
            f"{row['mean_semantic_bits_per_frame']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This table separates the payload question from the semantic-extraction reliability question. "
            "If low-SNR F1 collapses while high-SNR F1 remains usable, the next contribution should focus on "
            "soft semantic packets, multi-view fusion, and SNR-aware confidence rather than only reducing bits.",
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
