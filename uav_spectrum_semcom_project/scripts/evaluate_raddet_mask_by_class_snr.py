from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import RADDET_CLASS_NAMES, iter_raddet_frames
from train_raddet_occupancy_mask import mask_to_boxes, xyxy_iou, yolo_to_xyxy


def preprocess_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr[None, None, :, :]


def greedy_match(pred_boxes: list[np.ndarray], truth_items: list[tuple[int, np.ndarray]], iou_threshold: float):
    candidates = []
    for pi, pred in enumerate(pred_boxes):
        for ti, (_, truth) in enumerate(truth_items):
            iou = xyxy_iou(pred, truth)
            if iou >= iou_threshold:
                candidates.append((iou, pi, ti))
    candidates.sort(reverse=True)
    used_pred = set()
    used_truth = set()
    matches = []
    for iou, pi, ti in candidates:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        matches.append((pi, ti, iou))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RadDet mask localization recall by class and SNR.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--result-json", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import torch

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

    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    stats = defaultdict(lambda: {"truth": 0, "matched": 0, "iou_sum": 0.0})
    snr_totals = defaultdict(lambda: {"truth": 0, "matched": 0})
    class_totals = defaultdict(lambda: {"truth": 0, "matched": 0})

    with torch.no_grad():
        for frame in frames:
            x = torch.tensor(preprocess_image(frame.image_path), dtype=torch.float32, device=device)
            prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            pred_boxes = mask_to_boxes(prob >= threshold, min_cells=min_cells)
            truth_items = [
                (
                    box.class_idx,
                    yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32)),
                )
                for box in frame.boxes
            ]
            matches = greedy_match(pred_boxes, truth_items, iou_threshold)
            matched_truth = {ti: iou for _, ti, iou in matches}
            for truth_idx, (class_idx, _) in enumerate(truth_items):
                key = (frame.snr_db, class_idx)
                stats[key]["truth"] += 1
                snr_totals[frame.snr_db]["truth"] += 1
                class_totals[class_idx]["truth"] += 1
                if truth_idx in matched_truth:
                    stats[key]["matched"] += 1
                    stats[key]["iou_sum"] += matched_truth[truth_idx]
                    snr_totals[frame.snr_db]["matched"] += 1
                    class_totals[class_idx]["matched"] += 1

    rows = []
    for (snr, class_idx), item in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if item["truth"] == 0:
            continue
        rows.append(
            {
                "snr_db": snr,
                "class_idx": class_idx,
                "class_name": RADDET_CLASS_NAMES.get(class_idx, f"class_{class_idx}"),
                "truth": item["truth"],
                "matched": item["matched"],
                "recall": item["matched"] / item["truth"],
                "mean_iou": item["iou_sum"] / item["matched"] if item["matched"] else 0.0,
            }
        )

    class_rows = []
    for class_idx, item in sorted(class_totals.items()):
        class_rows.append(
            {
                "class_idx": class_idx,
                "class_name": RADDET_CLASS_NAMES.get(class_idx, f"class_{class_idx}"),
                "truth": item["truth"],
                "matched": item["matched"],
                "recall": item["matched"] / item["truth"] if item["truth"] else 0.0,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "raddet_occupancy_mask_by_class_snr.csv"
    json_path = args.out_dir / "raddet_occupancy_mask_by_class_snr.json"
    md_path = args.out_dir / "raddet_occupancy_mask_by_class_snr.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "class_idx", "class_name", "truth", "matched", "recall", "mean_iou"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": previous["dataset"],
        "model": previous["model"],
        "split": args.split,
        "max_frames": args.max_frames,
        "threshold": threshold,
        "iou_threshold": iou_threshold,
        "class_totals": class_rows,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RadDet Mask Localization Recall by Class and SNR",
        "",
        "## Overall class recall",
        "",
        "| Class | Truth | Matched | Recall |",
        "|---|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(f"| {row['class_name']} | {row['truth']} | {row['matched']} | {row['recall']:.4f} |")
    lines.extend(
        [
            "",
            "## Class × SNR recall",
            "",
            "| SNR | Class | Truth | Matched | Recall | Mean IoU |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['snr_db']:.0f} | {row['class_name']} | {row['truth']} | {row['matched']} | "
            f"{row['recall']:.4f} | {row['mean_iou']:.4f} |"
        )
    lines.extend(["", f"CSV: `{csv_path}`", f"JSON: `{json_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
