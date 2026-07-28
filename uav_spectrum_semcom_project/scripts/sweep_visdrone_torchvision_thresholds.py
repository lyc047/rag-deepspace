from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from infer_visdrone_torchvision import COCO_TO_VISDRONE, box_xyxy_from_visdrone, evaluate_frame
from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.visdrone import iter_visdrone_frames


def load_model(device: str):
    import torch
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.0).to(resolved_device)
    model.eval()
    return model, weights.transforms(), torch


def plot_threshold_sweep(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    thresholds = [row["score_threshold"] for row in rows]
    ax.plot(thresholds, [row["precision"] for row in rows], marker="o", label="Precision")
    ax.plot(thresholds, [row["recall"] for row in rows], marker="o", label="Recall")
    ax.plot(thresholds, [row["f1"] for row in rows], marker="o", label="F1")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep score thresholds for zero-shot Torchvision VisDrone inference.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thresholds", type=str, default="0.03,0.05,0.1,0.15,0.25,0.35,0.5")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    model, transform, torch = load_model(args.device)
    device = next(model.parameters()).device
    frames = iter_visdrone_frames(args.root, max_frames=args.max_frames)

    cached: list[dict] = []
    with torch.no_grad():
        for frame in frames:
            img = Image.open(frame.image_path).convert("RGB")
            output = model([transform(img).to(device)])[0]
            pred = []
            for box, label, score in zip(output["boxes"].detach().cpu().numpy(), output["labels"].detach().cpu().numpy(), output["scores"].detach().cpu().numpy()):
                category = COCO_TO_VISDRONE.get(int(label))
                if category is None:
                    continue
                pred.append((np.asarray(box, dtype=np.float32), category, float(score)))
            truth_boxes = [box_xyxy_from_visdrone(box) for box in frame.boxes if box.category in {1, 2, 3, 4, 5, 6, 9, 10}]
            truth_categories = [box.category for box in frame.boxes if box.category in {1, 2, 3, 4, 5, 6, 9, 10}]
            cached.append(
                {
                    "jpeg_bits": frame.jpeg_bits,
                    "truth_boxes": truth_boxes,
                    "truth_categories": truth_categories,
                    "pred": pred,
                }
            )

    rows: list[dict] = []
    for threshold in thresholds:
        total = {"tp": 0, "fp": 0, "fn": 0, "matched_class_correct": 0, "mean_iou_sum": 0.0}
        pred_bits = []
        jpeg_bits = []
        n_pred = []
        n_truth = []
        for item in cached:
            selected = [p for p in item["pred"] if p[2] >= threshold]
            pred_boxes = [p[0] for p in selected]
            pred_categories = [p[1] for p in selected]
            metrics = evaluate_frame(
                pred_boxes,
                pred_categories,
                item["truth_boxes"],
                item["truth_categories"],
                args.iou_threshold,
            )
            for key in total:
                total[key] += metrics[key]
            bits = hard_box_packet_bits(len(pred_boxes), label_bits=8, coord_bits=16)
            pred_bits.append(bits)
            jpeg_bits.append(item["jpeg_bits"])
            n_pred.append(len(pred_boxes))
            n_truth.append(len(item["truth_boxes"]))
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "score_threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_class_accuracy": total["matched_class_correct"] / total["tp"] if total["tp"] else 0.0,
                "mean_matched_iou": total["mean_iou_sum"] / total["tp"] if total["tp"] else 0.0,
                "tp": total["tp"],
                "fp": total["fp"],
                "fn": total["fn"],
                "mean_pred_boxes_per_frame": float(np.mean(n_pred)),
                "mean_truth_boxes_per_frame": float(np.mean(n_truth)),
                "mean_pred_semantic_bits_per_frame": float(np.mean(pred_bits)),
                "mean_jpeg_bits_per_frame": float(np.mean(jpeg_bits)),
                "jpeg_to_pred_semantic_ratio": float(np.mean(jpeg_bits) / max(np.mean(pred_bits), 1)),
            }
        )

    best = max(rows, key=lambda row: row["f1"])
    csv_path = args.out_dir / "visdrone_torchvision_threshold_sweep.csv"
    json_path = args.out_dir / "visdrone_torchvision_threshold_sweep.json"
    md_path = args.out_dir / "visdrone_torchvision_threshold_sweep.md"
    plot_path = args.out_dir / "visdrone_torchvision_threshold_sweep.png"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"frames": len(frames), "best_by_f1": best, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_threshold_sweep(rows, plot_path)

    lines = [
        "# VisDrone Torchvision Threshold Sweep",
        "",
        f"- Frames: {len(frames)}",
        f"- IoU threshold: {args.iou_threshold}",
        "",
        "| Score threshold | Precision | Recall | F1 | pred boxes/frame | semantic bits/frame | JPEG/semantic |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['score_threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['mean_pred_boxes_per_frame']:.1f} | {row['mean_pred_semantic_bits_per_frame']:.1f} | {row['jpeg_to_pred_semantic_ratio']:.1f}x |"
        )
    lines.extend(
        [
            "",
            "## Best threshold by F1",
            "",
            f"- Threshold: {best['score_threshold']:.2f}",
            f"- Precision: {best['precision']:.4f}",
            f"- Recall: {best['recall']:.4f}",
            f"- F1: {best['f1']:.4f}",
            "",
            f"Plot: `{plot_path}`",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
