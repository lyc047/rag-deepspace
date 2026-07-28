from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.visdrone import VisDroneBox, VisDroneFrame, iter_visdrone_frames


COCO_TO_VISDRONE = {
    1: 1,  # person -> pedestrian/people coarse human class
    2: 3,  # bicycle
    3: 4,  # car
    4: 10,  # motorcycle -> motor
    6: 9,  # bus
    8: 6,  # truck
}


def box_xyxy_from_visdrone(box: VisDroneBox) -> np.ndarray:
    return np.array([box.x, box.y, box.x + box.width, box.y + box.height], dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_class_match(pred_category: int, truth_category: int) -> bool:
    # VisDrone distinguishes pedestrian and people, while COCO only has person.
    if pred_category == 1 and truth_category in {1, 2}:
        return True
    return pred_category == truth_category


def evaluate_frame(
    pred_boxes: list[np.ndarray],
    pred_categories: list[int],
    truth_boxes: list[np.ndarray],
    truth_categories: list[int],
    iou_threshold: float,
) -> dict[str, float]:
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_boxes):
        for ti, truth in enumerate(truth_boxes):
            iou = box_iou(pred, truth)
            if iou >= iou_threshold:
                candidates.append((iou, pi, ti))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matched_ious: list[float] = []
    class_matches = 0
    for iou, pi, ti in candidates:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
        matched_ious.append(iou)
        if is_class_match(pred_categories[pi], truth_categories[ti]):
            class_matches += 1
    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(truth_boxes) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_class_correct": class_matches,
        "mean_iou_sum": float(sum(matched_ious)),
    }


def load_model(device: str):
    import torch
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.0).to(resolved_device)
    model.eval()
    transform = weights.transforms()
    return model, transform, torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight pretrained detector on VisDrone and evaluate visual semantics.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, transform, torch = load_model(args.device)
    device = next(model.parameters()).device
    frames = iter_visdrone_frames(args.root, max_frames=args.max_frames)

    rows: list[dict] = []
    total = {"tp": 0, "fp": 0, "fn": 0, "matched_class_correct": 0, "mean_iou_sum": 0.0}
    pred_bits: list[float] = []
    truth_bits: list[float] = []
    jpeg_bits: list[float] = []

    with torch.no_grad():
        for frame in frames:
            img = Image.open(frame.image_path).convert("RGB")
            x = transform(img).to(device)
            output = model([x])[0]
            pred_boxes: list[np.ndarray] = []
            pred_categories: list[int] = []
            pred_scores: list[float] = []
            for box, label, score in zip(output["boxes"].detach().cpu().numpy(), output["labels"].detach().cpu().numpy(), output["scores"].detach().cpu().numpy()):
                if float(score) < args.score_threshold:
                    continue
                category = COCO_TO_VISDRONE.get(int(label))
                if category is None:
                    continue
                pred_boxes.append(np.asarray(box, dtype=np.float32))
                pred_categories.append(category)
                pred_scores.append(float(score))

            truth_boxes = [box_xyxy_from_visdrone(box) for box in frame.boxes if box.category in {1, 2, 3, 4, 5, 6, 9, 10}]
            truth_categories = [box.category for box in frame.boxes if box.category in {1, 2, 3, 4, 5, 6, 9, 10}]
            metrics = evaluate_frame(pred_boxes, pred_categories, truth_boxes, truth_categories, args.iou_threshold)
            for key in total:
                total[key] += metrics[key]
            bits = hard_box_packet_bits(len(pred_boxes), label_bits=8, coord_bits=16)
            pred_bits.append(bits)
            truth_bits.append(hard_box_packet_bits(len(truth_boxes), label_bits=8, coord_bits=16))
            jpeg_bits.append(frame.jpeg_bits)
            rows.append(
                {
                    "stem": frame.stem,
                    "n_truth": len(truth_boxes),
                    "n_pred": len(pred_boxes),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "pred_semantic_bits": bits,
                    "truth_semantic_bits": truth_bits[-1],
                    "jpeg_bits": frame.jpeg_bits,
                    "jpeg_to_pred_semantic_ratio": frame.jpeg_bits / max(bits, 1),
                    "mean_pred_score": float(np.mean(pred_scores)) if pred_scores else 0.0,
                }
            )

    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_class_accuracy = total["matched_class_correct"] / total["tp"] if total["tp"] else 0.0
    mean_iou = total["mean_iou_sum"] / total["tp"] if total["tp"] else 0.0

    summary = {
        "dataset": "VisDrone2019-DET-val",
        "detector": "torchvision fasterrcnn_mobilenet_v3_large_320_fpn COCO pretrained",
        "frames": len(frames),
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_class_accuracy": matched_class_accuracy,
        "mean_matched_iou": mean_iou,
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "mean_pred_boxes_per_frame": float(np.mean([row["n_pred"] for row in rows])),
        "mean_truth_boxes_per_frame": float(np.mean([row["n_truth"] for row in rows])),
        "mean_pred_semantic_bits_per_frame": float(np.mean(pred_bits)),
        "mean_truth_semantic_bits_per_frame": float(np.mean(truth_bits)),
        "mean_jpeg_bits_per_frame": float(np.mean(jpeg_bits)),
        "jpeg_to_pred_semantic_ratio": float(np.mean(jpeg_bits) / max(np.mean(pred_bits), 1)),
        "notes": [
            "This is a zero-shot COCO-pretrained detector, not fine-tuned on VisDrone.",
            "COCO person is treated as a coarse match for VisDrone pedestrian/people.",
            "The result estimates the feasibility of model-generated visual semantic packets before VisDrone fine-tuning.",
        ],
    }

    csv_path = args.out_dir / "visdrone_torchvision_inference.csv"
    json_path = args.out_dir / "visdrone_torchvision_inference.json"
    md_path = args.out_dir / "visdrone_torchvision_inference.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VisDrone Torchvision Visual Semantic Inference",
        "",
        "## Setup",
        "",
        f"- Detector: {summary['detector']}",
        f"- Frames: {summary['frames']}",
        f"- Score threshold: {args.score_threshold}",
        f"- IoU threshold: {args.iou_threshold}",
        "",
        "## Detection performance",
        "",
        "| Precision | Recall | F1 | Matched class acc | Mean matched IoU | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {precision:.4f} | {recall:.4f} | {f1:.4f} | {matched_class_accuracy:.4f} | {mean_iou:.4f} | {total['tp']} | {total['fp']} | {total['fn']} |",
        "",
        "## Payload",
        "",
        "| Payload | Mean bits/frame | Ratio |",
        "|---|---:|---:|",
        f"| JPEG image | {summary['mean_jpeg_bits_per_frame']:.1f} | {summary['jpeg_to_pred_semantic_ratio']:.1f}x vs predicted semantics |",
        f"| Predicted visual semantic boxes | {summary['mean_pred_semantic_bits_per_frame']:.1f} | 1.0x |",
        f"| Ground-truth visual semantic boxes | {summary['mean_truth_semantic_bits_per_frame']:.1f} | - |",
        "",
        "## Interpretation",
        "",
        "The zero-shot COCO detector provides a first model-generated visual-semantic baseline. "
        "Because VisDrone contains many small aerial-view objects, recall is expected to be limited without dataset-specific fine-tuning. "
        "This result is mainly used to establish the next training/fine-tuning target and to quantify the payload of predicted visual semantic packets.",
        "",
        f"CSV: `{csv_path}`",
        f"JSON: `{json_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
