from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import hard_box_packet_bits, iq_payload_bits, soft_box_packet_bits, stft_payload_bits
from spectrum_semcom.raddet import iter_raddet_frames
from train_raddet_occupancy_mask import mask_to_boxes, match_boxes, parse_thresholds, yolo_to_xyxy


class RadDetMaskClassDataset:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame = self.frames[idx]
        img = Image.open(frame.image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        mask = np.zeros((128, 128), dtype=np.float32)
        truth_boxes = []
        truth_classes = []
        for box in frame.boxes:
            xyxy = yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
            x0 = int(np.clip(np.floor(xyxy[0] * 128), 0, 127))
            y0 = int(np.clip(np.floor(xyxy[1] * 128), 0, 127))
            x1 = int(np.clip(np.ceil(xyxy[2] * 128), x0 + 1, 128))
            y1 = int(np.clip(np.ceil(xyxy[3] * 128), y0 + 1, 128))
            mask[y0:y1, x0:x1] = 1.0
            truth_boxes.append(xyxy)
            truth_classes.append(box.class_idx)
        has_object = len(truth_classes) > 0
        # First class is sufficient for the first mostly-single-object baseline.
        frame_class = truth_classes[0] if has_object else 0
        return {
            "image": arr[None, :, :],
            "mask": mask[None, :, :],
            "truth_boxes": truth_boxes,
            "truth_classes": truth_classes,
            "has_object": np.array(float(has_object), dtype=np.float32),
            "frame_class": np.array(frame_class, dtype=np.int64),
        }


def collate(batch):
    import torch

    return {
        "image": torch.tensor(np.stack([b["image"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32),
        "has_object": torch.tensor(np.stack([b["has_object"] for b in batch]), dtype=torch.float32),
        "frame_class": torch.tensor(np.stack([b["frame_class"] for b in batch]), dtype=torch.long),
        "truth_boxes": [b["truth_boxes"] for b in batch],
        "truth_classes": [b["truth_classes"] for b in batch],
    }


def evaluate(model, loader, device: str, threshold: float, iou_threshold: float, min_cells: int) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    ious_all = []
    semantic_bits_hard = []
    semantic_bits_soft = []
    matched_class_correct = 0
    positive_frames = 0
    positive_frame_class_correct = 0
    positive_frame_count = 0
    with torch.no_grad():
        for batch in loader:
            mask_logits, class_logits = model(batch["image"].to(device))
            probs = torch.sigmoid(mask_logits).detach().cpu().numpy()[:, 0]
            pred_classes = class_logits.argmax(dim=1).detach().cpu().numpy()
            for idx, prob in enumerate(probs):
                pred_boxes = mask_to_boxes(prob >= threshold, min_cells=min_cells)
                truth_boxes = batch["truth_boxes"][idx]
                truth_classes = batch["truth_classes"][idx]
                matches = match_boxes(pred_boxes, truth_boxes, iou_threshold=iou_threshold)
                a, b, c, ious = matches
                tp += a
                fp += b
                fn += c
                ious_all.extend(ious)
                n_pred = len(pred_boxes)
                semantic_bits_hard.append(hard_box_packet_bits(n_pred))
                semantic_bits_soft.append(soft_box_packet_bits(n_pred, n_classes=11))
                if n_pred > 0:
                    positive_frames += 1
                if truth_classes:
                    positive_frame_count += 1
                    positive_frame_class_correct += int(pred_classes[idx] == truth_classes[0])
                # match_boxes does not return matched indices, so recompute greedily for class correctness.
                if a:
                    candidates = []
                    from train_raddet_occupancy_mask import xyxy_iou

                    for pi, pb in enumerate(pred_boxes):
                        for ti, tb in enumerate(truth_boxes):
                            iou = xyxy_iou(pb, tb)
                            if iou >= iou_threshold:
                                candidates.append((iou, pi, ti))
                    candidates.sort(reverse=True)
                    used_pred = set()
                    used_truth = set()
                    for _, pi, ti in candidates:
                        if pi in used_pred or ti in used_truth:
                            continue
                        used_pred.add(pi)
                        used_truth.add(ti)
                        matched_class_correct += int(pred_classes[idx] == truth_classes[ti])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "matched_class_accuracy": matched_class_correct / tp if tp else 0.0,
        "positive_frame_class_accuracy": positive_frame_class_correct / positive_frame_count if positive_frame_count else 0.0,
        "positive_frames_predicted": positive_frames,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_hard_semantic_bits_per_frame": float(np.mean(semantic_bits_hard)) if semantic_bits_hard else 0.0,
        "mean_soft_semantic_bits_per_frame": float(np.mean(semantic_bits_soft)) if semantic_bits_soft else 0.0,
        "mean_raw_spectrogram_8bit_bits_per_frame": float(stft_payload_bits((128, 128), 8)),
        "mean_raw_iq_12bit_bits_per_frame": float(iq_payload_bits(1_000_000, 12, 12)),
    }


def evaluate_thresholds(model, loader, device: str, thresholds: list[float], iou_threshold: float, min_cells: int):
    sweep = []
    for threshold in thresholds:
        metrics = evaluate(model, loader, device, threshold, iou_threshold, min_cells)
        sweep.append({"threshold": threshold, **metrics})
    best = max(sweep, key=lambda item: (item["f1"], item["matched_class_accuracy"], -item["mean_hard_semantic_bits_per_frame"]))
    return best, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RadDet occupancy mask + frame class baseline.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--train-frames", type=int, default=3000)
    parser.add_argument("--val-frames", type=int, default=1000)
    parser.add_argument("--test-frames", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thresholds", type=str, default="0.15,0.25,0.35,0.45,0.5,0.55,0.65,0.75,0.85")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-cells", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=80.0)
    parser.add_argument("--class-loss-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from spectrum_semcom.models import TinyOccupancyClassCNN

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    def load(split: str, n: int):
        return iter_raddet_frames(args.root, split, read_metadata=False, default_sequence_length=1_000_000, max_frames=n)

    train_loader = DataLoader(RadDetMaskClassDataset(load("train", args.train_frames)), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RadDetMaskClassDataset(load("val", args.val_frames)), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(RadDetMaskClassDataset(load("test", args.test_frames)), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TinyOccupancyClassCNN(n_classes=11, channels=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    thresholds = parse_thresholds(args.thresholds)

    best_val = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            target = batch["mask"].to(device)
            has_object = batch["has_object"].to(device) >= 0.5
            frame_class = batch["frame_class"].to(device)
            mask_logits, class_logits = model(x)
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pos_weight)
            if has_object.any():
                class_loss = F.cross_entropy(class_logits[has_object], frame_class[has_object])
            else:
                class_loss = torch.tensor(0.0, device=device)
            loss = mask_loss + args.class_loss_weight * class_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_best, val_sweep = evaluate_thresholds(model, val_loader, device, thresholds, args.iou_threshold, args.min_cells)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_best": val_best, "val_sweep": val_sweep})
        if best_val is None or val_best["f1"] > best_val["f1"]:
            best_val = val_best
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} val_f1={val_best['f1']:.4f} "
            f"cls_acc={val_best['matched_class_accuracy']:.4f} threshold={val_best['threshold']:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    best_threshold = float(best_val["threshold"])
    test_metrics = evaluate(model, test_loader, device, best_threshold, args.iou_threshold, args.min_cells)
    test_metrics["spectrogram8_to_hard_semantic_ratio"] = test_metrics["mean_raw_spectrogram_8bit_bits_per_frame"] / max(
        test_metrics["mean_hard_semantic_bits_per_frame"], 1
    )
    test_metrics["iq12_to_hard_semantic_ratio"] = test_metrics["mean_raw_iq_12bit_bits_per_frame"] / max(
        test_metrics["mean_hard_semantic_bits_per_frame"], 1
    )
    test_metrics["spectrogram8_to_soft_semantic_ratio"] = test_metrics["mean_raw_spectrogram_8bit_bits_per_frame"] / max(
        test_metrics["mean_soft_semantic_bits_per_frame"], 1
    )
    test_metrics["iq12_to_soft_semantic_ratio"] = test_metrics["mean_raw_iq_12bit_bits_per_frame"] / max(
        test_metrics["mean_soft_semantic_bits_per_frame"], 1
    )

    result = {
        "dataset": "RadDet40k128HW001Tv2",
        "model": "TinyOccupancyClassCNN mask+class baseline",
        "device": device,
        "train_frames": args.train_frames,
        "val_frames": args.val_frames,
        "test_frames": args.test_frames,
        "epochs": args.epochs,
        "threshold": best_threshold,
        "thresholds": thresholds,
        "iou_threshold": args.iou_threshold,
        "min_cells": args.min_cells,
        "pos_weight": args.pos_weight,
        "class_loss_weight": args.class_loss_weight,
        "best_val": best_val,
        "history": history,
        "test": test_metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raddet_mask_class_result.json"
    pt_path = args.out_dir / "raddet_mask_class.pt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), pt_path)
    print(f"wrote {json_path}")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
