from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.bit_budget import hard_box_packet_bits, iq_payload_bits, stft_payload_bits
from spectrum_semcom.raddet import iter_raddet_frames


def yolo_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_to_boxes(mask: np.ndarray, min_cells: int = 4, max_boxes: int = 8) -> list[np.ndarray]:
    labeled, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
    if n <= 0:
        return []
    slices = ndimage.find_objects(labeled)
    counts = np.bincount(labeled.ravel(), minlength=n + 1)
    boxes: list[tuple[int, np.ndarray]] = []
    h, w = mask.shape
    for component_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        count = int(counts[component_id])
        if count < min_cells:
            continue
        ys, xs = component_slice
        x0, x1 = xs.start / w, xs.stop / w
        y0, y1 = ys.start / h, ys.stop / h
        boxes.append((count, np.array([x0, y0, x1, y1], dtype=np.float32)))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:max_boxes]]


def match_boxes(pred: list[np.ndarray], truth: list[np.ndarray], iou_threshold: float) -> tuple[int, int, int, list[float]]:
    candidates: list[tuple[float, int, int]] = []
    for pi, pb in enumerate(pred):
        for ti, tb in enumerate(truth):
            iou = xyxy_iou(pb, tb)
            if iou >= iou_threshold:
                candidates.append((iou, pi, ti))
    candidates.sort(reverse=True)
    used_p = set()
    used_t = set()
    ious = []
    for iou, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        ious.append(iou)
    tp = len(ious)
    fp = max(0, len(pred) - tp)
    fn = max(0, len(truth) - tp)
    return tp, fp, fn, ious


class RadDetMaskDataset:
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
        for box in frame.boxes:
            xyxy = yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
            x0 = int(np.clip(np.floor(xyxy[0] * 128), 0, 127))
            y0 = int(np.clip(np.floor(xyxy[1] * 128), 0, 127))
            x1 = int(np.clip(np.ceil(xyxy[2] * 128), x0 + 1, 128))
            y1 = int(np.clip(np.ceil(xyxy[3] * 128), y0 + 1, 128))
            mask[y0:y1, x0:x1] = 1.0
            truth_boxes.append(xyxy)
        return {"image": arr[None, :, :], "mask": mask[None, :, :], "truth_boxes": truth_boxes}


def collate(batch):
    import torch

    return {
        "image": torch.tensor(np.stack([b["image"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32),
        "truth_boxes": [b["truth_boxes"] for b in batch],
    }


def evaluate(model, loader, device: str, threshold: float, iou_threshold: float, min_cells: int) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    ious_all = []
    semantic_bits = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            for idx, prob in enumerate(probs):
                pred_boxes = mask_to_boxes(prob >= threshold, min_cells=min_cells)
                truth_boxes = batch["truth_boxes"][idx]
                a, b, c, ious = match_boxes(pred_boxes, truth_boxes, iou_threshold=iou_threshold)
                tp += a
                fp += b
                fn += c
                ious_all.extend(ious)
                semantic_bits.append(hard_box_packet_bits(len(pred_boxes)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_semantic_bits_per_frame": float(np.mean(semantic_bits)) if semantic_bits else 0.0,
        "mean_raw_spectrogram_8bit_bits_per_frame": float(stft_payload_bits((128, 128), 8)),
        "mean_raw_iq_12bit_bits_per_frame": float(iq_payload_bits(1_000_000, 12, 12)),
    }


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("threshold list is empty")
    return values


def evaluate_thresholds(
    model,
    loader,
    device: str,
    thresholds: list[float],
    iou_threshold: float,
    min_cells: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    sweep = []
    for threshold in thresholds:
        metrics = evaluate(model, loader, device, threshold, iou_threshold, min_cells)
        metrics = {"threshold": threshold, **metrics}
        sweep.append(metrics)
    best = max(sweep, key=lambda item: (item["f1"], item["recall"], -item["mean_semantic_bits_per_frame"]))
    return best, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny RadDet occupancy-mask baseline.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--train-frames", type=int, default=3000)
    parser.add_argument("--val-frames", type=int, default=1000)
    parser.add_argument("--test-frames", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default="0.25,0.35,0.45,0.5,0.55,0.65,0.75")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-cells", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from spectrum_semcom.models import TinyOccupancyCNN

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    def load(split: str, n: int):
        return iter_raddet_frames(
            args.root,
            split,
            read_metadata=False,
            default_sequence_length=1_000_000,
            max_frames=n,
        )

    train_loader = DataLoader(RadDetMaskDataset(load("train", args.train_frames)), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RadDetMaskDataset(load("val", args.val_frames)), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(RadDetMaskDataset(load("test", args.test_frames)), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TinyOccupancyCNN(channels=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    thresholds = parse_thresholds(args.thresholds)

    history = []
    best_val = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            target = batch["mask"].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_best, val_sweep = evaluate_thresholds(model, val_loader, device, thresholds, args.iou_threshold, args.min_cells)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_best": val_best,
                "val_sweep": val_sweep,
            }
        )
        if best_val is None or val_best["f1"] > best_val["f1"]:
            best_val = val_best
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_f1={val_best['f1']:.4f} threshold={val_best['threshold']:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    best_threshold = float(best_val["threshold"] if best_val is not None else args.threshold)
    test_metrics = evaluate(model, test_loader, device, best_threshold, args.iou_threshold, args.min_cells)
    test_metrics["spectrogram8_to_semantic_ratio"] = test_metrics["mean_raw_spectrogram_8bit_bits_per_frame"] / max(
        test_metrics["mean_semantic_bits_per_frame"], 1
    )
    test_metrics["iq12_to_semantic_ratio"] = test_metrics["mean_raw_iq_12bit_bits_per_frame"] / max(
        test_metrics["mean_semantic_bits_per_frame"], 1
    )
    result = {
        "dataset": "RadDet40k128HW001Tv2",
        "model": "TinyOccupancyCNN mask baseline",
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
        "best_val": best_val,
        "history": history,
        "test": test_metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raddet_occupancy_mask_result.json"
    pt_path = args.out_dir / "raddet_occupancy_mask.pt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), pt_path)
    print(f"wrote {json_path}")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
