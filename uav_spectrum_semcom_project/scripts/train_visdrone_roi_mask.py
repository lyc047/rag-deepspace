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

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.visdrone import VisDroneFrame, iter_visdrone_frames
from train_raddet_occupancy_mask import mask_to_boxes, match_boxes


def truth_box_xyxy_norm(box, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            box.x / width,
            box.y / height,
            (box.x + box.width) / width,
            (box.y + box.height) / height,
        ],
        dtype=np.float32,
    )


class VisDroneMaskDataset:
    def __init__(self, frames: list[VisDroneFrame], image_size: int = 128, min_box_size_px: int = 2):
        self.frames = frames
        self.image_size = image_size
        self.min_box_size_px = min_box_size_px

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame = self.frames[idx]
        img = Image.open(frame.image_path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        truth_boxes: list[np.ndarray] = []
        for box in frame.boxes:
            xyxy = truth_box_xyxy_norm(box, frame.width, frame.height)
            x0 = int(np.clip(np.floor(xyxy[0] * self.image_size), 0, self.image_size - 1))
            y0 = int(np.clip(np.floor(xyxy[1] * self.image_size), 0, self.image_size - 1))
            x1 = int(np.clip(np.ceil(xyxy[2] * self.image_size), x0 + 1, self.image_size))
            y1 = int(np.clip(np.ceil(xyxy[3] * self.image_size), y0 + 1, self.image_size))
            if (x1 - x0) < self.min_box_size_px or (y1 - y0) < self.min_box_size_px:
                continue
            mask[y0:y1, x0:x1] = 1.0
            truth_boxes.append(xyxy)
        return {
            "image": arr[None, :, :],
            "mask": mask[None, :, :],
            "truth_boxes": truth_boxes,
            "jpeg_bits": frame.jpeg_bits,
        }


def collate(batch):
    import torch

    return {
        "image": torch.tensor(np.stack([b["image"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32),
        "truth_boxes": [b["truth_boxes"] for b in batch],
        "jpeg_bits": [b["jpeg_bits"] for b in batch],
    }


def evaluate(model, loader, device: str, threshold: float, iou_threshold: float, min_cells: int, max_boxes: int) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    ious_all = []
    semantic_bits = []
    jpeg_bits = []
    mask_inter = mask_union = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            truth_masks = batch["mask"].detach().cpu().numpy()[:, 0]
            for idx, prob in enumerate(probs):
                binary = prob >= threshold
                truth_mask = truth_masks[idx] >= 0.5
                mask_inter += float(np.logical_and(binary, truth_mask).sum())
                mask_union += float(np.logical_or(binary, truth_mask).sum())
                pred_boxes = mask_to_boxes(binary, min_cells=min_cells, max_boxes=max_boxes)
                truth_boxes = batch["truth_boxes"][idx]
                a, b, c, ious = match_boxes(pred_boxes, truth_boxes, iou_threshold=iou_threshold)
                tp += a
                fp += b
                fn += c
                ious_all.extend(ious)
                semantic_bits.append(hard_box_packet_bits(len(pred_boxes)))
                jpeg_bits.append(batch["jpeg_bits"][idx])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_semantic_bits = float(np.mean(semantic_bits)) if semantic_bits else 0.0
    mean_jpeg_bits = float(np.mean(jpeg_bits)) if jpeg_bits else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "mask_iou": mask_inter / mask_union if mask_union > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_semantic_bits_per_frame": mean_semantic_bits,
        "mean_jpeg_bits_per_frame": mean_jpeg_bits,
        "jpeg_to_semantic_ratio": mean_jpeg_bits / max(mean_semantic_bits, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny visual ROI occupancy mask model on VisDrone.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=548)
    parser.add_argument("--train-frames", type=int, default=360)
    parser.add_argument("--val-frames", type=int, default=80)
    parser.add_argument("--test-frames", type=int, default=108)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--thresholds", type=str, default="0.15,0.25,0.35,0.45,0.55,0.65,0.75")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=909)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = iter_visdrone_frames(args.root, max_frames=args.max_frames)
    rng = random.Random(args.seed)
    rng.shuffle(frames)
    train = frames[: args.train_frames]
    val = frames[args.train_frames : args.train_frames + args.val_frames]
    test = frames[args.train_frames + args.val_frames : args.train_frames + args.val_frames + args.test_frames]

    train_loader = DataLoader(
        VisDroneMaskDataset(train, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        VisDroneMaskDataset(val, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        VisDroneMaskDataset(test, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = TinyOccupancyCNN(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device).view(1, 1, 1, 1))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    history = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = criterion(logits, batch["mask"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_sweep = [
            {
                "threshold": threshold,
                **evaluate(model, val_loader, device, threshold, args.iou_threshold, args.min_cells, args.max_boxes),
            }
            for threshold in thresholds
        ]
        epoch_best = max(val_sweep, key=lambda row: (row["f1"], row["mask_iou"]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_best": epoch_best, "val_sweep": val_sweep})
        if best is None or epoch_best["f1"] > best["f1"]:
            best = epoch_best
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best is not None and best_state is not None
    model.load_state_dict(best_state)
    ckpt_path = args.out_dir / "visdrone_roi_mask.pt"
    torch.save(best_state, ckpt_path)
    test_metrics = evaluate(model, test_loader, device, best["threshold"], args.iou_threshold, args.min_cells, args.max_boxes)

    result = {
        "dataset": "VisDrone2019-DET-val",
        "model": "TinyOccupancyCNN visual ROI mask",
        "train_frames": len(train),
        "val_frames": len(val),
        "test_frames": len(test),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "threshold": best["threshold"],
        "iou_threshold": args.iou_threshold,
        "min_cells": args.min_cells,
        "max_boxes": args.max_boxes,
        "best_val": best,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(ckpt_path),
        "notes": [
            "This model predicts visual ROI/occupancy semantics rather than fine-grained object classes.",
            "Box-level evaluation uses connected components from the predicted mask and a low IoU threshold because small VisDrone objects merge at 128x128 resolution.",
        ],
    }
    json_path = args.out_dir / "visdrone_roi_mask_result.json"
    md_path = args.out_dir / "visdrone_roi_mask_result.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VisDrone Visual ROI Mask Baseline",
        "",
        "## Setup",
        "",
        f"- Train/val/test frames: {len(train)}/{len(val)}/{len(test)}",
        f"- Image size: {args.image_size}x{args.image_size}",
        f"- Epochs: {args.epochs}",
        f"- Selected threshold: {best['threshold']:.2f}",
        f"- Box IoU threshold: {args.iou_threshold}",
        "",
        "## Test performance",
        "",
        "| Precision | Recall | F1 | Mask IoU | Mean matched IoU | bits/frame | JPEG/semantic |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1']:.4f} | "
            f"{test_metrics['mask_iou']:.4f} | {test_metrics['mean_iou']:.4f} | "
            f"{test_metrics['mean_semantic_bits_per_frame']:.1f} | {test_metrics['jpeg_to_semantic_ratio']:.1f}x |"
        ),
        "",
        "## Interpretation",
        "",
        "This is the first trainable visual-semantic extractor in the project. "
        "It targets ROI/occupancy semantics for adaptive UAV image transmission, not fine-grained object category semantics. "
        "Its output can drive whether to transmit only object/ROI semantics or request ROI image patches when spectrum resources are clean.",
        "",
        f"Checkpoint: `{ckpt_path}`",
        f"JSON: `{json_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {ckpt_path}")


if __name__ == "__main__":
    main()
