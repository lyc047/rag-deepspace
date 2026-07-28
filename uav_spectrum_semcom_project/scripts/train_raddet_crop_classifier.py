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
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import iter_raddet_frames
from train_raddet_occupancy_mask import mask_to_boxes, match_boxes, xyxy_iou, yolo_to_xyxy


def clip_xyxy(box: np.ndarray) -> np.ndarray:
    out = np.asarray(box, dtype=np.float32).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, 1.0)
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, 1.0)
    if out[2] <= out[0]:
        out[2] = min(1.0, out[0] + 1 / 128)
    if out[3] <= out[1]:
        out[3] = min(1.0, out[1] + 1 / 128)
    return out


def jitter_box(box: np.ndarray, rng: np.random.Generator, amount: float = 0.08) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    dx = rng.normal(0, amount * max(w, 1 / 128), size=2)
    dy = rng.normal(0, amount * max(h, 1 / 128), size=2)
    return clip_xyxy(np.array([x0 + dx[0], y0 + dy[0], x1 + dx[1], y1 + dy[1]], dtype=np.float32))


def crop_patch(image_path: Path, box_xyxy: np.ndarray, size: int = 32) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    w, h = img.size
    box = clip_xyxy(box_xyxy)
    left = int(np.floor(box[0] * w))
    top = int(np.floor(box[1] * h))
    right = int(np.ceil(box[2] * w))
    bottom = int(np.ceil(box[3] * h))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    patch = img.crop((left, top, right, bottom)).resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr[None, :, :]


class RadDetCropDataset:
    def __init__(self, frames, patch_size: int = 32, jitter_copies: int = 2, seed: int = 0):
        self.samples = []
        rng = np.random.default_rng(seed)
        for frame in frames:
            for box in frame.boxes:
                xyxy = yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
                self.samples.append((frame.image_path, clip_xyxy(xyxy), box.class_idx))
                for _ in range(jitter_copies):
                    self.samples.append((frame.image_path, jitter_box(xyxy, rng), box.class_idx))
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, box, class_idx = self.samples[idx]
        return {
            "patch": crop_patch(image_path, box, size=self.patch_size),
            "class_idx": np.array(class_idx, dtype=np.int64),
        }


def collate_crop(batch):
    import torch

    return {
        "patch": torch.tensor(np.stack([b["patch"] for b in batch]), dtype=torch.float32),
        "class_idx": torch.tensor(np.stack([b["class_idx"] for b in batch]), dtype=torch.long),
    }


def build_patch_classifier(n_classes: int = 11, channels: int = 24):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels * 2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels * 4),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(channels * 4, n_classes),
    )


def evaluate_patch_classifier(classifier, loader, device: str) -> float:
    import torch

    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = classifier(batch["patch"].to(device))
            pred = logits.argmax(dim=1).detach().cpu()
            truth = batch["class_idx"]
            correct += int((pred == truth).sum().item())
            total += int(truth.numel())
    return correct / total if total else 0.0


def evaluate_two_stage(
    mask_model,
    classifier,
    frames,
    device: str,
    threshold: float,
    iou_threshold: float,
    min_cells: int,
    patch_size: int,
) -> dict[str, float]:
    import torch

    mask_model.eval()
    classifier.eval()
    tp = fp = fn = 0
    class_correct = 0
    ious_all = []
    hard_bits = []
    soft_bits = []
    with torch.no_grad():
        for frame in frames:
            img = Image.open(frame.image_path).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            x = torch.tensor(arr[None, None, :, :], dtype=torch.float32, device=device)
            prob = torch.sigmoid(mask_model(x))[0, 0].detach().cpu().numpy()
            pred_boxes = mask_to_boxes(prob >= threshold, min_cells=min_cells)
            truth_boxes = [
                yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
                for box in frame.boxes
            ]
            truth_classes = [box.class_idx for box in frame.boxes]
            a, b, c, ious = match_boxes(pred_boxes, truth_boxes, iou_threshold=iou_threshold)
            tp += a
            fp += b
            fn += c
            ious_all.extend(ious)
            hard_bits.append(hard_box_packet_bits(len(pred_boxes)))
            soft_bits.append(soft_box_packet_bits(len(pred_boxes), n_classes=11))

            pred_classes = []
            for pred_box in pred_boxes:
                patch = crop_patch(frame.image_path, pred_box, size=patch_size)
                logits = classifier(torch.tensor(patch[None, :, :, :], dtype=torch.float32, device=device))
                pred_classes.append(int(logits.argmax(dim=1).item()))

            candidates = []
            for pi, pred_box in enumerate(pred_boxes):
                for ti, truth_box in enumerate(truth_boxes):
                    iou = xyxy_iou(pred_box, truth_box)
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
                class_correct += int(pred_classes[pi] == truth_classes[ti])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "matched_class_accuracy": class_correct / tp if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_hard_semantic_bits_per_frame": float(np.mean(hard_bits)) if hard_bits else 0.0,
        "mean_soft_semantic_bits_per_frame": float(np.mean(soft_bits)) if soft_bits else 0.0,
        "mean_raw_spectrogram_8bit_bits_per_frame": float(stft_payload_bits((128, 128), 8)),
        "mean_raw_iq_12bit_bits_per_frame": float(iq_payload_bits(1_000_000, 12, 12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-stage RadDet mask + crop-classifier semantic baseline.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--train-frames", type=int, default=3000)
    parser.add_argument("--val-frames", type=int, default=1000)
    parser.add_argument("--test-frames", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--jitter-copies", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    mask_cfg = json.loads(args.mask_result.read_text(encoding="utf-8"))
    threshold = float(mask_cfg["threshold"])
    iou_threshold = float(mask_cfg["iou_threshold"])
    min_cells = int(mask_cfg["min_cells"])

    def load(split: str, n: int, read_metadata: bool = False):
        return iter_raddet_frames(args.root, split, read_metadata=read_metadata, default_sequence_length=1_000_000, max_frames=n)

    train_frames = load("train", args.train_frames)
    val_frames = load("val", args.val_frames)
    test_frames = load("test", args.test_frames)
    train_dataset = RadDetCropDataset(train_frames, patch_size=args.patch_size, jitter_copies=args.jitter_copies, seed=args.seed)
    val_dataset = RadDetCropDataset(val_frames, patch_size=args.patch_size, jitter_copies=0, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_crop)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_crop)

    classifier = build_patch_classifier(n_classes=11).to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        losses = []
        for batch in train_loader:
            logits = classifier(batch["patch"].to(device))
            loss = F.cross_entropy(logits, batch["class_idx"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_acc = evaluate_patch_classifier(classifier, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_crop_accuracy": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_crop_acc={val_acc:.4f}")

    if best_state is not None:
        classifier.load_state_dict(best_state)

    mask_model = TinyOccupancyCNN(channels=16).to(device)
    mask_model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    test_metrics = evaluate_two_stage(
        mask_model,
        classifier,
        test_frames,
        device=device,
        threshold=threshold,
        iou_threshold=iou_threshold,
        min_cells=min_cells,
        patch_size=args.patch_size,
    )
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
        "model": "Two-stage TinyOccupancyCNN + crop classifier",
        "device": device,
        "train_frames": args.train_frames,
        "val_frames": args.val_frames,
        "test_frames": args.test_frames,
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "jitter_copies": args.jitter_copies,
        "mask_threshold": threshold,
        "iou_threshold": iou_threshold,
        "min_cells": min_cells,
        "best_val_crop_accuracy": best_acc,
        "history": history,
        "test": test_metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raddet_two_stage_mask_class_result.json"
    pt_path = args.out_dir / "raddet_crop_classifier.pt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(classifier.state_dict(), pt_path)
    print(f"wrote {json_path}")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
