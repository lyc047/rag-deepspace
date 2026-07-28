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
from train_raddet_occupancy_mask import match_boxes, parse_thresholds, xyxy_iou


def yolo_center_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def clip_box(box: np.ndarray) -> np.ndarray:
    out = np.asarray(box, dtype=np.float32).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, 1.0)
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, 1.0)
    return out


def nms_boxes(boxes: list[np.ndarray], scores: list[float], iou_threshold: float = 0.5, max_boxes: int = 16) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    keep: list[int] = []
    for idx in order:
        if len(keep) >= max_boxes:
            break
        if all(xyxy_iou(boxes[idx], boxes[kept]) < iou_threshold for kept in keep):
            keep.append(idx)
    return keep


class RadDetGridDataset:
    def __init__(self, frames, grid_size: int = 8, n_classes: int = 11):
        self.frames = frames
        self.grid_size = grid_size
        self.n_classes = n_classes

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame = self.frames[idx]
        img = Image.open(frame.image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        obj = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        box_target = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        cls_target = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        truth_boxes = []
        truth_classes = []
        for box in frame.boxes:
            cx = float(np.clip(box.x_center, 0.0, 0.999999))
            cy = float(np.clip(box.y_center, 0.0, 0.999999))
            gx = int(cx * self.grid_size)
            gy = int(cy * self.grid_size)
            obj[0, gy, gx] = 1.0
            dx = cx * self.grid_size - gx
            dy = cy * self.grid_size - gy
            box_target[:, gy, gx] = np.array([dx, dy, box.width, box.height], dtype=np.float32)
            cls_target[gy, gx] = int(box.class_idx)
            truth_boxes.append(yolo_center_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32)))
            truth_classes.append(int(box.class_idx))
        return {
            "image": arr[None, :, :],
            "obj": obj,
            "box": box_target,
            "cls": cls_target,
            "truth_boxes": truth_boxes,
            "truth_classes": truth_classes,
        }


def collate(batch):
    import torch

    return {
        "image": torch.tensor(np.stack([b["image"] for b in batch]), dtype=torch.float32),
        "obj": torch.tensor(np.stack([b["obj"] for b in batch]), dtype=torch.float32),
        "box": torch.tensor(np.stack([b["box"] for b in batch]), dtype=torch.float32),
        "cls": torch.tensor(np.stack([b["cls"] for b in batch]), dtype=torch.long),
        "truth_boxes": [b["truth_boxes"] for b in batch],
        "truth_classes": [b["truth_classes"] for b in batch],
    }


def decode_predictions(obj_logit, box_pred, cls_logit, threshold: float, nms_iou: float, max_boxes: int):
    import torch

    obj_prob = torch.sigmoid(obj_logit).detach().cpu().numpy()
    boxes_np = box_pred.detach().cpu().numpy()
    cls_np = cls_logit.detach().cpu().numpy()
    batch_boxes = []
    batch_scores = []
    batch_classes = []
    for b in range(obj_prob.shape[0]):
        boxes = []
        scores = []
        classes = []
        _, h, w = obj_prob[b].shape
        for gy in range(h):
            for gx in range(w):
                score = float(obj_prob[b, 0, gy, gx])
                if score < threshold:
                    continue
                dx, dy, bw, bh = boxes_np[b, :, gy, gx]
                cx = (gx + dx) / w
                cy = (gy + dy) / h
                box = clip_box(yolo_center_to_xyxy(np.array([cx, cy, bw, bh], dtype=np.float32)))
                boxes.append(box)
                scores.append(score)
                classes.append(int(np.argmax(cls_np[b, :, gy, gx])))
        keep = nms_boxes(boxes, scores, iou_threshold=nms_iou, max_boxes=max_boxes) if boxes else []
        batch_boxes.append([boxes[i] for i in keep])
        batch_scores.append([scores[i] for i in keep])
        batch_classes.append([classes[i] for i in keep])
    return batch_boxes, batch_scores, batch_classes


def evaluate(model, loader, device: str, threshold: float, iou_threshold: float, nms_iou: float, max_boxes: int) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    ious_all = []
    class_correct = 0
    hard_bits = []
    soft_bits = []
    with torch.no_grad():
        for batch in loader:
            obj_logit, box_pred, cls_logit = model(batch["image"].to(device))
            pred_boxes, _, pred_classes = decode_predictions(obj_logit, box_pred, cls_logit, threshold, nms_iou, max_boxes)
            for idx, boxes in enumerate(pred_boxes):
                truth_boxes = batch["truth_boxes"][idx]
                truth_classes = batch["truth_classes"][idx]
                a, b, c, ious = match_boxes(boxes, truth_boxes, iou_threshold=iou_threshold)
                tp += a
                fp += b
                fn += c
                ious_all.extend(ious)
                hard_bits.append(hard_box_packet_bits(len(boxes)))
                soft_bits.append(soft_box_packet_bits(len(boxes), n_classes=11))
                candidates = []
                for pi, pred_box in enumerate(boxes):
                    for ti, truth_box in enumerate(truth_boxes):
                        iou = xyxy_iou(pred_box, truth_box)
                        if iou >= iou_threshold:
                            candidates.append((iou, pi, ti))
                candidates.sort(reverse=True)
                used_p = set()
                used_t = set()
                for _, pi, ti in candidates:
                    if pi in used_p or ti in used_t:
                        continue
                    used_p.add(pi)
                    used_t.add(ti)
                    class_correct += int(pred_classes[idx][pi] == truth_classes[ti])
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


def evaluate_thresholds(model, loader, device: str, thresholds: list[float], iou_threshold: float, nms_iou: float, max_boxes: int):
    sweep = []
    for threshold in thresholds:
        metrics = evaluate(model, loader, device, threshold, iou_threshold, nms_iou, max_boxes)
        sweep.append({"threshold": threshold, **metrics})
    best = max(sweep, key=lambda item: (item["f1"], item["matched_class_accuracy"], -item["mean_hard_semantic_bits_per_frame"]))
    return best, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny YOLO-style RadDet grid detector.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--train-frames", type=int, default=3000)
    parser.add_argument("--val-frames", type=int, default=1000)
    parser.add_argument("--test-frames", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thresholds", type=str, default="0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.4)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8, choices=[8, 16])
    parser.add_argument("--obj-pos-weight", type=float, default=30.0)
    parser.add_argument("--box-loss-weight", type=float, default=8.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from spectrum_semcom.models import TinyGridDetector16CNN, TinyGridDetectorCNN

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    thresholds = parse_thresholds(args.thresholds)

    def load(split: str, n: int):
        return iter_raddet_frames(args.root, split, read_metadata=False, default_sequence_length=1_000_000, max_frames=n)

    train_loader = DataLoader(
        RadDetGridDataset(load("train", args.train_frames), grid_size=args.grid_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        RadDetGridDataset(load("val", args.val_frames), grid_size=args.grid_size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        RadDetGridDataset(load("test", args.test_frames), grid_size=args.grid_size),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    if args.grid_size == 16:
        model = TinyGridDetector16CNN(n_classes=11, channels=20).to(device)
        model_name = "TinyGridDetector16CNN YOLO-style baseline"
    else:
        model = TinyGridDetectorCNN(n_classes=11, channels=24).to(device)
        model_name = "TinyGridDetectorCNN YOLO-style baseline"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor(args.obj_pos_weight, dtype=torch.float32, device=device)
    history = []
    best_val = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            obj_t = batch["obj"].to(device)
            box_t = batch["box"].to(device)
            cls_t = batch["cls"].to(device)
            obj_logit, box_pred, cls_logit = model(x)
            obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj_t, pos_weight=pos_weight)
            pos = obj_t[:, 0] >= 0.5
            if pos.any():
                box_loss = F.smooth_l1_loss(box_pred.permute(0, 2, 3, 1)[pos], box_t.permute(0, 2, 3, 1)[pos])
                cls_loss = F.cross_entropy(cls_logit.permute(0, 2, 3, 1)[pos], cls_t[pos])
            else:
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)
            loss = obj_loss + args.box_loss_weight * box_loss + args.class_loss_weight * cls_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_best, val_sweep = evaluate_thresholds(model, val_loader, device, thresholds, args.iou_threshold, args.nms_iou, args.max_boxes)
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
    test_metrics = evaluate(model, test_loader, device, best_threshold, args.iou_threshold, args.nms_iou, args.max_boxes)
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
        "model": model_name,
        "device": device,
        "train_frames": args.train_frames,
        "val_frames": args.val_frames,
        "test_frames": args.test_frames,
        "epochs": args.epochs,
        "threshold": best_threshold,
        "thresholds": thresholds,
        "iou_threshold": args.iou_threshold,
        "nms_iou": args.nms_iou,
        "max_boxes": args.max_boxes,
        "grid_size": args.grid_size,
        "best_val": best_val,
        "history": history,
        "test": test_metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "16" if args.grid_size == 16 else "8"
    json_path = args.out_dir / f"raddet_grid{suffix}_detector_result.json"
    pt_path = args.out_dir / f"raddet_grid{suffix}_detector.pt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), pt_path)
    print(f"wrote {json_path}")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
