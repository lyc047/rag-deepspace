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

from spectrum_semcom.bit_budget import hard_box_packet_bits, iq_payload_bits, stft_payload_bits
from spectrum_semcom.raddet import iter_raddet_frames


def yolo_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def box_iou_yolo(a: np.ndarray, b: np.ndarray) -> float:
    ax = yolo_to_xyxy(a)
    bx = yolo_to_xyxy(b)
    left = max(ax[0], bx[0])
    top = max(ax[1], bx[1])
    right = min(ax[2], bx[2])
    bottom = min(ax[3], bx[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, ax[2] - ax[0]) * max(0.0, ax[3] - ax[1])
    area_b = max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class RadDetSingleBoxDataset:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame = self.frames[idx]
        img = Image.open(frame.image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        if frame.boxes:
            box = frame.boxes[0]
            obj = 1.0
            cls = box.class_idx
            yolo_box = np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32)
        else:
            obj = 0.0
            cls = 0
            yolo_box = np.zeros(4, dtype=np.float32)
        return {
            "image": arr[None, :, :],
            "obj": np.array(obj, dtype=np.float32),
            "cls": np.array(cls, dtype=np.int64),
            "box": yolo_box,
        }


def collate(batch):
    import torch

    return {
        "image": torch.tensor(np.stack([b["image"] for b in batch]), dtype=torch.float32),
        "obj": torch.tensor(np.stack([b["obj"] for b in batch]), dtype=torch.float32),
        "cls": torch.tensor(np.stack([b["cls"] for b in batch]), dtype=torch.long),
        "box": torch.tensor(np.stack([b["box"] for b in batch]), dtype=torch.float32),
    }


def load_split(root: Path, split: str, max_frames: int):
    return iter_raddet_frames(
        root,
        split,
        read_metadata=False,
        default_sequence_length=1_000_000,
        max_frames=max_frames,
    )


def evaluate(model, loader, device: str, threshold: float, iou_threshold: float) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    matched_iou = []
    class_correct = 0
    pred_count = 0
    semantic_bits = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            obj = batch["obj"].to(device)
            cls = batch["cls"].to(device)
            box = batch["box"].to(device)
            obj_logit, class_logits, pred_box = model(x)
            prob = torch.sigmoid(obj_logit)
            pred_obj = prob >= threshold
            pred_cls = class_logits.argmax(dim=1)
            for i in range(x.shape[0]):
                has_truth = bool(obj[i].item() >= 0.5)
                has_pred = bool(pred_obj[i].item())
                semantic_bits.append(hard_box_packet_bits(1 if has_pred else 0))
                if has_pred:
                    pred_count += 1
                if has_pred and has_truth:
                    iou = box_iou_yolo(pred_box[i].detach().cpu().numpy(), box[i].detach().cpu().numpy())
                    if iou >= iou_threshold:
                        tp += 1
                        matched_iou.append(iou)
                        class_correct += int(pred_cls[i].item() == cls[i].item())
                    else:
                        fp += 1
                        fn += 1
                elif has_pred and not has_truth:
                    fp += 1
                elif (not has_pred) and has_truth:
                    fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
        "matched_class_accuracy": class_correct / tp if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_semantic_bits_per_frame": float(np.mean(semantic_bits)) if semantic_bits else 0.0,
        "mean_raw_spectrogram_8bit_bits_per_frame": float(stft_payload_bits((128, 128), 8)),
        "mean_raw_iq_12bit_bits_per_frame": float(iq_payload_bits(1_000_000, 12, 12)),
        "predicted_positive_frames": pred_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny single-box RadDet detector.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--train-frames", type=int, default=3000)
    parser.add_argument("--val-frames", type=int, default=1000)
    parser.add_argument("--test-frames", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from spectrum_semcom.models import TinyRadDetCNN

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    train_frames = load_split(args.root, "train", args.train_frames)
    val_frames = load_split(args.root, "val", args.val_frames)
    test_frames = load_split(args.root, "test", args.test_frames)

    train_loader = DataLoader(RadDetSingleBoxDataset(train_frames), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RadDetSingleBoxDataset(val_frames), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(RadDetSingleBoxDataset(test_frames), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TinyRadDetCNN(n_classes=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            obj = batch["obj"].to(device)
            cls = batch["cls"].to(device)
            box = batch["box"].to(device)
            obj_logit, class_logits, pred_box = model(x)
            obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj)
            pos = obj >= 0.5
            if pos.any():
                cls_loss = F.cross_entropy(class_logits[pos], cls[pos])
                box_loss = F.smooth_l1_loss(pred_box[pos], box[pos])
            else:
                cls_loss = torch.tensor(0.0, device=device)
                box_loss = torch.tensor(0.0, device=device)
            loss = obj_loss + cls_loss * 0.7 + box_loss * 5.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(model, val_loader, device, args.threshold, args.iou_threshold)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics})
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_f1={val_metrics['f1']:.4f}")

    test_metrics = evaluate(model, test_loader, device, args.threshold, args.iou_threshold)
    result = {
        "dataset": "RadDet40k128HW001Tv2",
        "model": "TinyRadDetCNN single-box baseline",
        "device": device,
        "train_frames": len(train_frames),
        "val_frames": len(val_frames),
        "test_frames": len(test_frames),
        "epochs": args.epochs,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "history": history,
        "test": test_metrics,
    }
    result["test"]["spectrogram8_to_semantic_ratio"] = result["test"]["mean_raw_spectrogram_8bit_bits_per_frame"] / max(
        result["test"]["mean_semantic_bits_per_frame"], 1
    )
    result["test"]["iq12_to_semantic_ratio"] = result["test"]["mean_raw_iq_12bit_bits_per_frame"] / max(
        result["test"]["mean_semantic_bits_per_frame"], 1
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raddet_tiny_detector_result.json"
    pt_path = args.out_dir / "raddet_tiny_detector.pt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), pt_path)
    print(f"wrote {json_path}")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
