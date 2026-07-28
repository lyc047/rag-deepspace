from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.baselines import binary_mask_to_boxes
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits
from spectrum_semcom.datasets import boxes_to_occupancy_mask, normalize_stft_patch, sample_stft_patches
from spectrum_semcom.frame_manifest import load_frame_manifest, load_iq_frame_from_manifest_row
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket


def _filter_truth(boxes):
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def _build_patch_dataset(rows, n_fft: int, hop_length: int, patches_per_frame: int, seed: int):
    xs = []
    ys = []
    for idx, row in enumerate(rows):
        frame = load_iq_frame_from_manifest_row(row)
        stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=n_fft, hop_length=hop_length)
        mask = boxes_to_occupancy_mask(stft, _filter_truth(frame.boxes))
        patches = sample_stft_patches(
            stft.power_db,
            mask,
            patch_shape=(128, 128),
            max_patches=patches_per_frame,
            positive_fraction=0.7,
            seed=seed + idx,
        )
        xs.append(patches.x)
        ys.append(patches.y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _tiled_predict(model, power_db: np.ndarray, device, patch_shape=(128, 128), stride=(96, 96)):
    import torch

    ph, pw = patch_shape
    sf, st = stride
    height, width = power_db.shape
    prob_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    f_positions = list(range(0, max(1, height - ph + 1), sf))
    t_positions = list(range(0, max(1, width - pw + 1), st))
    if f_positions[-1] != height - ph:
        f_positions.append(height - ph)
    if t_positions[-1] != width - pw:
        t_positions.append(width - pw)

    model.eval()
    with torch.no_grad():
        for f0 in f_positions:
            batch = []
            locs = []
            for t0 in t_positions:
                patch = normalize_stft_patch(power_db[f0 : f0 + ph, t0 : t0 + pw])
                batch.append(patch[None, :, :])
                locs.append((f0, t0))
                if len(batch) == 32:
                    xb = torch.from_numpy(np.stack(batch)).to(device)
                    probs = torch.sigmoid(model(xb)).cpu().numpy()[:, 0]
                    for prob, (ff, tt) in zip(probs, locs):
                        prob_sum[ff : ff + ph, tt : tt + pw] += prob
                        count[ff : ff + ph, tt : tt + pw] += 1.0
                    batch = []
                    locs = []
            if batch:
                xb = torch.from_numpy(np.stack(batch)).to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()[:, 0]
                for prob, (ff, tt) in zip(probs, locs):
                    prob_sum[ff : ff + ph, tt : tt + pw] += prob
                    count[ff : ff + ph, tt : tt + pw] += 1.0
    return prob_sum / np.maximum(count, 1.0)


def _evaluate_frames(rows, model, device, thresholds, n_fft: int, hop_length: int, min_cells: int):
    by_threshold = []
    for threshold in thresholds:
        eval_rows = []
        for row in rows:
            frame = load_iq_frame_from_manifest_row(row)
            stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=n_fft, hop_length=hop_length)
            prob = _tiled_predict(model, stft.power_db, device=device)
            pred = binary_mask_to_boxes(prob >= threshold, stft, min_cells=min_cells, max_boxes=64)
            truth = _filter_truth(frame.boxes)
            metrics = evaluate_detections(pred, truth, iou_threshold=0.05)
            packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred, metadata_bits=160)
            bits = semantic_packet_bits(packet)
            eval_rows.append(
                {
                    "frame_id": row.frame_id,
                    "threshold": threshold,
                    **asdict(metrics),
                    "semantic_packet_bps": BitBudget("semantic_packet", bits, frame.duration_s).bitrate_bps,
                    "semantic_packet_bits_per_frame": bits,
                    "raw_iq_12bit_iq_bps": BitBudget(
                        "raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s
                    ).bitrate_bps,
                }
            )
        tp = sum(item["true_positive"] for item in eval_rows)
        fp = sum(item["false_positive"] for item in eval_rows)
        fn = sum(item["false_negative"] for item in eval_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_threshold.append(
            {
                "threshold": threshold,
                "aggregate": {
                    "n_frames": len(rows),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_frame_f1": sum(item["f1"] for item in eval_rows) / len(eval_rows) if eval_rows else 0.0,
                    "mean_semantic_packet_bps": sum(item["semantic_packet_bps"] for item in eval_rows) / len(eval_rows) if eval_rows else 0.0,
                },
                "rows": eval_rows,
            }
        )
    return by_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyOccupancyCNN with frame-level train/val/test split.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "frame_manifest.csv")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patches-per-frame", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from spectrum_semcom.models import TinyOccupancyCNN

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False in this Python environment.")
    torch.manual_seed(args.seed)

    rows = load_frame_manifest(args.manifest, project_dir=PROJECT_DIR)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]

    x_np, y_np = _build_patch_dataset(train_rows, args.n_fft, args.hop_length, args.patches_per_frame, args.seed)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    model = TinyOccupancyCNN(channels=8).to(device)
    positive = float(y.sum())
    negative = float(y.numel() - y.sum())
    pos_weight_value = min(30.0, negative / max(positive, 1.0))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device).view(1, 1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        losses = []
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    val_results = _evaluate_frames(val_rows, model, device, args.thresholds, args.n_fft, args.hop_length, args.min_cells)
    best_val = max(val_results, key=lambda item: item["aggregate"]["f1"])
    test_results = _evaluate_frames(test_rows, model, device, [best_val["threshold"]], args.n_fft, args.hop_length, args.min_cells)[0]

    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "n_train_frames": len(train_rows),
        "n_val_frames": len(val_rows),
        "n_test_frames": len(test_rows),
        "n_train_patches": int(x_np.shape[0]),
        "pos_weight": pos_weight_value,
        "history": history,
        "best_val": best_val,
        "test": test_results,
        "note": "Frame-level split smoke result on one short recording; still not a final generalization claim.",
    }

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_cnn_train_val_test.json"
    model_path = out_dir / "frame_cnn.pt"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model": model.state_dict(), "result": result}, model_path)
    print(json.dumps({"best_val": best_val["aggregate"], "test": test_results["aggregate"], "device": str(device)}, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()

