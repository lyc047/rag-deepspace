from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.data_io import load_sigmf_frame
from spectrum_semcom.datasets import boxes_to_occupancy_mask, sample_stft_patches
from spectrum_semcom.preprocessing import stft_power


DEFAULT_META = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sigmf_5g_short"
    / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
)


def _filter_truth(boxes):
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def _pixel_metrics(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    pred = prob >= threshold
    truth = target >= 0.5
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny STFT-CNN occupancy detector on a small real SigMF sample.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from spectrum_semcom.models import TinyOccupancyCNN
    except Exception as exc:
        raise SystemExit(f"PyTorch is required for this script: {exc}")

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False in this Python environment.")

    frame = load_sigmf_frame(args.meta, normalize=True)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=512, hop_length=256)
    truth = _filter_truth(frame.boxes)
    mask = boxes_to_occupancy_mask(stft, truth)
    patches = sample_stft_patches(
        stft.power_db,
        mask,
        patch_shape=(128, 128),
        max_patches=args.max_patches,
        positive_fraction=0.7,
        seed=args.seed,
    )

    x = torch.from_numpy(patches.x)
    y = torch.from_numpy(patches.y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TinyOccupancyCNN(channels=8).to(device)
    positive = float(y.sum())
    negative = float(y.numel() - y.sum())
    pos_weight_value = min(30.0, negative / max(positive, 1.0))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device).view(1, 1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
    metrics = _pixel_metrics(prob, patches.y, threshold=0.5)

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "script": "train_tiny_cnn_smoke.py",
        "n_patches": int(patches.x.shape[0]),
        "patch_shape": list(patches.x.shape[-2:]),
        "epochs": args.epochs,
        "device": str(device),
        "pos_weight": pos_weight_value,
        "mask_positive_fraction_full_stft": float(mask.mean()),
        "history": history,
        "metrics_on_training_patches": metrics,
        "note": "Smoke/overfit experiment only; not a valid generalization result.",
    }
    out_path = out_dir / "tiny_cnn_smoke_result.json"
    model_path = out_dir / "tiny_cnn_smoke.pt"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model": model.state_dict(), "result": result}, model_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
