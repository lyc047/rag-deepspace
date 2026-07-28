from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.bit_budget import feature_token_packet_bits, soft_box_packet_bits
from spectrum_semcom.radioml import load_radioml2016a, make_radioml_split, raw_iq_bits_per_radioml_sample


DEFAULT_DATA = PROJECT_DIR / "data" / "raw" / "radioml2016_10a" / "RML2016.10a_dict_optimized.pkl"


def _accuracy(pred, y):
    return float((pred == y).mean())


def _per_snr_accuracy(pred, y, snr):
    out = {}
    for s in sorted(set(int(v) for v in snr.tolist())):
        mask = snr == s
        out[str(s)] = _accuracy(pred[mask], y[mask])
    return out


def _normalize_iq_batch(x: np.ndarray) -> np.ndarray:
    """Per-example zero-mean and unit-power normalization for I/Q snippets."""

    x = np.asarray(x, dtype=np.float32).copy()
    x = x - x.mean(axis=2, keepdims=True)
    power = np.mean(x**2, axis=(1, 2), keepdims=True)
    return x / np.sqrt(power + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN classifier on RadioML2016.10A.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-per-group", type=int, default=200)
    parser.add_argument("--val-per-group", type=int, default=50)
    parser.add_argument("--test-per-group", type=int, default=50)
    parser.add_argument("--min-snr", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from spectrum_semcom.models import TinyRadioMLCNN

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this Python environment.")
    torch.manual_seed(args.seed)

    data = load_radioml2016a(args.data)
    split = make_radioml_split(
        data,
        train_per_group=args.train_per_group,
        val_per_group=args.val_per_group,
        test_per_group=args.test_per_group,
        min_snr=args.min_snr,
        seed=args.seed,
    )
    x_train = _normalize_iq_batch(split.x_train)
    x_val = _normalize_iq_batch(split.x_val)
    x_test = _normalize_iq_batch(split.x_test)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(split.y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = TinyRadioMLCNN(n_classes=len(split.classes), channels=32, feature_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    def predict(x_np):
        preds = []
        probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x_np), args.batch_size):
                xb = torch.from_numpy(x_np[i : i + args.batch_size]).to(device)
                logits = model(xb)
                p = torch.softmax(logits, dim=1)
                probs.append(p.cpu().numpy())
                preds.append(p.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)

    val_pred, _ = predict(x_val)
    test_pred, test_prob = predict(x_test)
    val_acc = _accuracy(val_pred, split.y_val)
    test_acc = _accuracy(test_pred, split.y_test)

    raw_bits = raw_iq_bits_per_radioml_sample()
    hard_class_bits = math.ceil(math.log2(len(split.classes)))
    soft_prob_bits = len(split.classes) * 8
    feature_token_bits = feature_token_packet_bits(n_tokens=1, token_dim=16, bits_per_value=6, metadata_bits=0, position_bits_per_token=0, global_context_bits=0)
    payload = {
        "raw_iq_12bit_iq_bits_per_sample": raw_bits,
        "hard_class_bits_per_sample": hard_class_bits,
        "soft_prob_8bit_bits_per_sample": soft_prob_bits,
        "feature_token_16d_6bit_bits_per_sample": feature_token_bits,
        "raw_to_hard_ratio": raw_bits / hard_class_bits,
        "raw_to_soft_ratio": raw_bits / soft_prob_bits,
        "raw_to_token_ratio": raw_bits / feature_token_bits,
    }

    result = {
        "dataset": "RadioML2016.10A",
        "classes": split.classes,
        "n_classes": len(split.classes),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_train": int(len(split.x_train)),
        "n_val": int(len(split.x_val)),
        "n_test": int(len(split.x_test)),
        "history": history,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_accuracy_by_snr": _per_snr_accuracy(test_pred, split.y_test, split.snr_test),
        "payload_bits": payload,
        "note": "Medium dataset validation for class-level I/Q semantics; not a wideband detection benchmark.",
    }

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "radioml_classifier_result.json"
    model_path = out_dir / "radioml_tiny_cnn.pt"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model": model.state_dict(), "classes": split.classes, "result": result}, model_path)
    print(json.dumps({"val_accuracy": val_acc, "test_accuracy": test_acc, "payload_bits": payload, "device": str(device)}, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
