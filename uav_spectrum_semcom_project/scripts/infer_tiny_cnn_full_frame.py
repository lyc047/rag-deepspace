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
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits, stft_payload_bits
from spectrum_semcom.data_io import load_sigmf_frame
from spectrum_semcom.datasets import boxes_to_occupancy_mask, normalize_stft_patch
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket


DEFAULT_META = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sigmf_5g_short"
    / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
)
DEFAULT_MODEL = PROJECT_DIR / "results" / "phase1" / "tiny_cnn_smoke.pt"


def _filter_truth(boxes):
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def _tiled_predict(model, power_db: np.ndarray, patch_shape=(128, 128), stride=(96, 96), device=None):
    import torch
    if device is None:
        device = torch.device("cpu")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-frame inference with the tiny STFT-CNN smoke model.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    import torch
    from spectrum_semcom.models import TinyOccupancyCNN

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False in this Python environment.")

    frame = load_sigmf_frame(args.meta, normalize=True)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=512, hop_length=256)
    truth = _filter_truth(frame.boxes)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = TinyOccupancyCNN(channels=8).to(device)
    model.load_state_dict(checkpoint["model"])

    prob = _tiled_predict(model, stft.power_db, device=device)
    rows = []
    for threshold in args.thresholds:
        active = prob >= threshold
        pred = binary_mask_to_boxes(active, stft, min_cells=args.min_cells, max_boxes=128)
        metrics = evaluate_detections(pred, truth, iou_threshold=0.05)
        packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred, metadata_bits=160)
        rows.append(
            {
                "threshold": threshold,
                **asdict(metrics),
                "semantic_packet_bits_per_frame": semantic_packet_bits(packet),
                "semantic_packet_bps": BitBudget("semantic_packet", semantic_packet_bits(packet), frame.duration_s).bitrate_bps,
                "raw_iq_32bit_iq_bps": BitBudget(
                    "raw_iq_32bit_iq", iq_payload_bits(frame.n_samples, 32, 32), frame.duration_s
                ).bitrate_bps,
                "raw_iq_12bit_iq_bps": BitBudget(
                    "raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s
                ).bitrate_bps,
                "stft_8bit_bps": BitBudget(
                    "stft_8bit", stft_payload_bits(stft.power_db.shape, 8), frame.duration_s
                ).bitrate_bps,
                "pred_boxes": [asdict(box) for box in pred],
            }
        )

    best = max(rows, key=lambda row: row["f1"])
    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tiny_cnn_full_frame_result.json"
    npy_path = out_dir / "tiny_cnn_full_frame_prob.npy"
    out_path.write_text(json.dumps({"device": str(device), "rows": rows, "best": best}, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(npy_path, prob.astype(np.float32))

    print(json.dumps({k: best[k] for k in best.keys() if k != "pred_boxes"}, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"wrote {npy_path}")


if __name__ == "__main__":
    main()
