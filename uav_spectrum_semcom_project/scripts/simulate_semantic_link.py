from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import hard_box_packet_bits, iq_payload_bits, soft_box_packet_bits, stft_payload_bits
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.raddet import iter_raddet_frames
from train_raddet_occupancy_mask import mask_to_boxes, match_boxes, yolo_to_xyxy


def preprocess_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr[None, None, :, :]


def evaluate_box_lists(predictions: list[list[np.ndarray]], truths: list[list[np.ndarray]], iou_threshold: float) -> dict[str, float]:
    tp = fp = fn = 0
    ious_all: list[float] = []
    for pred, truth in zip(predictions, truths):
        a, b, c, ious = match_boxes(pred, truth, iou_threshold=iou_threshold)
        tp += a
        fp += b
        fn += c
        ious_all.extend(ious)
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
    }


def repetition_ber(ber: float, repeats: int) -> float:
    if repeats <= 1:
        return ber
    # Majority-vote bit repetition. Only odd repeats are intended.
    threshold = repeats // 2 + 1
    prob = 0.0
    from math import comb

    for k in range(threshold, repeats + 1):
        prob += comb(repeats, k) * (ber**k) * ((1.0 - ber) ** (repeats - k))
    return prob


def repetition_packet_loss(packet_loss: float, repeats: int) -> float:
    return packet_loss**max(1, repeats)


def transmit_semantic_boxes(
    predictions: list[list[np.ndarray]],
    packet_loss: float,
    ber: float,
    scheme: str,
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], float]:
    delivered: list[list[np.ndarray]] = []
    total_bits = 0
    if scheme == "hard":
        repeats = 1
        per_box_bits = 8 + 4 * 16
        packet_bits_fn = hard_box_packet_bits
    elif scheme == "hard_rep3":
        repeats = 3
        per_box_bits = 8 + 4 * 16
        packet_bits_fn = hard_box_packet_bits
    elif scheme == "soft":
        repeats = 1
        per_box_bits = 4 * 16 + 8 + 11 * 8
        packet_bits_fn = lambda n: soft_box_packet_bits(n, n_classes=11)
    elif scheme == "soft_rep3":
        repeats = 3
        per_box_bits = 4 * 16 + 8 + 11 * 8
        packet_bits_fn = lambda n: soft_box_packet_bits(n, n_classes=11)
    else:
        raise ValueError(f"unknown semantic scheme {scheme}")

    eff_packet_loss = repetition_packet_loss(packet_loss, repeats)
    eff_ber = repetition_ber(ber, repeats)
    for boxes in predictions:
        total_bits += repeats * packet_bits_fn(len(boxes))
        if rng.random() < eff_packet_loss:
            delivered.append([])
            continue
        kept = []
        for box in boxes:
            corrupt_prob = 1.0 - (1.0 - eff_ber) ** per_box_bits
            if rng.random() >= corrupt_prob:
                kept.append(box)
        delivered.append(kept)
    return delivered, total_bits / max(len(predictions), 1)


def transmit_large_payload_gate(
    predictions: list[list[np.ndarray]],
    packet_loss: float,
    ber: float,
    payload_bits_per_frame: int,
    packet_bits: int,
    rng: np.random.Generator,
) -> tuple[list[list[np.ndarray]], float]:
    """A coarse baseline for spectrum/IQ forwarding.

    The receiver can run the same detector only if all packets for a frame
    arrive and no bit error occurs. This is intentionally conservative, but it
    highlights how large payloads are vulnerable without strong FEC/ARQ.
    """

    packets = int(np.ceil(payload_bits_per_frame / packet_bits))
    frame_success = ((1.0 - packet_loss) ** packets) * ((1.0 - ber) ** payload_bits_per_frame)
    delivered = [boxes if rng.random() < frame_success else [] for boxes in predictions]
    return delivered, float(payload_bits_per_frame)


def load_or_generate_predictions(args) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    cache_path = args.out_dir / "semantic_link_prediction_cache.json"
    if cache_path.exists() and not args.refresh_cache:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        predictions = [[np.array(box, dtype=np.float32) for box in frame] for frame in data["predictions"]]
        truths = [[np.array(box, dtype=np.float32) for box in frame] for frame in data["truths"]]
        return predictions, truths

    import torch

    cfg = json.loads(args.mask_result.read_text(encoding="utf-8"))
    threshold = float(cfg["threshold"])
    min_cells = int(cfg["min_cells"])
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    frames = iter_raddet_frames(
        args.root,
        args.split,
        read_metadata=False,
        default_sequence_length=1_000_000,
        max_frames=args.max_frames,
    )
    model = TinyOccupancyCNN(channels=16).to(device)
    model.load_state_dict(torch.load(args.mask_checkpoint, map_location=device))
    model.eval()

    predictions: list[list[np.ndarray]] = []
    truths: list[list[np.ndarray]] = []
    with torch.no_grad():
        for frame in frames:
            x = torch.tensor(preprocess_image(frame.image_path), dtype=torch.float32, device=device)
            prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            predictions.append(mask_to_boxes(prob >= threshold, min_cells=min_cells))
            truths.append(
                [
                    yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
                    for box in frame.boxes
                ]
            )
    serializable = {
        "predictions": [[box.tolist() for box in frame] for frame in predictions],
        "truths": [[box.tolist() for box in frame] for frame in truths],
    }
    cache_path.write_text(json.dumps(serializable, ensure_ascii=False) + "\n", encoding="utf-8")
    return predictions, truths


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate semantic communication links for RadDet box semantics.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-losses", type=str, default="0,0.01,0.03,0.05,0.1,0.2")
    parser.add_argument("--bers", type=str, default="0,1e-6,1e-5,1e-4,1e-3")
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions, truths = load_or_generate_predictions(args)
    iou_threshold = float(json.loads(args.mask_result.read_text(encoding="utf-8"))["iou_threshold"])
    baseline = evaluate_box_lists(predictions, truths, iou_threshold=iou_threshold)
    packet_losses = [float(x) for x in args.packet_losses.split(",") if x.strip()]
    bers = [float(x) for x in args.bers.split(",") if x.strip()]

    rows = []
    schemes = ["hard", "hard_rep3", "soft", "soft_rep3", "spectrogram8", "iq12"]
    raw_spectrogram_bits = stft_payload_bits((128, 128), 8)
    raw_iq_bits = iq_payload_bits(1_000_000, 12, 12)

    for packet_loss in packet_losses:
        for ber in bers:
            for scheme in schemes:
                metrics_accum = []
                bit_accum = []
                for trial in range(args.trials):
                    rng = np.random.default_rng(args.seed + trial + int(packet_loss * 1e6) + int(ber * 1e9))
                    if scheme in {"hard", "hard_rep3", "soft", "soft_rep3"}:
                        delivered, mean_bits = transmit_semantic_boxes(predictions, packet_loss, ber, scheme, rng)
                    elif scheme == "spectrogram8":
                        delivered, mean_bits = transmit_large_payload_gate(
                            predictions, packet_loss, ber, raw_spectrogram_bits, args.packet_bits, rng
                        )
                    elif scheme == "iq12":
                        delivered, mean_bits = transmit_large_payload_gate(
                            predictions, packet_loss, ber, raw_iq_bits, args.packet_bits, rng
                        )
                    else:
                        raise AssertionError(scheme)
                    m = evaluate_box_lists(delivered, truths, iou_threshold=iou_threshold)
                    metrics_accum.append(m)
                    bit_accum.append(mean_bits)
                mean = {
                    key: float(np.mean([m[key] for m in metrics_accum]))
                    for key in ["precision", "recall", "f1", "mean_iou", "tp", "fp", "fn"]
                }
                mean_bits = float(np.mean(bit_accum))
                rows.append(
                    {
                        "scheme": scheme,
                        "packet_loss": packet_loss,
                        "ber": ber,
                        "mean_bits_per_frame": mean_bits,
                        "compression_vs_spectrogram8": raw_spectrogram_bits / max(mean_bits, 1),
                        "compression_vs_iq12": raw_iq_bits / max(mean_bits, 1),
                        **mean,
                    }
                )

    csv_path = args.out_dir / "semantic_link_simulation.csv"
    json_path = args.out_dir / "semantic_link_simulation.json"
    md_path = args.out_dir / "semantic_link_simulation.md"
    fieldnames = [
        "scheme",
        "packet_loss",
        "ber",
        "mean_bits_per_frame",
        "compression_vs_spectrogram8",
        "compression_vs_iq12",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "tp",
        "fp",
        "fn",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "RadDet40k128HW001Tv2",
        "source_detector": "TinyOccupancyCNN",
        "split": args.split,
        "frames": len(predictions),
        "trials": args.trials,
        "packet_bits": args.packet_bits,
        "baseline_no_link": baseline,
        "rows": rows,
        "notes": [
            "semantic schemes drop frames under packet erasure and drop boxes under bit corruption",
            "rep3 uses simple majority-vote bit repetition and independent repeated packet transmission",
            "spectrogram8/iq12 are conservative large-payload forwarding gates without FEC or ARQ",
        ],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pick(pl: float, ber: float) -> list[dict]:
        return [row for row in rows if abs(row["packet_loss"] - pl) < 1e-12 and abs(row["ber"] - ber) < 1e-15]

    lines = [
        "# Semantic Link Simulation",
        "",
        "## Baseline detector before communication",
        "",
        f"- F1: {baseline['f1']:.4f}",
        f"- Precision: {baseline['precision']:.4f}",
        f"- Recall: {baseline['recall']:.4f}",
        f"- Mean IoU: {baseline['mean_iou']:.4f}",
        "",
        "## Representative operating points",
        "",
    ]
    for pl, ber in [(0.0, 0.0), (0.05, 1e-5), (0.1, 1e-4), (0.2, 1e-4)]:
        selected = pick(pl, ber)
        if not selected:
            continue
        lines.extend(
            [
                f"### packet_loss={pl:g}, BER={ber:g}",
                "",
                "| Scheme | F1 | Precision | Recall | bits/frame | vs 8-bit spectrogram | vs 12-bit IQ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['scheme']} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['mean_bits_per_frame']:.1f} | {row['compression_vs_spectrogram8']:.1f}x | "
                f"{row['compression_vs_iq12']:.1f}x |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This simulation turns the detector output into a communication object. "
            "Hard semantic packets have the smallest payload; repeated semantic packets trade a small payload increase for much better packet-erasure robustness. "
            "Large-payload spectrogram/IQ forwarding is modeled conservatively and becomes fragile without FEC/ARQ because each frame spans many packets.",
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
