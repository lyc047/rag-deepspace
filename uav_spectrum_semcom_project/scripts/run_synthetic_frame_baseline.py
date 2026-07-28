from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.baselines import connected_component_energy_detector
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits
from spectrum_semcom.data_io import load_npz_frame
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket


def _read_manifest(path: Path, split: str):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    return rows


def _aggregate(rows: list[dict]) -> dict:
    tp = sum(row["true_positive"] for row in rows)
    fp = sum(row["false_positive"] for row in rows)
    fn = sum(row["false_negative"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_frames": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_frame_f1": sum(row["f1"] for row in rows) / len(rows) if rows else 0.0,
        "mean_semantic_packet_bps": sum(row["semantic_packet_bps"] for row in rows) / len(rows) if rows else 0.0,
        "mean_raw_iq_12bit_iq_bps": sum(row["raw_iq_12bit_iq_bps"] for row in rows) / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traditional baseline on synthetic controlled IQ frames.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "synthetic_frame_manifest.csv")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--enhancement", default="freq_median")
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--min-cells", type=int, default=8)
    args = parser.parse_args()

    manifest_rows = _read_manifest(args.manifest, args.split)
    rows = []
    for row in manifest_rows:
        frame = load_npz_frame(PROJECT_DIR / row["path"])
        stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=args.n_fft, hop_length=args.hop_length)
        pred = connected_component_energy_detector(
            stft,
            threshold_sigma=args.sigma,
            enhancement=args.enhancement,
            min_cells=args.min_cells,
            max_boxes=64,
        )
        metrics = evaluate_detections(pred, frame.boxes, iou_threshold=0.05)
        packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred, metadata_bits=160)
        bits = semantic_packet_bits(packet)
        rows.append(
            {
                "frame_id": frame.frame_id,
                "split": row["split"],
                **asdict(metrics),
                "semantic_packet_bits_per_frame": bits,
                "semantic_packet_bps": BitBudget("semantic_packet", bits, frame.duration_s).bitrate_bps,
                "raw_iq_12bit_iq_bps": BitBudget(
                    "raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s
                ).bitrate_bps,
            }
        )

    aggregate = _aggregate(rows)
    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_frame_baseline_{args.split}.json"
    out_path.write_text(json.dumps({"args": vars(args) | {"manifest": str(args.manifest)}, "aggregate": aggregate, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

