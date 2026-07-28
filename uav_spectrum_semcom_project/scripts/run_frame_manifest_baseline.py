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
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits, stft_payload_bits
from spectrum_semcom.frame_manifest import load_frame_manifest, load_iq_frame_from_manifest_row
from spectrum_semcom.metrics import evaluate_detections
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket


def _filter_truth(boxes):
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def _evaluate_one(row, args) -> dict:
    frame = load_iq_frame_from_manifest_row(row)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=args.n_fft, hop_length=args.hop_length)
    truth = _filter_truth(frame.boxes)
    pred = connected_component_energy_detector(
        stft,
        threshold_sigma=args.sigma,
        enhancement=args.enhancement,
        min_cells=args.min_cells,
        max_boxes=args.max_boxes,
    )
    metrics = evaluate_detections(pred, truth, iou_threshold=args.iou_threshold)
    packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred, metadata_bits=160)
    semantic_bits = semantic_packet_bits(packet)
    return {
        "frame_id": row.frame_id,
        "split": row.split,
        "frame_number": row.frame_number,
        "n_samples": frame.n_samples,
        "duration_s": frame.duration_s,
        **asdict(metrics),
        "semantic_packet_bits_per_frame": semantic_bits,
        "semantic_packet_bps": BitBudget("semantic_packet", semantic_bits, frame.duration_s).bitrate_bps,
        "raw_iq_32bit_iq_bps": BitBudget(
            "raw_iq_32bit_iq", iq_payload_bits(frame.n_samples, 32, 32), frame.duration_s
        ).bitrate_bps,
        "raw_iq_12bit_iq_bps": BitBudget(
            "raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s
        ).bitrate_bps,
        "stft_8bit_bps": BitBudget("stft_8bit", stft_payload_bits(stft.power_db.shape, 8), frame.duration_s).bitrate_bps,
    }


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
    parser = argparse.ArgumentParser(description="Evaluate traditional baseline on frame-level manifest splits.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "frame_manifest.csv")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--enhancement", default="freq_median")
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    args = parser.parse_args()

    manifest_rows = load_frame_manifest(args.manifest, project_dir=PROJECT_DIR)
    if args.split != "all":
        manifest_rows = [row for row in manifest_rows if row.split == args.split]
    rows = [_evaluate_one(row, args) for row in manifest_rows]
    aggregate = _aggregate(rows)

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.split
    json_path = out_dir / f"frame_manifest_baseline_{suffix}.json"
    csv_path = out_dir / f"frame_manifest_baseline_{suffix}.csv"
    args_json = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    json_path.write_text(json.dumps({"args": args_json, "aggregate": aggregate, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
