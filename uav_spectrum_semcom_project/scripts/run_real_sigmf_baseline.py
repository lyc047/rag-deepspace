from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.baselines import connected_component_energy_detector, enhance_stft_for_detection
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits, stft_payload_bits
from spectrum_semcom.data_io import load_sigmf_frame
from spectrum_semcom.metrics import evaluate_detections, match_boxes
from spectrum_semcom.plotting import plot_stft_with_boxes
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket, SignalBox


DEFAULT_META = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sigmf_5g_short"
    / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
)


def _filter_truth(boxes: list[SignalBox], include_frames: bool) -> list[SignalBox]:
    if include_frames:
        return boxes
    return [box for box in boxes if not box.label.startswith("Frame") and box.label != "unknown"]


def _run_one(frame, stft, enhanced_power, sigma: float, min_cells: int, enhancement: str, include_frames: bool) -> dict:
    pred = connected_component_energy_detector(
        stft,
        threshold_sigma=sigma,
        enhancement=enhancement,
        enhanced_power=enhanced_power,
        min_cells=min_cells,
        min_time_bins=1,
        min_freq_bins=1,
        max_boxes=128,
        binary_opening=False,
    )
    truth = _filter_truth(frame.boxes, include_frames=include_frames)
    metrics = evaluate_detections(pred, truth, iou_threshold=0.05)
    matches = match_boxes(pred, truth, iou_threshold=0.05)
    matched_truth_labels = Counter(truth[truth_idx].label for _, truth_idx, _ in matches)
    truth_labels = Counter(box.label for box in truth)
    packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred, metadata_bits=160)
    payloads = {
        "raw_iq_32bit_iq_bps": BitBudget(
            "raw_iq_32bit_iq", iq_payload_bits(frame.n_samples, 32, 32), frame.duration_s
        ).bitrate_bps,
        "raw_iq_12bit_iq_bps": BitBudget(
            "raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s
        ).bitrate_bps,
        "stft_8bit_bps": BitBudget("stft_8bit", stft_payload_bits(stft.power_db.shape, 8), frame.duration_s).bitrate_bps,
        "semantic_packet_bps": BitBudget(
            "semantic_packet", semantic_packet_bits(packet), frame.duration_s
        ).bitrate_bps,
        "semantic_packet_bits_per_frame": semantic_packet_bits(packet),
    }
    return {
        "threshold_sigma": sigma,
        "min_cells": min_cells,
        "enhancement": enhancement,
        "n_fft": stft.n_fft,
        "hop_length": stft.hop_length,
        **asdict(metrics),
        **payloads,
        "truth_label_counts": dict(sorted(truth_labels.items())),
        "matched_truth_label_counts": dict(sorted(matched_truth_labels.items())),
        "pred_boxes": [asdict(box) for box in pred],
    }


def _boxes_from_dicts(items: list[dict]) -> list[SignalBox]:
    return [SignalBox(**item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate connected-component energy baseline on a real SigMF IQ file.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    parser.add_argument("--min-cells", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--enhancements",
        nargs="+",
        default=["raw", "freq_median", "time_median"],
    )
    parser.add_argument("--include-frames", action="store_true")
    args = parser.parse_args()

    frame = load_sigmf_frame(args.meta, max_samples=args.max_samples, normalize=True)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=args.n_fft, hop_length=args.hop_length)
    rows = []
    for enhancement in args.enhancements:
        enhanced_power = enhance_stft_for_detection(stft.power_db, mode=enhancement)
        for sigma in args.sigmas:
            for min_cells in args.min_cells:
                rows.append(_run_one(frame, stft, enhanced_power, sigma, min_cells, enhancement, args.include_frames))

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_sigmf_energy_baseline.json"
    csv_path = out_dir / "real_sigmf_energy_baseline_summary.csv"

    json_payload = {
        "frame_id": frame.frame_id,
        "n_samples": frame.n_samples,
        "duration_s": frame.duration_s,
        "sample_rate_hz": frame.sample_rate_hz,
        "center_freq_hz": frame.center_freq_hz,
        "truth_count_all": len(frame.boxes),
        "truth_count_eval": len(_filter_truth(frame.boxes, include_frames=args.include_frames)),
        "include_frames": args.include_frames,
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_keys = [key for key in rows[0].keys() if key != "pred_boxes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in summary_keys})

    best = max(rows, key=lambda row: row["f1"])
    plot_path = out_dir / "real_sigmf_energy_baseline_best.png"
    plot_stft_with_boxes(
        stft,
        truth=_filter_truth(frame.boxes, include_frames=args.include_frames),
        pred=_boxes_from_dicts(best["pred_boxes"]),
        output_path=plot_path,
        title=f"Real SigMF energy baseline, sigma={best['threshold_sigma']}, F1={best['f1']:.3f}",
    )
    print(json.dumps({key: best[key] for key in summary_keys}, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
