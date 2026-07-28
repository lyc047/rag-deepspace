from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.bit_budget import hard_box_packet_bits, packet_header_bits
from spectrum_semcom.visdrone import VisDroneFrame, iter_visdrone_frames


PRIORITY_CLASSES = {
    1,  # pedestrian
    2,  # people
    4,  # car
    5,  # van
    6,  # truck
    9,  # bus
    10,  # motor
}


def visual_priority(frame: VisDroneFrame, min_priority_objects: int) -> bool:
    count = sum(1 for box in frame.boxes if box.category in PRIORITY_CLASSES)
    return count >= min_priority_objects


def roi_area_fraction(frame: VisDroneFrame) -> float:
    if frame.width <= 0 or frame.height <= 0:
        return 0.0
    total = 0.0
    image_area = frame.width * frame.height
    for box in frame.boxes:
        total += max(0.0, box.width) * max(0.0, box.height)
    return min(1.0, total / image_area)


def estimate_roi_bits(frame: VisDroneFrame, overhead: float = 1.2, max_fraction: float = 0.75) -> float:
    fraction = min(max_fraction, overhead * roi_area_fraction(frame))
    return frame.jpeg_bits * fraction


def load_spectrum_rows(path: Path, packet_loss: float, ber: float) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in data["rows"]:
        if abs(float(row["packet_loss"]) - packet_loss) < 1e-12 and abs(float(row["ber"]) - ber) < 1e-15:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No spectrum rows found for packet_loss={packet_loss}, BER={ber} in {path}")
    return rows


def simulate_for_scheme(
    frames: list[VisDroneFrame],
    clean_rate: float,
    rng: np.random.Generator,
    min_priority_objects: int,
) -> dict[str, float]:
    full_jpeg_bits = []
    semantic_only_bits = []
    adaptive_bits = []
    adaptive_detail = []
    adaptive_semantic_success = []
    priority_flags = []

    for frame in frames:
        high_priority = visual_priority(frame, min_priority_objects)
        priority_flags.append(high_priority)
        is_clean = rng.random() < clean_rate
        semantic_bits = frame.semantic_box_bits
        summary_bits = packet_header_bits(metadata_bits=128)
        roi_bits = estimate_roi_bits(frame)

        full_jpeg_bits.append(frame.jpeg_bits)
        semantic_only_bits.append(semantic_bits)

        if is_clean:
            adaptive_bits.append(semantic_bits + roi_bits)
            adaptive_detail.append(1.0 if high_priority else 0.0)
            adaptive_semantic_success.append(1.0)
        elif high_priority:
            adaptive_bits.append(semantic_bits)
            adaptive_detail.append(0.0)
            adaptive_semantic_success.append(1.0)
        else:
            adaptive_bits.append(summary_bits)
            adaptive_detail.append(0.0)
            adaptive_semantic_success.append(1.0)

    priority_count = max(sum(priority_flags), 1)
    mean_full = float(np.mean(full_jpeg_bits))
    mean_semantic = float(np.mean(semantic_only_bits))
    mean_adaptive = float(np.mean(adaptive_bits))
    return {
        "clean_rate": clean_rate,
        "priority_frame_ratio": float(np.mean(priority_flags)),
        "mean_full_jpeg_bits": mean_full,
        "mean_semantic_only_bits": mean_semantic,
        "mean_adaptive_bits": mean_adaptive,
        "adaptive_vs_full_jpeg_ratio": mean_full / max(mean_adaptive, 1.0),
        "semantic_only_vs_full_jpeg_ratio": mean_full / max(mean_semantic, 1.0),
        "priority_detail_rate": float(sum(adaptive_detail) / priority_count),
        "priority_detail_per_mbit": float((sum(adaptive_detail) / priority_count) / (mean_adaptive / 1e6)),
        "semantic_success_rate": float(np.mean(adaptive_semantic_success)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a first spectrum-aware UAV visual semantic policy.")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--spectrum-summary", type=Path, default=PROJECT_DIR / "results" / "phase1" / "resource_optimization_simulation.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--packet-loss", type=float, default=0.2)
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-priority-objects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=707)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_visdrone_frames(args.visdrone_root, max_frames=args.max_frames)
    spectrum_rows = load_spectrum_rows(args.spectrum_summary, args.packet_loss, args.ber)

    rows = []
    for row in spectrum_rows:
        scheme = row["scheme"]
        if scheme == "oracle":
            continue
        stable_offset = int(hashlib.sha256(scheme.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + stable_offset % 100_000)
        result = simulate_for_scheme(
            frames,
            clean_rate=float(row["clean_channel_rate"]),
            rng=rng,
            min_priority_objects=args.min_priority_objects,
        )
        rows.append(
            {
                "spectrum_scheme": scheme,
                "packet_loss": args.packet_loss,
                "ber": args.ber,
                "frames": len(frames),
                **result,
            }
        )

    csv_path = args.out_dir / "multimodal_policy_simulation.csv"
    json_path = args.out_dir / "multimodal_policy_simulation.json"
    md_path = args.out_dir / "multimodal_policy_simulation.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "visual_dataset": "VisDrone2019-DET-val",
        "spectrum_source": str(args.spectrum_summary),
        "packet_loss": args.packet_loss,
        "ber": args.ber,
        "frames": len(frames),
        "min_priority_objects": args.min_priority_objects,
        "notes": [
            "This is an initial policy-level multimodal simulation using ground-truth visual boxes.",
            "Adaptive policy sends visual semantics for high-priority frames under poor spectrum and sends semantic plus estimated ROI image payload under clean spectrum.",
            "Clean spectrum probability is taken from the spectrum resource-optimization simulation.",
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Spectrum-Aware UAV Visual Semantic Policy Simulation",
        "",
        "## Setup",
        "",
        f"- Visual dataset: VisDrone2019-DET-val, {len(frames)} frames.",
        f"- Spectrum source: `{args.spectrum_summary}`",
        f"- Packet loss: {args.packet_loss:g}",
        f"- BER: {args.ber:g}",
        f"- High-priority visual frame: at least {args.min_priority_objects} priority-class objects.",
        "",
        "## Results",
        "",
        "| Spectrum scheme | Clean rate | Full JPEG bits/frame | Semantic-only bits/frame | Adaptive bits/frame | Adaptive/full ratio | Priority detail rate | Detail/Mbit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["mean_adaptive_bits"]):
        lines.append(
            f"| {row['spectrum_scheme']} | {row['clean_rate']:.4f} | "
            f"{row['mean_full_jpeg_bits']:.1f} | {row['mean_semantic_only_bits']:.1f} | "
            f"{row['mean_adaptive_bits']:.1f} | {row['adaptive_vs_full_jpeg_ratio']:.1f}x | "
            f"{row['priority_detail_rate']:.4f} | {row['priority_detail_per_mbit']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the first bridge from the completed spectrum system to UAV visual semantic transmission. "
            "The adaptive policy uses spectrum resource quality to decide whether to transmit ROI image detail or only object-level visual semantics. "
            "The result should be treated as a policy-level feasibility check, not yet as a final trained multimodal semantic communication system.",
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
