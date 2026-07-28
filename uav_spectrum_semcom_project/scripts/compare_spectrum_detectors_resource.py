from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from simulate_resource_optimization import aggregate_rows, load_or_generate_predictions, run_trial
from spectrum_semcom.models import TinyGridDetector16CNN
from spectrum_semcom.raddet import iter_raddet_frames
from train_raddet_grid_detector import decode_predictions
from train_raddet_occupancy_mask import yolo_to_xyxy


def preprocess_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr[None, :, :]


def load_grid16_predictions(args: argparse.Namespace) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    cache = args.out_dir / "grid16_prediction_cache.json"
    if cache.exists() and not args.refresh_grid_cache:
        data = json.loads(cache.read_text(encoding="utf-8"))
        predictions = [[np.array(box, dtype=np.float32) for box in frame] for frame in data["predictions"]]
        truths = [[np.array(box, dtype=np.float32) for box in frame] for frame in data["truths"]]
        return predictions, truths

    import torch

    cfg = json.loads(args.grid16_result.read_text(encoding="utf-8"))
    threshold = float(cfg["threshold"])
    model = TinyGridDetector16CNN(n_classes=11, channels=20).to(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(args.grid16_checkpoint, map_location=device))
    model.eval()
    frames = iter_raddet_frames(args.root, "test", read_metadata=False, default_sequence_length=1_000_000, max_frames=args.max_frames)
    predictions: list[list[np.ndarray]] = []
    truths: list[list[np.ndarray]] = []
    with torch.no_grad():
        for frame in frames:
            x = torch.tensor(preprocess_image(frame.image_path)[None], dtype=torch.float32, device=device)
            obj_logit, box_pred, cls_logit = model(x)
            pred_boxes, _, _ = decode_predictions(
                obj_logit,
                box_pred,
                cls_logit,
                threshold=threshold,
                nms_iou=float(cfg["nms_iou"]),
                max_boxes=int(cfg["max_boxes"]),
            )
            predictions.append(pred_boxes[0])
            truths.append(
                [
                    yolo_to_xyxy(np.array([box.x_center, box.y_center, box.width, box.height], dtype=np.float32))
                    for box in frame.boxes
                ]
            )
    cache.write_text(
        json.dumps(
            {
                "predictions": [[box.tolist() for box in frame] for frame in predictions],
                "truths": [[box.tolist() for box in frame] for frame in truths],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions, truths


def run_resource(predictions: list[list[np.ndarray]], truths: list[list[np.ndarray]], args: argparse.Namespace) -> list[dict]:
    run_args = SimpleNamespace(
        seed=args.seed,
        n_channels=args.n_channels,
        demand_channels=args.demand_channels,
        channel_axis=args.channel_axis,
        clean_threshold=args.clean_threshold,
        low_interference_threshold=args.low_interference_threshold,
        switch_cost=args.switch_cost,
        decision_steps=args.decision_steps,
        uavs_per_step=args.uavs_per_step,
        packet_bits=args.packet_bits,
        large_fec_overhead=args.large_fec_overhead,
        large_min_recovery_fraction=args.large_min_recovery_fraction,
    )
    detailed = []
    for trial in range(args.trials):
        detailed.extend(run_trial(predictions, truths, args.packet_loss, args.ber, run_args, trial))
    return aggregate_rows(detailed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mask and lightweight YOLO-style spectrum detectors in resource optimization.")
    parser.add_argument("--root", type=Path, default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2")
    parser.add_argument("--mask-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask_result.json")
    parser.add_argument("--mask-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_occupancy_mask.pt")
    parser.add_argument("--grid16-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_grid16_detector_result.json")
    parser.add_argument("--grid16-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "raddet_grid16_detector.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--packet-loss", type=float, default=0.2)
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--decision-steps", type=int, default=600)
    parser.add_argument("--uavs-per-step", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--demand-channels", type=int, default=2)
    parser.add_argument("--channel-axis", type=str, default="y")
    parser.add_argument("--clean-threshold", type=float, default=0.02)
    parser.add_argument("--low-interference-threshold", type=float, default=0.10)
    parser.add_argument("--switch-cost", type=float, default=0.005)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--large-fec-overhead", type=float, default=1.25)
    parser.add_argument("--large-min-recovery-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=5151)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-grid-cache", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_predictions, mask_truths = load_or_generate_predictions(args)
    grid_predictions, grid_truths = load_grid16_predictions(args)

    detector_rows = []
    for detector, predictions, truths in [
        ("mask_baseline", mask_predictions, mask_truths),
        ("grid16_yolo_style", grid_predictions, grid_truths),
    ]:
        rows = run_resource(predictions, truths, args)
        for row in rows:
            row["detector"] = detector
            detector_rows.append(row)

    csv_path = args.out_dir / "spectrum_detector_resource_comparison.csv"
    json_path = args.out_dir / "spectrum_detector_resource_comparison.json"
    md_path = args.out_dir / "spectrum_detector_resource_comparison.md"
    fieldnames = ["detector", *[k for k in detector_rows[0].keys() if k != "detector"]]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detector_rows)
    summary = {"packet_loss": args.packet_loss, "ber": args.ber, "rows": detector_rows}
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Spectrum Detector Resource Comparison",
        "",
        f"- Packet loss: {args.packet_loss:g}",
        f"- BER: {args.ber:g}",
        f"- UAV observations per decision: {args.uavs_per_step}",
        "",
        "| Detector | Scheme | Clean rate | Net utility | Regret | bits/decision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    keep_schemes = {"semantic_hard", "semantic_hard_rep3", "semantic_soft_rep3", "spectrogram8_partial", "random", "oracle"}
    for row in detector_rows:
        if row["scheme"] not in keep_schemes:
            continue
        lines.append(
            f"| {row['detector']} | {row['scheme']} | {row['clean_channel_rate']:.4f} | "
            f"{row['mean_net_utility']:.4f} | {row['mean_regret_occupancy']:.4f} | "
            f"{row['mean_bits_per_frame']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This compares whether the improved lightweight YOLO-style detector changes the downstream resource decision, not only the detection F1.",
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
