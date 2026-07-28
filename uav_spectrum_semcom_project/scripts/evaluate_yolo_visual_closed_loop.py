from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.visdrone import VISDRONE_CLASS_NAMES, VisDroneBox, VisDroneFrame, iter_visdrone_frames
from evaluate_weak_visual_closed_loop import evaluate_visual_source, make_decision_args
from simulate_multimodal_decision_policy import load_spectrum_operating_points
from train_multimodal_dqn_policy import DQNPolicyNet
from train_raddet_occupancy_mask import match_boxes
from train_visdrone_roi_mask import truth_box_xyxy_norm


PRIORITY_YOLO_NAMES = ["pedestrian", "people", "car", "van", "truck", "bus", "motor"]
YOLO_TO_VISDRONE_CATEGORY = {
    0: 1,
    1: 2,
    2: 4,
    3: 5,
    4: 6,
    5: 9,
    6: 10,
}
TASK_RELEVANT_VISDRONE_CATEGORIES = set(YOLO_TO_VISDRONE_CATEGORY.values())


def load_dqn_model(path: Path, device: str) -> DQNPolicyNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("args", {}).get("hidden_dim", 64))
    model = DQNPolicyNet(state_dim=11, action_dim=5, hidden_dim=hidden_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def yolo_line_to_box(line: str, frame: VisDroneFrame, conf_threshold: float) -> VisDroneBox | None:
    parts = line.strip().split()
    if len(parts) not in {5, 6}:
        return None
    cls = int(float(parts[0]))
    if cls not in YOLO_TO_VISDRONE_CATEGORY:
        return None
    conf = float(parts[5]) if len(parts) == 6 else 1.0
    if conf < conf_threshold:
        return None
    x_center, y_center, width, height = [float(x) for x in parts[1:5]]
    x = max(0.0, (x_center - width / 2.0) * frame.width)
    y = max(0.0, (y_center - height / 2.0) * frame.height)
    w = max(0.0, min(width * frame.width, frame.width - x))
    h = max(0.0, min(height * frame.height, frame.height - y))
    return VisDroneBox(
        x=x,
        y=y,
        width=w,
        height=h,
        score=int(round(conf * 1000)),
        category=YOLO_TO_VISDRONE_CATEGORY[cls],
        truncation=0,
        occlusion=0,
    )


def prediction_path(prediction_dir: Path, frame: VisDroneFrame) -> Path:
    return prediction_dir / f"{frame.stem}.txt"


def frame_from_yolo_prediction(
    frame: VisDroneFrame,
    prediction_dir: Path,
    conf_threshold: float,
    max_boxes_per_frame: int = 0,
) -> tuple[VisDroneFrame, dict[str, float]]:
    path = prediction_path(prediction_dir, frame)
    pred_boxes: list[VisDroneBox] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            box = yolo_line_to_box(line, frame, conf_threshold)
            if box is not None:
                pred_boxes.append(box)
    # Keep a permissive confidence threshold for recall, then cap only the
    # lowest-confidence candidates.  This directly models a semantic-packet
    # budget without requiring a detector retraining run.
    pred_boxes.sort(key=lambda box: box.score, reverse=True)
    if max_boxes_per_frame > 0:
        pred_boxes = pred_boxes[:max_boxes_per_frame]

    pred_xyxy = [
        np.asarray(
            [
                box.x / frame.width,
                box.y / frame.height,
                (box.x + box.width) / frame.width,
                (box.y + box.height) / frame.height,
            ],
            dtype=np.float32,
        )
        for box in pred_boxes
    ]
    truth_xyxy = [
        truth_box_xyxy_norm(box, frame.width, frame.height)
        for box in frame.boxes
        if box.category in TASK_RELEVANT_VISDRONE_CATEGORIES
    ]
    tp, fp, fn, ious = match_boxes(pred_xyxy, truth_xyxy, iou_threshold=0.1)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    predicted = replace(frame, boxes=pred_boxes, semantic_box_bits=hard_box_packet_bits(len(pred_boxes), label_bits=8, coord_bits=16))
    return predicted, {
        "pred_box_count": float(len(pred_boxes)),
        "truth_box_count": float(len(truth_xyxy)),
        "semantic_quality": float(recall),
        "precision": float(precision),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def write_report(rows: list[dict[str, float | str]], out_path: Path, args: argparse.Namespace) -> None:
    selected = [
        row
        for row in rows
        if row["spectrum_scheme"] == "semantic_hard_rep3"
        and abs(float(row["packet_loss"]) - args.report_packet_loss) < 1e-12
    ]
    lines = [
        "# YOLO Visual Semantics Closed-Loop Evaluation",
        "",
        "## Purpose",
        "",
        "This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. "
        "It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.",
        "",
        f"- Prediction directory: `{args.prediction_dir}`",
        f"- Confidence threshold: {args.conf_threshold:g}",
        f"- Maximum semantic boxes per frame: {args.max_boxes_per_frame if args.max_boxes_per_frame > 0 else 'unlimited'}",
        "",
        "## Representative result",
        "",
        "| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(selected, key=lambda r: -float(r["mean_utility"])):
        action_names = ["summary_only", "semantic_only", "semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"]
        actions = ", ".join(
            f"{action}:{float(row[f'action_{action}_ratio']):.2f}"
            for action in action_names
            if float(row[f"action_{action}_ratio"]) > 0.01
        )
        lines.append(
            f"| {row['policy']} | {float(row['mean_bits_per_frame']):.1f} | {float(row['mean_utility']):.4f} | "
            f"{float(row['semantic_score']):.4f} | {float(row['priority_detail_score']):.4f} | "
            f"{float(row['mean_visual_semantic_quality']):.4f} | {float(row['mean_pred_boxes']):.1f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. "
            "Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. "
            "The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-format visual predictions in the spectrum-aware DQN closed loop.")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--spectrum-summary", type=Path, default=PROJECT_DIR / "results" / "phase1" / "resource_optimization_simulation.json")
    parser.add_argument("--dqn-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "multimodal_dqn_policy.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.15)
    parser.add_argument(
        "--max-boxes-per-frame",
        type=int,
        default=0,
        help="Keep at most this many highest-confidence semantic boxes per frame; 0 disables the cap.",
    )
    parser.add_argument("--ber", type=float, default=1e-4)
    parser.add_argument("--spectrum-schemes", type=str, default="semantic_hard_rep3,spectrogram8_partial,random")
    parser.add_argument("--policies", type=str, default="semantic_only,spectrum_rule,dqn,oracle_policy")
    parser.add_argument("--min-priority-objects", type=int, default=10)
    parser.add_argument("--clean-high", type=float, default=0.90)
    parser.add_argument("--clean-mid", type=float, default=0.78)
    parser.add_argument("--roi-heavy-threshold", type=float, default=0.18)
    parser.add_argument("--lowres-fraction", type=float, default=0.04)
    parser.add_argument("--packet-bits", type=int, default=1024)
    parser.add_argument("--bit-cost-per-mbit", type=float, default=0.12)
    parser.add_argument("--priority-weight", type=float, default=1.8)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--detail-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=944)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report-packet-loss", type=float, default=0.2)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = iter_visdrone_frames(args.visdrone_root, max_frames=args.max_frames)
    observed_frames = []
    qualities = []
    for frame in frames:
        observed, quality = frame_from_yolo_prediction(
            frame,
            args.prediction_dir,
            args.conf_threshold,
            args.max_boxes_per_frame,
        )
        observed_frames.append(observed)
        qualities.append(quality)

    scheme_filter = {x.strip() for x in args.spectrum_schemes.split(",") if x.strip()}
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    spectrum_rows = load_spectrum_operating_points(args.spectrum_summary, args.ber, scheme_filter)
    dqn_model = load_dqn_model(args.dqn_checkpoint, device)
    decision_args = make_decision_args(args)

    rows: list[dict[str, float | str]] = []
    for spectrum_row in spectrum_rows:
        rows.extend(
            evaluate_visual_source(
                "yolo_predictions",
                frames,
                observed_frames,
                qualities,
                spectrum_row,
                policies,
                dqn_model,
                decision_args,
                device,
            )
        )

    csv_path = args.out_dir / "yolo_visual_closed_loop.csv"
    json_path = args.out_dir / "yolo_visual_closed_loop.json"
    md_path = args.out_dir / "yolo_visual_closed_loop.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "prediction_dir": str(args.prediction_dir),
        "class_names": PRIORITY_YOLO_NAMES,
        "conf_threshold": args.conf_threshold,
        "max_boxes_per_frame": args.max_boxes_per_frame,
        "mean_visual_quality": float(np.mean([q["semantic_quality"] for q in qualities])) if qualities else 0.0,
        "mean_pred_boxes": float(np.mean([q["pred_box_count"] for q in qualities])) if qualities else 0.0,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(rows, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
