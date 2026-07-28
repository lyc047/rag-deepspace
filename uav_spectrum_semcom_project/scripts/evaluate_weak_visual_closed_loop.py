from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from spectrum_semcom.bit_budget import hard_box_packet_bits
from spectrum_semcom.models import TinyOccupancyCNN
from spectrum_semcom.visdrone import VisDroneBox, VisDroneFrame, iter_visdrone_frames
from simulate_multimodal_decision_policy import (
    ACTION_BITS,
    action_bits,
    delivered_probability,
    load_spectrum_operating_points,
    policy_action,
)
from simulate_multimodal_policy import PRIORITY_CLASSES, visual_priority
from train_multimodal_dqn_policy import DQNPolicyNet, state_features
from train_raddet_occupancy_mask import mask_to_boxes, match_boxes
from train_visdrone_roi_mask import truth_box_xyxy_norm


ACTIONS = list(ACTION_BITS.keys())


def infer_channels_from_checkpoint(state: dict[str, torch.Tensor]) -> int:
    first_weight = state.get("net.0.weight")
    if first_weight is None:
        raise ValueError("Cannot infer TinyOccupancyCNN channels from checkpoint")
    return int(first_weight.shape[0])


def load_roi_model(checkpoint_path: Path, device: str) -> TinyOccupancyCNN:
    state = torch.load(checkpoint_path, map_location="cpu")
    channels = infer_channels_from_checkpoint(state)
    model = TinyOccupancyCNN(channels=channels)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_dqn_model(path: Path, device: str) -> DQNPolicyNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("args", {}).get("hidden_dim", 64))
    model = DQNPolicyNet(state_dim=11, action_dim=len(ACTIONS), hidden_dim=hidden_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(frame: VisDroneFrame, image_size: int) -> torch.Tensor:
    img = Image.open(frame.image_path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return torch.tensor(arr[None, None, :, :], dtype=torch.float32)


def predicted_frame_from_roi_mask(
    frame: VisDroneFrame,
    model: TinyOccupancyCNN,
    device: str,
    image_size: int,
    threshold: float,
    min_cells: int,
    max_boxes: int,
    category: int,
) -> tuple[VisDroneFrame, dict[str, float]]:
    with torch.no_grad():
        x = preprocess_image(frame, image_size).to(device)
        prob = torch.sigmoid(model(x)).detach().cpu().numpy()[0, 0]
    boxes_norm = mask_to_boxes(prob >= threshold, min_cells=min_cells, max_boxes=max_boxes)
    pred_boxes: list[VisDroneBox] = []
    for box in boxes_norm:
        x0, y0, x1, y1 = box
        pred_boxes.append(
            VisDroneBox(
                x=float(x0 * frame.width),
                y=float(y0 * frame.height),
                width=float(max(0.0, (x1 - x0) * frame.width)),
                height=float(max(0.0, (y1 - y0) * frame.height)),
                score=1,
                category=category,
                truncation=0,
                occlusion=0,
            )
        )
    truth_boxes = [truth_box_xyxy_norm(box, frame.width, frame.height) for box in frame.boxes]
    tp, fp, fn, ious = match_boxes(boxes_norm, truth_boxes, iou_threshold=0.1)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    pred = replace(frame, boxes=pred_boxes, semantic_box_bits=hard_box_packet_bits(len(pred_boxes), label_bits=8, coord_bits=16))
    return pred, {
        "pred_box_count": float(len(pred_boxes)),
        "truth_box_count": float(len(frame.boxes)),
        "semantic_quality": float(recall),
        "precision": float(precision),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def make_decision_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        packet_bits=args.packet_bits,
        lowres_fraction=args.lowres_fraction,
        bit_cost_per_mbit=args.bit_cost_per_mbit,
        priority_weight=args.priority_weight,
        detail_weight=args.detail_weight,
        semantic_weight=args.semantic_weight,
        min_priority_objects=args.min_priority_objects,
        clean_high=args.clean_high,
        clean_mid=args.clean_mid,
        roi_heavy_threshold=args.roi_heavy_threshold,
        seed=args.seed,
    )


def select_dqn_action(model: DQNPolicyNet, observed: VisDroneFrame, spectrum_row: dict, args: argparse.Namespace, device: str) -> str:
    state = state_features(observed, spectrum_row, args)
    with torch.no_grad():
        q = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return ACTIONS[int(torch.argmax(q, dim=1).item())]


def true_aware_score(
    true_frame: VisDroneFrame,
    observed_frame: VisDroneFrame,
    semantic_quality: float,
    action: str,
    clean_rate: float,
    packet_loss: float,
    ber: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    bits = action_bits(observed_frame, action, args.lowres_fraction)
    p_deliver = delivered_probability(bits, clean_rate, packet_loss, ber, args.packet_bits)
    high_priority = visual_priority(true_frame, args.min_priority_objects)
    priority_count = sum(1 for box in true_frame.boxes if box.category in PRIORITY_CLASSES)

    semantic_value = 0.0
    detail_value = 0.0
    if action in {"semantic_only", "semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        semantic_value = args.semantic_weight * semantic_quality
    if action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        detail_value = args.detail_weight * semantic_quality
    if action == "summary_only":
        semantic_value = 0.15 * args.semantic_weight

    task_value = semantic_value + detail_value
    if high_priority:
        task_value *= args.priority_weight
    if action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"}:
        task_value *= min(1.4, 1.0 + priority_count / 80.0)
    expected_task = p_deliver * task_value
    utility = expected_task - args.bit_cost_per_mbit * (bits / 1e6)
    return {
        "bits": float(bits),
        "delivery_probability": float(p_deliver),
        "expected_task": float(expected_task),
        "utility": float(utility),
        "semantic_score": float(p_deliver * semantic_quality if action != "summary_only" else 0.15 * p_deliver),
        "priority_detail_score": float(p_deliver * semantic_quality if high_priority and action in {"semantic_plus_roi", "lowres_plus_semantic", "jpeg_full"} else 0.0),
    }


def evaluate_visual_source(
    name: str,
    true_frames: list[VisDroneFrame],
    observed_frames: list[VisDroneFrame],
    visual_quality: list[dict[str, float]],
    spectrum_row: dict,
    policies: list[str],
    dqn_model: DQNPolicyNet,
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    clean_rate = float(spectrum_row["clean_channel_rate"])
    packet_loss = float(spectrum_row["packet_loss"])
    ber = float(spectrum_row["ber"])
    for policy in policies:
        action_counts = {action: 0 for action in ACTIONS}
        scores = []
        qualities = []
        for true_frame, observed_frame, quality in zip(true_frames, observed_frames, visual_quality):
            if policy == "dqn":
                action = select_dqn_action(dqn_model, observed_frame, spectrum_row, args, device)
            elif policy == "oracle_policy":
                # True-scene upper bound: action is selected from the real visual state,
                # and semantic quality is one because this uses annotation semantics.
                best_action = None
                best_utility = -1e9
                for candidate in ACTIONS:
                    score = true_aware_score(true_frame, true_frame, 1.0, candidate, clean_rate, packet_loss, ber, args)
                    if score["utility"] > best_utility:
                        best_utility = score["utility"]
                        best_action = candidate
                action = str(best_action)
                observed_for_bits = true_frame
                semantic_quality = 1.0
            else:
                action = policy_action(policy, observed_frame, clean_rate, packet_loss, ber, args)
                observed_for_bits = observed_frame
                semantic_quality = float(quality["semantic_quality"])
            if policy != "oracle_policy":
                observed_for_bits = observed_frame
                semantic_quality = float(quality["semantic_quality"])
            action_counts[action] += 1
            scores.append(true_aware_score(true_frame, observed_for_bits, semantic_quality, action, clean_rate, packet_loss, ber, args))
            qualities.append(quality)

        mean_bits = float(np.mean([score["bits"] for score in scores]))
        row = {
            "visual_source": name,
            "spectrum_scheme": spectrum_row["scheme"],
            "packet_loss": packet_loss,
            "clean_rate": clean_rate,
            "policy": policy,
            "frames": len(true_frames),
            "mean_bits_per_frame": mean_bits,
            "mean_utility": float(np.mean([score["utility"] for score in scores])),
            "semantic_score": float(np.mean([score["semantic_score"] for score in scores])),
            "priority_detail_score": float(np.mean([score["priority_detail_score"] for score in scores])),
            "mean_visual_semantic_quality": float(np.mean([q["semantic_quality"] for q in qualities])),
            "mean_pred_boxes": float(np.mean([q["pred_box_count"] for q in qualities])),
            **{f"action_{action}_ratio": action_counts[action] / len(true_frames) for action in ACTIONS},
        }
        rows.append(row)
    return rows


def write_report(rows: list[dict[str, float | str]], out_path: Path, args: argparse.Namespace) -> None:
    selected = [
        row
        for row in rows
        if row["spectrum_scheme"] == "semantic_hard_rep3"
        and abs(float(row["packet_loss"]) - args.report_packet_loss) < 1e-12
    ]
    lines = [
        "# Weak Visual Semantics in the Closed Loop",
        "",
        "## Purpose",
        "",
        "This experiment keeps the current lightweight visual ROI-mask model and plugs its predicted visual semantics into the spectrum-aware DQN decision loop. "
        "It is a hardware-safe baseline for the current 2GB-GPU machine; stronger YOLO-style detectors can replace only the visual front-end later.",
        "",
        "## Representative closed-loop result",
        "",
        f"Spectrum scheme: `semantic_hard_rep3`, packet loss: {args.report_packet_loss:g}.",
        "",
        "| Visual source | Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(selected, key=lambda r: (str(r["visual_source"]), -float(r["mean_utility"]))):
        actions = ", ".join(
            f"{action}:{float(row[f'action_{action}_ratio']):.2f}"
            for action in ACTIONS
            if float(row[f"action_{action}_ratio"]) > 0.01
        )
        lines.append(
            f"| {row['visual_source']} | {row['policy']} | {float(row['mean_bits_per_frame']):.1f} | "
            f"{float(row['mean_utility']):.4f} | {float(row['semantic_score']):.4f} | "
            f"{float(row['priority_detail_score']):.4f} | {float(row['mean_visual_semantic_quality']):.4f} | "
            f"{float(row['mean_pred_boxes']):.1f} | {actions} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `gt_annotations` is the visual-semantic upper bound and should not be claimed as a deployed detector.",
            "- `roi_mask_128_class_agnostic` is the current safest real visual front-end: it predicts ROI boxes but no reliable category, so semantic quality and task utility are limited.",
            "- `roi_mask_128_priority_proxy` treats predicted ROI boxes as priority-like objects. It is not a final model; it estimates what happens if the weak ROI detector is used as a coarse priority trigger.",
            "- The result makes the current bottleneck explicit: the system closed loop works, but visual semantic quality is now the limiting factor.",
            "",
            "## Hardware note",
            "",
            "The current ROI-mask model is a tiny CNN and is safe for a 2GB MX450 GPU. A 256x256 retraining run completed without memory overflow but did not improve F1, so the 128x128 checkpoint remains the current weak visual baseline.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weak visual semantics inside the spectrum-aware multimodal closed loop.")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_DIR / "data" / "raw" / "visdrone" / "VisDrone2019-DET-val")
    parser.add_argument("--spectrum-summary", type=Path, default=PROJECT_DIR / "results" / "phase1" / "resource_optimization_simulation.json")
    parser.add_argument("--dqn-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "multimodal_dqn_policy.pt")
    parser.add_argument("--roi128-checkpoint", type=Path, default=PROJECT_DIR / "results" / "phase1" / "visdrone_roi_mask.pt")
    parser.add_argument("--roi128-result", type=Path, default=PROJECT_DIR / "results" / "phase1" / "visdrone_roi_mask_result.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "phase1")
    parser.add_argument("--max-frames", type=int, default=None)
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
    scheme_filter = {x.strip() for x in args.spectrum_schemes.split(",") if x.strip()}
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    spectrum_rows = load_spectrum_operating_points(args.spectrum_summary, args.ber, scheme_filter)
    dqn_model = load_dqn_model(args.dqn_checkpoint, device)

    roi_meta = json.loads(args.roi128_result.read_text(encoding="utf-8"))
    roi_threshold = float(roi_meta["threshold"])
    roi_image_size = int(roi_meta["image_size"])
    roi_min_cells = int(roi_meta["min_cells"])
    roi_max_boxes = int(roi_meta["max_boxes"])
    roi_model = load_roi_model(args.roi128_checkpoint, device)

    gt_quality = [
        {
            "pred_box_count": float(len(frame.boxes)),
            "truth_box_count": float(len(frame.boxes)),
            "semantic_quality": 1.0,
            "precision": 1.0,
            "mean_iou": 1.0,
        }
        for frame in frames
    ]
    roi_class_frames = []
    roi_priority_frames = []
    roi_quality = []
    for frame in frames:
        class_frame, quality = predicted_frame_from_roi_mask(
            frame,
            roi_model,
            device,
            roi_image_size,
            roi_threshold,
            roi_min_cells,
            roi_max_boxes,
            category=11,
        )
        priority_frame = replace(class_frame, boxes=[replace(box, category=4) for box in class_frame.boxes])
        roi_class_frames.append(class_frame)
        roi_priority_frames.append(priority_frame)
        roi_quality.append(quality)

    visual_sources = [
        ("gt_annotations", frames, gt_quality),
        ("roi_mask_128_class_agnostic", roi_class_frames, roi_quality),
        ("roi_mask_128_priority_proxy", roi_priority_frames, roi_quality),
    ]
    rows: list[dict[str, float | str]] = []
    for spectrum_row in spectrum_rows:
        for name, observed_frames, quality in visual_sources:
            rows.extend(evaluate_visual_source(name, frames, observed_frames, quality, spectrum_row, policies, dqn_model, args, device))

    csv_path = args.out_dir / "weak_visual_closed_loop.csv"
    json_path = args.out_dir / "weak_visual_closed_loop.json"
    md_path = args.out_dir / "weak_visual_closed_loop.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "roi128_result": roi_meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(rows, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
