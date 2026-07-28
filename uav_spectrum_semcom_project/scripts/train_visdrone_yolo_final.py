"""Reproducible final-stage VisDrone training for a 2 GB GPU.

The defaults favor small-object recall (640-pixel inputs) and start from the
pretrained YOLOv8s checkpoint.  `batch=1` leaves a safety margin for the
MX450's 2 GB VRAM; use the same script with `--model yolov8n.pt` if a larger
model cannot train reliably on another machine.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Final VisDrone priority-detector fine-tuning on a 2 GB GPU.")
    parser.add_argument("--data", type=Path, default=Path("uav_spectrum_semcom_project/data/processed/visdrone_yolo_priority/visdrone_uav_semantic.yaml"))
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--project", type=Path, default=Path("C:/yolo_visdrone_runs"))
    parser.add_argument("--name", type=str, default="visdrone_yolo_final_2gb")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=944)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # Required by the Windows Conda environment used for this project.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=args.amp,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        plots=args.plots,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
