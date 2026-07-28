from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="2GB-GPU-safe YOLOv8n smoke training on the VisDrone priority YOLO dataset.")
    parser.add_argument("--data", type=Path, default=Path("uav_spectrum_semcom_project/data/processed/visdrone_yolo_priority/visdrone_uav_semantic.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--project", type=Path, default=Path("C:/yolo_visdrone_runs"))
    parser.add_argument("--name", type=str, default="yolov8n_priority_smoke_2gb")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Windows/Conda on this machine can load duplicate OpenMP runtimes.
    # This workaround is acceptable for smoke testing and avoids the libiomp5md.dll abort.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=args.amp,
        plots=args.plots,
        val=args.val,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
