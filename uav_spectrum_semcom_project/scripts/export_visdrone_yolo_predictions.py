from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reproducible YOLO predictions for VisDrone closed-loop evaluation.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("uav_spectrum_semcom_project/data/raw/visdrone/VisDrone2019-DET-val/images"),
    )
    parser.add_argument("--project", type=Path, default=Path("C:/yolo_visdrone_runs"))
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
