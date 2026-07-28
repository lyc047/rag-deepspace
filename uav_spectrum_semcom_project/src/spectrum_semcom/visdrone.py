from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .bit_budget import hard_box_packet_bits


VISDRONE_CLASS_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


@dataclass(frozen=True)
class VisDroneBox:
    x: float
    y: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int

    @property
    def class_name(self) -> str:
        return VISDRONE_CLASS_NAMES.get(self.category, f"class_{self.category}")

    @property
    def is_ignored(self) -> bool:
        return self.category == 0


@dataclass(frozen=True)
class VisDroneFrame:
    stem: str
    image_path: Path
    annotation_path: Path
    width: int
    height: int
    boxes: list[VisDroneBox]
    jpeg_bits: int
    raw_rgb_8bit_bits: int
    semantic_box_bits: int


def parse_visdrone_annotation(path: str | Path, ignore_regions: bool = True) -> list[VisDroneBox]:
    annotation_path = Path(path)
    boxes: list[VisDroneBox] = []
    if not annotation_path.exists():
        return boxes
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            raise ValueError(f"Invalid VisDrone annotation line in {annotation_path}: {line!r}")
        box = VisDroneBox(
            x=float(parts[0]),
            y=float(parts[1]),
            width=float(parts[2]),
            height=float(parts[3]),
            score=int(parts[4]),
            category=int(parts[5]),
            truncation=int(parts[6]),
            occlusion=int(parts[7]),
        )
        if ignore_regions and box.is_ignored:
            continue
        boxes.append(box)
    return boxes


def iter_visdrone_frames(root: str | Path, max_frames: int | None = None) -> list[VisDroneFrame]:
    root_path = Path(root)
    image_dir = root_path / "images"
    annotation_dir = root_path / "annotations"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[: max(0, int(max_frames))]

    frames: list[VisDroneFrame] = []
    for image_path in image_paths:
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        boxes = parse_visdrone_annotation(annotation_path, ignore_regions=True)
        with Image.open(image_path) as img:
            width, height = img.size
        frames.append(
            VisDroneFrame(
                stem=image_path.stem,
                image_path=image_path,
                annotation_path=annotation_path,
                width=width,
                height=height,
                boxes=boxes,
                jpeg_bits=image_path.stat().st_size * 8,
                raw_rgb_8bit_bits=width * height * 3 * 8,
                semantic_box_bits=hard_box_packet_bits(len(boxes), label_bits=8, coord_bits=16),
            )
        )
    return frames
