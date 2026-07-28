from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .bit_budget import hard_box_packet_bits, iq_payload_bits, stft_payload_bits


RADDET_CLASS_NAMES = {
    0: "Rect",
    1: "Barker",
    2: "Frank",
    3: "P1",
    4: "P2",
    5: "P3",
    6: "P4",
    7: "Px",
    8: "ZadoffChu",
    9: "LFM",
    10: "FMCW",
}


@dataclass(frozen=True)
class RadDetBox:
    class_idx: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def class_name(self) -> str:
        return RADDET_CLASS_NAMES.get(self.class_idx, f"class_{self.class_idx}")


@dataclass(frozen=True)
class RadDetFrame:
    split: str
    stem: str
    image_path: Path
    label_path: Path | None
    metadata_path: Path | None
    boxes: list[RadDetBox]
    image_compressed_bits: int
    raw_spectrogram_8bit_bits: int
    raw_iq_12bit_bits: int | None
    semantic_box_bits: int
    snr_db: float | None


def parse_raddet_label_file(path: str | Path) -> list[RadDetBox]:
    label_path = Path(path)
    if not label_path.exists():
        return []
    boxes: list[RadDetBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line!r}")
        boxes.append(
            RadDetBox(
                class_idx=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def read_raddet_metadata(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    metadata_path = Path(path)
    if not metadata_path.exists():
        return []
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def iter_raddet_frames(
    root: str | Path,
    split: str,
    read_metadata: bool = True,
    default_sequence_length: int | None = None,
    max_frames: int | None = None,
    frame_offset: int = 0,
) -> list[RadDetFrame]:
    root_path = Path(root)
    image_dir = root_path / "images" / split
    label_dir = root_path / "labels" / split
    metadata_dir = root_path / "metadata" / split
    frames: list[RadDetFrame] = []

    offset = int(frame_offset)
    if offset < 0:
        raise ValueError("frame_offset must be non-negative")
    image_paths = sorted(image_dir.glob("*.png"))[offset:]
    if max_frames is not None:
        image_paths = image_paths[: max(0, int(max_frames))]
    for image_path in image_paths:
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        metadata_path = metadata_dir / f"{stem}.json"
        boxes = parse_raddet_label_file(label_path)
        metadata = read_raddet_metadata(metadata_path) if read_metadata else []
        first_meta = metadata[0] if metadata else {}
        sequence_length = first_meta.get("SequenceLength", default_sequence_length)
        raw_iq_bits = iq_payload_bits(int(sequence_length), 12, 12) if sequence_length else None
        snr = first_meta.get("SNR")
        frames.append(
            RadDetFrame(
                split=split,
                stem=stem,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                metadata_path=metadata_path if metadata_path.exists() else None,
                boxes=boxes,
                image_compressed_bits=image_path.stat().st_size * 8,
                raw_spectrogram_8bit_bits=stft_payload_bits((128, 128), bits_per_value=8),
                raw_iq_12bit_bits=raw_iq_bits,
                semantic_box_bits=hard_box_packet_bits(len(boxes)),
                snr_db=float(snr) if snr is not None else None,
            )
        )
    return frames
