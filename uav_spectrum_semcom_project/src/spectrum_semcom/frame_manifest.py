from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .data_io import load_sigmf_frame
from .types import IQFrame


@dataclass(frozen=True)
class FrameManifestRow:
    frame_id: str
    dataset_id: str
    split: str
    meta_path: Path
    start_sample: int
    sample_count: int
    frame_number: int
    annotation_count: int
    labels_json: str


def load_frame_manifest(path: str | Path, project_dir: str | Path) -> List[FrameManifestRow]:
    path = Path(path)
    project_dir = Path(project_dir)
    rows: List[FrameManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                FrameManifestRow(
                    frame_id=row["frame_id"],
                    dataset_id=row["dataset_id"],
                    split=row["split"],
                    meta_path=(project_dir / row["meta_path"]).resolve(),
                    start_sample=int(row["start_sample"]),
                    sample_count=int(row["sample_count"]),
                    frame_number=int(row["frame_number"]),
                    annotation_count=int(row["annotation_count"]),
                    labels_json=row["labels_json"],
                )
            )
    return rows


def load_iq_frame_from_manifest_row(row: FrameManifestRow) -> IQFrame:
    return load_sigmf_frame(
        row.meta_path,
        start_sample=row.start_sample,
        max_samples=row.sample_count,
        normalize=True,
        frame_id=row.frame_id,
    )

