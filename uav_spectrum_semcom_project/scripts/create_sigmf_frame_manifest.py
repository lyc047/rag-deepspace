from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.data_io import read_sigmf_meta


DEFAULT_META = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sigmf_5g_short"
    / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
)
OUT_PATH = PROJECT_DIR / "data" / "frame_manifest.csv"


def _frame_index(label: str) -> int | None:
    match = re.fullmatch(r"Frame\s+(\d+)", label.strip())
    return int(match.group(1)) if match else None


def _split_for_rank(rank: int, n: int) -> str:
    if n <= 2:
        return "train"
    train_end = max(1, int(round(n * 0.6)))
    val_end = max(train_end + 1, int(round(n * 0.8)))
    if rank < train_end:
        return "train"
    if rank < val_end:
        return "val"
    return "test"


def main() -> None:
    meta = read_sigmf_meta(DEFAULT_META)
    annotations = meta.get("annotations", [])
    frames = []
    for ann in annotations:
        label = str(ann.get("core:label", ""))
        idx = _frame_index(label)
        if idx is None:
            continue
        frames.append(
            {
                "frame_number": idx,
                "start_sample": int(ann.get("core:sample_start", 0)),
                "sample_count": int(ann.get("core:sample_count", 0)),
            }
        )
    frames.sort(key=lambda row: row["frame_number"])

    rows = []
    for rank, frame in enumerate(frames):
        start = frame["start_sample"]
        end = start + frame["sample_count"]
        labels = []
        for ann in annotations:
            label = str(ann.get("core:label", "") or "unknown")
            if label.startswith("Frame"):
                continue
            ann_start = int(ann.get("core:sample_start", 0))
            ann_count = int(ann.get("core:sample_count", 0))
            ann_end = ann_start + ann_count
            if ann_count > 0 and ann_end > start and ann_start < end:
                labels.append(label)
        rows.append(
            {
                "frame_id": f"sigmf_5g_short_frame_{frame['frame_number']:03d}",
                "dataset_id": "sigmf_5g_short",
                "split": _split_for_rank(rank, len(frames)),
                "meta_path": "data/raw/sigmf_5g_short/1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta",
                "start_sample": start,
                "sample_count": frame["sample_count"],
                "frame_number": frame["frame_number"],
                "annotation_count": len(labels),
                "labels_json": json.dumps(sorted(labels), ensure_ascii=False),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    print(f"wrote {OUT_PATH}")
    print(json.dumps({"n_frames": len(rows), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()

