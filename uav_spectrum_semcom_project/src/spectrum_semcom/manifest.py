from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataManifestRow:
    dataset_id: str
    split: str
    format: str
    meta_path: Path
    data_path: Path
    source_url: str
    notes: str = ""


def load_manifest(manifest_path: str | Path, project_dir: str | Path | None = None) -> List[DataManifestRow]:
    """Load a CSV manifest and resolve paths relative to project_dir."""

    manifest_path = Path(manifest_path)
    root = Path(project_dir) if project_dir is not None else manifest_path.parent.parent
    rows: List[DataManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                DataManifestRow(
                    dataset_id=row["dataset_id"],
                    split=row.get("split", "unsplit"),
                    format=row["format"],
                    meta_path=(root / row["meta_path"]).resolve(),
                    data_path=(root / row["data_path"]).resolve(),
                    source_url=row.get("source_url", ""),
                    notes=row.get("notes", ""),
                )
            )
    return rows

