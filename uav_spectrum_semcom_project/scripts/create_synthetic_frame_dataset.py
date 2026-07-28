from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.data_io import save_npz_frame
from spectrum_semcom.synthetic import generate_synthetic_iq_frame


OUT_DIR = PROJECT_DIR / "data" / "processed" / "synthetic_frames"
MANIFEST = PROJECT_DIR / "data" / "synthetic_frame_manifest.csv"


def _split(index: int, total: int) -> str:
    if index < int(total * 0.6):
        return "train"
    if index < int(total * 0.8):
        return "val"
    return "test"


def main() -> None:
    n_frames = 36
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_frames):
        snr_db = [-4, 0, 4, 8, 12][i % 5]
        n_signals = 1 + (i % 4)
        frame = generate_synthetic_iq_frame(
            n_samples=32_768,
            sample_rate_hz=4e6,
            n_signals=n_signals,
            snr_db=snr_db,
            seed=2026 + i,
            frame_id=f"synthetic_frame_{i:04d}",
        )
        rel_path = Path("data") / "processed" / "synthetic_frames" / f"{frame.frame_id}.npz"
        save_npz_frame(frame, PROJECT_DIR / rel_path)
        rows.append(
            {
                "frame_id": frame.frame_id,
                "dataset_id": "synthetic_controlled_iq",
                "split": _split(i, n_frames),
                "path": str(rel_path).replace("\\", "/"),
                "sample_rate_hz": frame.sample_rate_hz,
                "n_samples": frame.n_samples,
                "duration_s": frame.duration_s,
                "snr_db": snr_db,
                "n_signals": n_signals,
                "labels_json": json.dumps([box.label for box in frame.boxes], ensure_ascii=False),
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {MANIFEST}")
    print(json.dumps({"n_frames": n_frames, "out_dir": str(OUT_DIR)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

