from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from spectrum_semcom.frame_manifest import load_frame_manifest
from run_frame_manifest_baseline import _aggregate, _evaluate_one


def _evaluate_rows(rows, args):
    return [_evaluate_one(row, args) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune traditional frame baseline on val split and evaluate test split.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "frame_manifest.csv")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--enhancements", nargs="+", default=["raw", "freq_median", "time_median"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    parser.add_argument("--min-cells", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_frame_manifest(args.manifest, project_dir=PROJECT_DIR)
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]

    trials = []
    for enhancement, sigma, min_cells in itertools.product(args.enhancements, args.sigmas, args.min_cells):
        trial_args = argparse.Namespace(
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            enhancement=enhancement,
            sigma=sigma,
            min_cells=min_cells,
            max_boxes=args.max_boxes,
            iou_threshold=args.iou_threshold,
        )
        val_eval_rows = _evaluate_rows(val_rows, trial_args)
        val_agg = _aggregate(val_eval_rows)
        trials.append(
            {
                "enhancement": enhancement,
                "sigma": sigma,
                "min_cells": min_cells,
                "val": val_agg,
            }
        )

    best = max(trials, key=lambda item: (item["val"]["f1"], item["val"]["precision"]))
    best_args = argparse.Namespace(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        enhancement=best["enhancement"],
        sigma=best["sigma"],
        min_cells=best["min_cells"],
        max_boxes=args.max_boxes,
        iou_threshold=args.iou_threshold,
    )
    test_eval_rows = _evaluate_rows(test_rows, best_args)
    test_agg = _aggregate(test_eval_rows)

    result = {
        "best_params": {
            "enhancement": best["enhancement"],
            "sigma": best["sigma"],
            "min_cells": best["min_cells"],
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "max_boxes": args.max_boxes,
            "iou_threshold": args.iou_threshold,
        },
        "best_val": best["val"],
        "test": test_agg,
        "test_rows": test_eval_rows,
        "trials": sorted(trials, key=lambda item: item["val"]["f1"], reverse=True),
    }

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_baseline_tuned_val_test.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: result[k] for k in ["best_params", "best_val", "test"]}, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

