from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.reproducibility import sha256_file, sha256_strings


def inspect_csv_manifest(path: Path, policy: dict) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    split_counts = Counter(row.get("split", "unspecified") for row in rows)
    frame_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        frame_id = row.get("frame_id")
        if frame_id:
            frame_splits[frame_id].add(row.get("split", "unspecified"))
    leakage = sorted(frame_id for frame_id, splits in frame_splits.items() if len(splits) > 1)
    source_group_key = policy.get("source_group_key")
    source_splits: dict[str, set[str]] = defaultdict(set)
    if source_group_key:
        for row in rows:
            source_group = row.get(source_group_key)
            if source_group:
                source_splits[source_group].add(row.get("split", "unspecified"))
    source_leakage = sorted(group for group, splits in source_splits.items() if len(splits) > 1)
    return {
        "path": path.relative_to(PROJECT_DIR).as_posix(),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_frame_ids": len(frame_splits),
        "cross_split_frame_ids": leakage,
        "role": policy.get("role", "unspecified"),
        "formal_eligible": bool(policy.get("formal_eligible", False)),
        "source_group_key": source_group_key,
        "unique_source_groups": len(source_splits),
        "cross_split_source_groups": source_leakage,
        "policy_reason": policy.get("reason", ""),
    }


def inspect_raddet(root: Path) -> dict | None:
    image_root = root / "images"
    if not image_root.exists():
        return None
    splits: dict[str, dict] = {}
    for split_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        paths = sorted(split_dir.glob("*.png"))
        relative_names = [path.relative_to(root).as_posix() for path in paths]
        splits[split_dir.name] = {
            "frames": len(paths),
            "filename_list_sha256": sha256_strings(relative_names),
        }
    return {
        "path": root.relative_to(PROJECT_DIR).as_posix(),
        "identity": "local_data_yaml_directory_split",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze dataset identities, split counts, and leakage checks for stage 1.")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "configs" / "dataset_registry.json")
    parser.add_argument("--policy", type=Path, default=PROJECT_DIR / "configs" / "dataset_policy.json")
    parser.add_argument(
        "--raddet-root",
        type=Path,
        default=PROJECT_DIR / "data" / "raw" / "raddet" / "RadDet40k128HW001Tv2",
    )
    args = parser.parse_args()

    with args.policy.open("r", encoding="utf-8") as f:
        dataset_policy = json.load(f)
    policy_by_path = dataset_policy.get("manifests", {})
    manifests = []
    for path in sorted((PROJECT_DIR / "data").glob("*.csv")):
        relative_path = path.relative_to(PROJECT_DIR).as_posix()
        manifests.append(inspect_csv_manifest(path, policy_by_path.get(relative_path, {})))
    registry = {
        "version": "1.0",
        "split_id": "stage1_dataset_registry_v1",
        "rules": [
            "Split source frames or acquisition sequences before generating channel-condition samples.",
            "Do not tune thresholds or model choices on the test split.",
            "The local RadDet data.yaml train/val/test directory mapping is frozen as observed; do not relabel it without provenance verification.",
            "Unrelated RadDet frames aggregated for pressure tests are not correlated multi-UAV observations.",
        ],
        "manifests": manifests,
        "raddet": (inspect_raddet(args.raddet_root) or {}) | dataset_policy.get("raddet", {}),
    }
    leakage = [
        entry["path"]
        for entry in manifests
        if entry["formal_eligible"] and (entry["cross_split_frame_ids"] or entry["cross_split_source_groups"])
    ]
    if leakage:
        raise SystemExit(f"cross-split frame leakage detected in: {', '.join(leakage)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
