#!/usr/bin/env python
"""Register exact C1-v4 Final members without reading measurement values."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_spectrum import (  # noqa: E402
    discover_aerpaw_zip_pairs,
    filter_aerpaw_pairs_by_local_window,
    thin_aerpaw_pairs_by_time,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402

STEM_PATTERN = re.compile(r"results_\d{8}_\d{6}")


def evenly_spaced(items: list, count: int) -> list:
    if count < 1 or len(items) < count:
        raise ValueError(f"cannot select {count} values from {len(items)} eligible values")
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    if len(set(int(value) for value in indices)) != count:
        raise RuntimeError("even spacing repeated an index")
    return [items[int(value)] for value in indices]


def resolve_artifact(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing Final inventory")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    state = json.loads((PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "not_accessed" or state.get("access_count") != 0:
        raise RuntimeError("Final inventory must be registered before access")

    exclusion_files = [resolve_artifact(value).resolve() for value in protocol["permanent_exclusion_artifacts"]]
    missing = [str(path) for path in exclusion_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing exclusion artifacts: " + "; ".join(missing))
    excluded: set[str] = set()
    exclusion_audit = []
    for path in exclusion_files:
        text = path.read_text(encoding="utf-8")
        stems = set(STEM_PATTERN.findall(text))
        excluded.update(stems)
        exclusion_audit.append({
            "path": str(path), "sha256": sha256_file(path), "stem_count": len(stems),
        })

    rule = protocol["final_partition"]
    scenes = []
    archive_audit = {}
    selected_stems: set[str] = set()
    for site, declared in protocol["archives"].items():
        path = Path(declared["path"]).resolve()
        if not path.is_file() or path.stat().st_size != int(declared["size_bytes"]):
            raise ValueError(f"missing or size-mismatched archive: {site}")
        pairs, errors = discover_aerpaw_zip_pairs(path)
        if errors:
            raise ValueError(f"archive pair errors for {site}: {'; '.join(errors[:10])}")
        window = filter_aerpaw_pairs_by_local_window(
            pairs,
            start_local_inclusive=rule["start_local_inclusive"],
            end_local_exclusive=rule["end_local_exclusive"],
        )
        eligible = [pair for pair in window if pair.stem not in excluded]
        thinned = thin_aerpaw_pairs_by_time(eligible, min_separation_s=float(rule["minimum_separation_s"]))
        chosen = evenly_spaced(thinned, int(rule["selected_per_site"]))
        with zipfile.ZipFile(path) as handle:
            for pair in chosen:
                meta = handle.getinfo(pair.meta_member)
                data = handle.getinfo(pair.data_member)
                scenes.append({
                    "site": site,
                    "source_stem": pair.stem,
                    "timestamp_local": pair.timestamp_local,
                    "cluster_id": f"{site}:{pair.timestamp_local[:13].replace('-', '').replace('T', 'T')}",
                    "meta_member": pair.meta_member,
                    "meta_size": meta.file_size,
                    "meta_crc32": f"{meta.CRC:08x}",
                    "data_member": pair.data_member,
                    "data_size": data.file_size,
                    "data_crc32": f"{data.CRC:08x}",
                })
        selected_stems.update(pair.stem for pair in chosen)
        archive_audit[site] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "verified_sha256": declared["sha256"], "complete_pair_count": len(pairs),
            "raw_window_pairs": len(window), "excluded_in_window": len(window) - len(eligible),
            "after_15_minute_thinning": len(thinned), "selected": len(chosen),
            "first_selected": chosen[0].timestamp_local, "last_selected": chosen[-1].timestamp_local,
        }
    scenes.sort(key=lambda row: (row["site"], row["timestamp_local"], row["source_stem"]))
    overlap = sorted(selected_stems & excluded)
    if overlap or len(selected_stems) != len(scenes):
        raise ValueError("Final selection overlaps prior provenance or contains duplicate stems")
    payload = {
        "version": "1.0", "status": "materialized_metadata_only_not_accessed",
        "protocol_sha256": sha256_file(args.protocol), "scene_count": len(scenes),
        "site_counts": {site: sum(row["site"] == site for row in scenes) for site in protocol["archives"]},
        "cluster_count": len({row["cluster_id"] for row in scenes}),
        "archive_audit": archive_audit, "exclusion_audit": exclusion_audit,
        "prior_provenance_overlap": 0, "measurement_member_bytes_read": 0,
        "measurement_values_loaded": False, "scenes": scenes,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "scenes": len(scenes), "clusters": payload["cluster_count"], "measurement_values_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
