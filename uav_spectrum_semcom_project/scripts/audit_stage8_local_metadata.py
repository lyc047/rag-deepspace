"""Audit local Stage-8 candidate sources without opening signal values."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    sources: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    eligible_total = 0

    for source in config["sources"]:
        path = Path(source["path"])
        exists = path.exists()
        if not exists:
            size_bytes = None
            path_kind = "missing"
        elif path.is_file():
            size_bytes = path.stat().st_size
            path_kind = "file"
        else:
            size_bytes = None
            path_kind = "directory"
        eligible = int(source["eligible_new_units"])
        eligible_total += eligible
        statuses[source["independence_status"]] += 1
        sources.append(
            {
                "source_id": source["source_id"],
                "path": source["path"],
                "path_exists": exists,
                "path_kind": path_kind,
                "file_size_bytes": size_bytes,
                "format": source["format"],
                "task_compatibility": source["task_compatibility"],
                "independence_status": source["independence_status"],
                "eligible_new_units": eligible,
                "reason": source["reason"],
            }
        )

    evidence = []
    for relative in config["evidence_references"]:
        path = ROOT / relative
        evidence.append({"path": relative, "exists": path.exists()})

    minimum = int(config["minimum_new_independent_units_required"])
    result = {
        "version": "1.0",
        "status": "METADATA_AUDIT_COMPLETE",
        "audit_id": config["audit_id"],
        "signal_values_accessed": False,
        "model_outputs_accessed": False,
        "archive_payloads_opened": False,
        "source_count": len(sources),
        "existing_source_count": sum(item["path_exists"] for item in sources),
        "eligible_new_independent_units": eligible_total,
        "minimum_new_independent_units_required": minimum,
        "independence_status_counts": dict(sorted(statuses.items())),
        "sources": sources,
        "evidence_references": evidence,
        "decision": {
            "local_external_final_ready": eligible_total >= minimum,
            "new_external_data_required": eligible_total < minimum,
            "execution_authorized": False,
        },
        "claim_boundary": [
            "Only paths, file sizes, and previously registered evidence were inspected.",
            "No IQ, power, occupancy, regret, or method output values were loaded.",
            "Existing local archives cannot be relabeled as new independent Final units.",
            "Task-incompatible datasets may be used for other research but not this frozen Final."
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage8_local_metadata_sources_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/stage8/stage8_local_metadata_audit_v1/result.json",
    )
    args = parser.parse_args()
    result = audit(args.config, args.output)
    print(json.dumps(result["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
