#!/usr/bin/env python
"""Record an irreversible Stage-6 Final execution failure.

This post-failure forensic tool reads catalog metadata and execution logs only.
It does not load SigMF sample arrays and cannot restart the Final.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def metadata_compatibility(catalog: dict) -> tuple[list[dict], dict]:
    rows = []
    handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for index, scene in enumerate(catalog["scenes"], start=1):
            archive = Path(scene["archive_path"])
            handle = handles.setdefault(archive, zipfile.ZipFile(archive))
            metadata = json.loads(
                handle.read(scene["meta_member"]).decode("utf-8")
            )
            labels = {
                str(row.get("core:comment", "")).strip().lower()
                for row in metadata.get("annotations", [])
                if isinstance(row, dict)
            }
            compatible = {"freqs", "powers"}.issubset(labels)
            rows.append(
                {
                    "catalog_index": index,
                    "scene_id": scene["scene_id"],
                    "campaign_id": scene["campaign_id"],
                    "required_annotations_present": compatible,
                    "annotation_labels": sorted(labels),
                    "datatype": metadata.get("global", {}).get(
                        "core:datatype"
                    ),
                    "metadata_loaded": True,
                    "sample_array_loaded": False,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()
    counts = Counter(
        (row["campaign_id"], row["required_annotations_present"])
        for row in rows
    )
    summary = {
        campaign: {
            "compatible_scene_count": counts[(campaign, True)],
            "incompatible_scene_count": counts[(campaign, False)],
        }
        for campaign in sorted({row["campaign_id"] for row in rows})
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_catalog_v1/catalog.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_access_state.json",
    )
    parser.add_argument(
        "--stdout",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_v1/execution.log",
    )
    parser.add_argument(
        "--stderr",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_v1/execution.stderr.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_v1/failure_receipt.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 failure receipt")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if (
        state.get("access_count") != 1
        or state.get("final_signal_values_accessed") is not True
        or state.get("reset_permitted") is not False
    ):
        raise ValueError("failure receipt requires consumed immutable access")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    stdout = args.stdout.read_text(encoding="utf-8", errors="replace")
    stderr = args.stderr.read_text(encoding="utf-8", errors="replace")
    matches = [
        int(value)
        for value in re.findall(
            r"loaded Final sweeps: (\d+)/200", stdout
        )
    ]
    rows, campaign_summary = metadata_compatibility(catalog)
    incompatible = [
        row for row in rows if not row["required_annotations_present"]
    ]
    receipt = {
        "version": "1.0",
        "status": "stage6_external_final_execution_failed_inconclusive",
        "failure_category": "external_data_adapter_schema_mismatch",
        "failure_message": (
            "The frozen adapter required freqs and powers annotations, "
            "but the 2025 campaign metadata did not provide them."
        ),
        "access_count": 1,
        "access_receipt_sha256": state["receipt_sha256"],
        "catalog_sha256": sha256_file(args.catalog),
        "stdout_sha256": sha256_file(args.stdout),
        "stderr_sha256": sha256_file(args.stderr),
        "last_reported_loaded_sweep_count": max(matches, default=0),
        "first_incompatible_scene": incompatible[0] if incompatible else None,
        "metadata_compatibility_by_campaign": campaign_summary,
        "compatible_scene_count": sum(
            row["required_annotations_present"] for row in rows
        ),
        "incompatible_scene_count": len(incompatible),
        "raw_algorithm_result_generated": False,
        "statistical_algorithm_result_generated": False,
        "algorithm_effectiveness_verdict": "inconclusive",
        "post_failure_forensic_scope": (
            "SigMF JSON metadata labels only; no sample arrays or method "
            "outputs were loaded by this forensic audit."
        ),
        "retry_on_this_final_permitted": False,
        "reset_permitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, receipt)
    updated = {
        **state,
        "status": "access_consumed_final_execution_failed",
        "final_method_outputs_accessed": False,
        "failure_receipt_sha256": sha256_file(args.output),
        "stage6_external_value_supported": None,
        "algorithm_effectiveness_verdict": "inconclusive",
        "retry_on_this_final_permitted": False,
        "reset_permitted": False,
    }
    atomic_write_json(args.state, updated)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": receipt["status"],
                "compatible_scene_count": receipt[
                    "compatible_scene_count"
                ],
                "incompatible_scene_count": receipt[
                    "incompatible_scene_count"
                ],
                "first_incompatible_scene": receipt[
                    "first_incompatible_scene"
                ],
                "algorithm_effectiveness_verdict": "inconclusive",
                "retry_permitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
