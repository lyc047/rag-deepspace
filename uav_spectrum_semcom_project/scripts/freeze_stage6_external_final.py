#!/usr/bin/env python
"""Freeze the Stage-6 Final executable set without reading Final signals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    build_stage6_code_snapshot,
)


def verify_development_evidence(protocol: dict) -> None:
    for label, entry in protocol["frozen_development_evidence"].items():
        actual = sha256_file(PROJECT_DIR / entry["path"])
        if actual != entry["sha256"]:
            raise ValueError(f"frozen development evidence changed: {label}")
    analysis_entry = protocol["frozen_development_evidence"][
        "task_scale_analysis"
    ]
    analysis = json.loads(
        (PROJECT_DIR / analysis_entry["path"]).read_text(encoding="utf-8")
    )
    decision = analysis["decision_summary"]
    checks = analysis["checks"]
    if (
        not decision["broad_rate_risk_support_passed"]
        or not decision["primary_reproduction_passed"]
        or not checks["all_grid_points_retained"]
        or not checks["wrong_codebook_actions_must_be_zero"]
        or checks["external_final_access_count"] != 0
    ):
        raise ValueError("S6.7c did not authorize external Final freezing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_access_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_freeze_v1/code_snapshot.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 Final snapshot")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if (
        state.get("access_count") != 0
        or state.get("final_signal_values_accessed") is not False
    ):
        raise RuntimeError("cannot freeze after Stage-6 external Final access")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    verify_development_evidence(protocol)
    snapshot = build_stage6_code_snapshot(PROJECT_DIR, protocol)
    snapshot["created_utc"] = datetime.now(timezone.utc).isoformat()
    snapshot["access_count_at_freeze"] = 0
    snapshot["final_signal_values_loaded_at_freeze"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, snapshot)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": len(snapshot["files"]),
                "snapshot_sha256": snapshot["executable_snapshot_sha256"],
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
