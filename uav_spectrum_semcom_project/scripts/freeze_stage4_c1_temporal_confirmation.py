#!/usr/bin/env python
"""Freeze all executable inputs before the one-time C1 temporal confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/c1_v3_temporal_freeze_v1/executable_snapshot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing temporal executable snapshot")
    access = json.loads((PROJECT_DIR / "configs/stage4_c1_temporal_access_state.json").read_text(encoding="utf-8"))
    if access.get("status") != "not_accessed" or access.get("access_count") != 0:
        raise RuntimeError("freeze must precede temporal confirmation access")
    manifest_path = PROJECT_DIR / "results/stage4/c1_v3_temporal_calibration_v1/selected_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_frozen_before_confirmation_access" or manifest.get("confirmation_accessed") is not False:
        raise ValueError("candidate manifest is not eligible for freezing")
    paths = [
        PROJECT_DIR / "configs/aerpaw_c1_temporal_holdout_protocol_v1.json",
        PROJECT_DIR / "configs/stage4_c1_v3_temporal_calibration.json",
        PROJECT_DIR / "configs/stage2_digital_link.json",
        PROJECT_DIR / "results/stage4/aerpaw_c1_temporal_prefinal_v1/prefinal_inventory.json",
        PROJECT_DIR / "results/stage4/c1_v3_temporal_calibration_v1/c1_v3_calibration_result.json",
        manifest_path,
        PROJECT_DIR / "scripts/run_stage2_digital_link.py",
        PROJECT_DIR / "scripts/run_aerpaw_c1_temporal_confirmation.py",
        PROJECT_DIR / "scripts/run_aerpaw_c1_temporal_statistics.py",
        PROJECT_DIR / "scripts/consume_stage4_c1_temporal_access.py",
        Path(__file__).resolve(),
        PROJECT_DIR / "pyproject.toml",
        PROJECT_DIR / "requirements-lock.txt",
    ]
    paths.extend(sorted((PROJECT_DIR / "src/spectrum_semcom").glob("*.py")))
    paths.extend(Path(row["path"]) for row in manifest["baseline_checkpoints"])
    paths.extend(PROJECT_DIR / row["checkpoint"] for row in manifest["candidate_checkpoints"])
    resolved = sorted({path.resolve() for path in paths}, key=lambda value: str(value).lower())
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("snapshot inputs are missing: " + "; ".join(missing))
    files = [
        {"path": str(path.relative_to(PROJECT_DIR)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in resolved
    ]
    payload = {
        "version": "1.0",
        "status": "frozen_before_single_temporal_confirmation_access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "executable_snapshot_sha256": canonical_json_sha256(files),
        "environment": environment_snapshot(["numpy", "torch"]),
        "access_state_deliberately_excluded": "configs/stage4_c1_temporal_access_state.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "files": len(files), "snapshot_sha256": payload["executable_snapshot_sha256"], "access_count": 0}, indent=2))


if __name__ == "__main__":
    main()
