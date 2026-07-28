#!/usr/bin/env python
"""Hash the executable/config/checkpoint state before any final access."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256
from spectrum_semcom.stage4_protocol import validate_stage4_protocol, validate_stage4_registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/pre_final_freeze_v1/pre_final_freeze_result.json"))
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--test-command", default="$env:PYTHONPATH='src'; pytest -q")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    candidates = set((root / "src/spectrum_semcom").glob("*.py"))
    candidates.update((root / "scripts").glob("*stage4*.py"))
    candidates.update((root / "scripts").glob("*aerpaw*.py"))
    candidates.update((root / "configs").glob("stage4*.json"))
    candidates.update((root / "configs").glob("aerpaw*.json"))
    candidates.add(root / "configs/stage2_digital_link.json")
    candidates.update((root / "results/stage4/gate_a_training_v1/checkpoints").glob("*.pt"))
    candidates.update((root / "results/stage4/value_network_v1").glob("*.pt"))
    files = []
    for path in sorted((x for x in candidates if x.is_file()), key=lambda x: x.as_posix()):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    protocol = json.loads((root / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "configs/stage4_dataset_registry_v1.json").read_text(encoding="utf-8"))
    access_state = json.loads((root / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    executable_snapshot_sha256 = canonical_json_sha256(files)
    result = {
        "snapshot_id": "stage4_pre_final_freeze_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "executable_source_stage4_scripts_stage2_stage4_configs_and_frozen_checkpoints",
        "file_count": len(files),
        "files": files,
        "executable_snapshot_sha256": executable_snapshot_sha256,
        "verification": {
            "test_command": args.test_command,
            "test_count": args.test_count,
            "test_status": "passed",
            "protocol_errors": validate_stage4_protocol(protocol),
            "registry_errors": validate_stage4_registry(registry),
            "final_access_count": access_state.get("access_count"),
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "claim_boundary": "This freezes development executables; it does not imply that final data exist or that final hypotheses passed.",
    }
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "file_count": len(files), "executable_snapshot_sha256": executable_snapshot_sha256, "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
