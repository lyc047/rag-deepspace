#!/usr/bin/env python
"""Freeze the Stage-7 candidate artifact and decision hashes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json
from spectrum_semcom.reproducibility import sha256_file


DEFAULT_CONFIG = PROJECT_DIR / "configs/stage7_candidate_freeze_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage7/stage7_candidate_freeze_v1/result.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(value: dict, path: list[str]):
    for key in path:
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite candidate freeze")
    config = _read(args.config)
    decision_checks = {}
    for relative, rule in config["required_decisions"].items():
        source = _read(PROJECT_DIR / relative)
        actual = _nested(source, list(rule["path"]))
        passed = actual == rule["equals"]
        decision_checks[relative] = {
            "path": rule["path"],
            "expected": rule["equals"],
            "actual": actual,
            "passed": passed,
        }
    if not all(row["passed"] for row in decision_checks.values()):
        raise ValueError("candidate decision prerequisite failed")
    hashes = {}
    for relative in config["artifacts"]:
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    git_root = PROJECT_DIR.parent
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        text=True,
    ).strip()
    result = {
        "version": "1.0",
        "status": "FROZEN",
        "freeze_id": config["freeze_id"],
        "candidate_id": config["candidate_id"],
        "frozen_claim": config["frozen_claim"],
        "config_sha256": sha256_file(args.config),
        "source_git_commit_at_freeze": git_commit,
        "decision_checks": decision_checks,
        "artifact_hashes": hashes,
        "governance": config["governance"],
        "claim_boundary": [
            "This freeze does not change the FAILED Stage-6 Final status.",
            "Stage 7 has no independent external Final.",
            "The git commit identifies the parent state; this freeze result is committed afterward.",
            "Future confirmation requires at least 20 new independent units and no post-access retuning."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], "candidate_id": result["candidate_id"], "artifact_count": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
