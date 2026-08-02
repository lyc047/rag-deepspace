#!/usr/bin/env python
"""Freeze the complete Stage-9 evidence registry and claim boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json
DEFAULT_CONFIG = PROJECT_DIR / "configs/stage9_closeout_freeze_v1.json"
DEFAULT_OUTPUT = PROJECT_DIR / "results/stage9/stage9_closeout_freeze_v1/result.json"
TEXT_SUFFIXES = {".json", ".md", ".py"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(value, path: list[str]):
    for key in path:
        value = value[key]
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_metadata(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitattributes"
    if is_text:
        payload = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hash_mode = "lf_normalized_bytes"
    else:
        payload = raw
        hash_mode = "exact_bytes"
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "working_tree_bytes": len(raw),
        "hash_mode": hash_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-9 closeout freeze")

    config = _read_json(args.config)
    check_rows: list[dict] = []
    for relative, rules in config["required_checks"].items():
        source = _read_json(PROJECT_DIR / relative)
        for rule in rules:
            actual = _nested(source, list(rule["path"]))
            passed = actual == rule["equals"]
            check_rows.append(
                {
                    "artifact": relative,
                    "path": rule["path"],
                    "expected": rule["equals"],
                    "actual": actual,
                    "passed": passed,
                }
            )
    if not all(row["passed"] for row in check_rows):
        failed = [row for row in check_rows if not row["passed"]]
        raise ValueError(f"Stage-9 closeout prerequisite failed: {failed}")

    artifact_hashes: dict[str, dict[str, object]] = {}
    for relative in config["artifacts"]:
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_hashes[relative] = _artifact_metadata(path)

    git_root = PROJECT_DIR.parent
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        text=True,
    ).strip()
    bundle_material = {
        "config_sha256": _artifact_metadata(args.config)["sha256"],
        "artifact_hashes": artifact_hashes,
        "required_check_results": check_rows,
        "frozen_stack": config["frozen_stack"],
        "governance": config["governance"],
    }
    result = {
        "version": "1.0",
        "status": "FROZEN",
        "freeze_id": config["freeze_id"],
        "evidence_bundle_id": config["evidence_bundle_id"],
        "bundle_semantics": config["bundle_semantics"],
        "source_git_commit_at_freeze": git_commit,
        "config_sha256": bundle_material["config_sha256"],
        "bundle_sha256": _canonical_sha256(bundle_material),
        "frozen_stack": config["frozen_stack"],
        "required_check_results": check_rows,
        "artifact_hashes": artifact_hashes,
        "governance": config["governance"],
        "claim_boundary": [
            "Stage 9 closes protocol and desktop software evidence, not the failed Stage-6 external Final.",
            "S9-EVIDENCE-BUNDLE-v1 is an evidence registry, not a newly tested end-to-end candidate.",
            "Stage-9.4 causal leakage and the resulting withdrawal of affected earlier claims remain part of the record.",
            "Stage-9.9 batching candidates remain rejected negative results.",
            "No RF, board-level, UAV power, real reboot-rate, or active-security claim is supported.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    args.output.write_bytes(args.output.read_bytes().replace(b"\r\n", b"\n"))
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_bundle_id": result["evidence_bundle_id"],
                "artifact_count": len(artifact_hashes),
                "required_check_count": len(check_rows),
                "bundle_sha256": result["bundle_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
