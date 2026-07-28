#!/usr/bin/env python
"""Atomically consume the single final-holdout access immediately before inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, consume_single_access


CONFIRMATION = "CONSUME_STAGE4_FINAL_ACCESS_ONCE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=PROJECT_DIR / "configs/stage4_final_registry_v1.json")
    parser.add_argument("--state", type=Path, default=PROJECT_DIR / "configs/stage4_final_access_state.json")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"refusing access: --confirm must equal {CONFIRMATION}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    updated = consume_single_access(state, registry, actor=args.actor, code_snapshot_sha256=args.code_snapshot_sha256)
    atomic_write_json(args.state, updated)
    print(json.dumps(updated, indent=2))


if __name__ == "__main__":
    main()
