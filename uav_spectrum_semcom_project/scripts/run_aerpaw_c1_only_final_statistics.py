#!/usr/bin/env python
"""Run the pre-registered C1-only statistics branch after an M1 failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.aerpaw_final_statistics import evaluate_aerpaw_c1_only_statistics  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.final_statistics import canonical_sha256, validate_access_binding  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, default=PROJECT_DIR / "configs/stage4_protocol.json")
    parser.add_argument(
        "--aerpaw-protocol",
        type=Path,
        default=PROJECT_DIR / "configs/aerpaw_three_site_prefinal_protocol_v1.json",
    )
    parser.add_argument(
        "--access-state",
        type=Path,
        default=PROJECT_DIR / "configs/stage4_final_access_state.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    aerpaw = json.loads(args.aerpaw_protocol.read_text(encoding="utf-8"))
    access = json.loads(args.access_state.read_text(encoding="utf-8"))
    errors = validate_access_binding(payload, access)
    if errors:
        raise ValueError("final access binding failed: " + "; ".join(errors))
    result = evaluate_aerpaw_c1_only_statistics(payload, base, aerpaw)
    result["input_sha256"] = canonical_sha256(payload)
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "scene_count": result["scene_count"], "family_decision": result["family_decision"]}, indent=2))


if __name__ == "__main__":
    main()
