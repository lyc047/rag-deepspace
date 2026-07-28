#!/usr/bin/env python
"""Run the frozen final statistics engine after single-use access is consumed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json
from spectrum_semcom.final_statistics import evaluate_final_statistics, validate_access_binding
from spectrum_semcom.stage4_protocol import assert_valid_stage4_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="scene-level metrics produced by the frozen final inference runner")
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "configs/stage4_protocol.json")
    parser.add_argument("--access-state", type=Path, default=PROJECT_DIR / "configs/stage4_final_access_state.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/stage4/final_holdout_v1/final_statistics_result.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); assert_valid_stage4_protocol(protocol)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    state = json.loads(args.access_state.read_text(encoding="utf-8"))
    binding_errors = validate_access_binding(payload, state)
    if binding_errors:
        raise ValueError("final access binding failed: " + "; ".join(binding_errors))
    result = evaluate_final_statistics(payload, protocol)
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "scene_count": result["scene_count"], "family_decisions": result["family_decisions"]}, indent=2))


if __name__ == "__main__":
    main()
