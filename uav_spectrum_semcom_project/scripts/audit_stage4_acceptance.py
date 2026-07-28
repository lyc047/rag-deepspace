#!/usr/bin/env python
"""Generate the stage-4 acceptance gap report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.stage4_acceptance import acceptance_markdown, audit_stage4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/stage4/acceptance_audit_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    result = audit_stage4(root)
    (output / "acceptance_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "acceptance_report.md").write_text(acceptance_markdown(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("development_acceptance", "final_acceptance", "blocking_item_ids", "failed_item_ids")}, indent=2))


if __name__ == "__main__":
    main()
