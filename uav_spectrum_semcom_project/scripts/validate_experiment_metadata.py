from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.research_scope import validate_experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment metadata against the canonical stage-0 thesis scope.")
    parser.add_argument("metadata", type=Path, help="Path to an experiment metadata JSON file.")
    args = parser.parse_args()

    with args.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    errors = validate_experiment_metadata(metadata)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"OK: {args.metadata} satisfies the canonical stage-0 metadata requirements.")


if __name__ == "__main__":
    main()
