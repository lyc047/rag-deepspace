#!/usr/bin/env python
"""Validate a new independent catalog and materialize the final registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, build_final_registry, verify_catalog_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, default=PROJECT_DIR / "configs/stage4_development_registry_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "configs/stage4_final_registry_v1.json")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    development = json.loads(args.development_registry.read_text(encoding="utf-8"))
    registry = build_final_registry(catalog, development)
    file_errors = verify_catalog_files(catalog, args.catalog.resolve().parent)
    if file_errors:
        raise ValueError("final catalog file-integrity audit failed: " + "; ".join(file_errors))
    registry["file_integrity_audit"] = {"all_declared_files_exist": True, "all_sha256_match": True}
    atomic_write_json(args.output, registry)
    print(json.dumps({"output": str(args.output), "scene_count": registry["scene_count"], "real_multi_receiver_scene_count": registry["real_multi_receiver_scene_count"], "access_count": 0}, indent=2))


if __name__ == "__main__":
    main()
