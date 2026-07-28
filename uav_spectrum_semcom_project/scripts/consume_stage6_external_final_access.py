#!/usr/bin/env python
"""Atomically consume the one Stage-6 Final access before signal read."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    validate_external_final_catalog,
    verify_stage6_code_snapshot,
)


def build_consumed_state(
    *,
    protocol_path: Path,
    registry_path: Path,
    state_path: Path,
    legacy_state_path: Path,
    snapshot_path: Path,
    catalog_path: Path,
    actor: str,
) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_state_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if (
        state.get("status") != "downloaded_integrity_verified_not_accessed"
        or state.get("access_count") != 0
        or state.get("final_signal_values_accessed") is not False
    ):
        raise RuntimeError("Stage-6 external Final access is not pristine")
    if (
        legacy.get("access_count") != 0
        or legacy.get("final_signal_values_accessed") is not False
    ):
        raise RuntimeError("legacy external Final state is not pristine")
    snapshot_errors = verify_stage6_code_snapshot(PROJECT_DIR, snapshot)
    if snapshot_errors:
        raise ValueError("frozen code changed: " + "; ".join(snapshot_errors))
    catalog_errors = validate_external_final_catalog(
        catalog, protocol, registry
    )
    if catalog_errors:
        raise ValueError("catalog validation failed: " + "; ".join(catalog_errors))
    if catalog.get("code_snapshot_sha256") != snapshot.get(
        "executable_snapshot_sha256"
    ):
        raise ValueError("catalog is not bound to this code snapshot")
    if not actor.strip():
        raise ValueError("actor is required")
    consumed = datetime.now(timezone.utc).isoformat()
    receipt = {
        "access_number": 1,
        "consumed_utc": consumed,
        "actor": actor,
        "protocol_sha256": sha256_file(protocol_path),
        "registry_sha256": sha256_file(registry_path),
        "catalog_sha256": canonical_json_sha256(catalog),
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "crash_policy": "access_remains_consumed_and_may_not_be_reset",
        "legacy_state_superseded": str(legacy_state_path),
    }
    receipt_sha = canonical_json_sha256(receipt)
    stage6 = {
        "version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "status": "access_consumed",
        "access_count": 1,
        "final_signal_values_accessed": True,
        "final_method_outputs_accessed": False,
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "catalog_sha256": receipt["catalog_sha256"],
        "consumed_at": consumed,
        "actor": actor,
        "receipt": receipt,
        "receipt_sha256": receipt_sha,
        "reset_permitted": False,
    }
    legacy_updated = {
        **legacy,
        "status": "superseded_by_stage6_external_final_access_consumed",
        "access_count": 1,
        "final_signal_values_accessed": True,
        "final_method_outputs_accessed": False,
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "catalog_sha256": receipt["catalog_sha256"],
        "consumed_at": consumed,
        "actor": actor,
        "superseding_protocol_id": protocol["protocol_id"],
        "receipt_sha256": receipt_sha,
        "reset_permitted": False,
    }
    return stage6, legacy_updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_external_final_registry_v1.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_external_final_access_state.json",
    )
    parser.add_argument(
        "--legacy-state",
        type=Path,
        default=PROJECT_DIR / "configs/stage5_external_final_access_state.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_freeze_v1/code_snapshot.json",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_DIR
        / "results/stage6/external_final_catalog_v1/catalog.json",
    )
    args = parser.parse_args()
    lock = args.state.with_suffix(args.state.suffix + ".consume.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            "consume lock exists; access may be consumed or a prior run "
            "crashed; automatic reset is forbidden"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "actor": args.actor,
                        "reset_permitted": False,
                    }
                )
            )
        stage6, legacy = build_consumed_state(
            protocol_path=args.protocol,
            registry_path=args.registry,
            state_path=args.state,
            legacy_state_path=args.legacy_state,
            snapshot_path=args.snapshot,
            catalog_path=args.catalog,
            actor=args.actor,
        )
        atomic_write_json(args.legacy_state, legacy)
        atomic_write_json(args.state, stage6)
    except Exception:
        raise
    print(
        json.dumps(
            {
                "status": stage6["status"],
                "access_count": stage6["access_count"],
                "receipt_sha256": stage6["receipt_sha256"],
                "legacy_state_superseded": True,
                "reset_permitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
