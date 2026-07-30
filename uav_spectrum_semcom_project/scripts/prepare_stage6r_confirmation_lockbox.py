#!/usr/bin/env python
"""Freeze the S6R-FH10-v1 lockbox protocol without reading signal values."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import read_json, sha256_file  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot  # noqa: E402
from spectrum_semcom.stage6r_confirmation_governance import (  # noqa: E402
    required_access_counts,
)
from spectrum_semcom.stage6r_external_final import code_snapshot  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_confirmation_lockbox_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/confirmation_lockbox_freeze_v1/result.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite lockbox freeze")

    config = read_json(args.config)
    paths = {
        name: Path(value) if name == "archive" else PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    registry = read_json(paths["registry"])
    access = read_json(paths["access_state"])
    parent = read_json(paths["parent_final_freeze"])
    candidate = read_json(paths["heartbeat_candidate_freeze"])

    if parent["status"] != "stage6r_electrosense_final_frozen_unaccessed":
        raise ValueError("parent Final freeze has unexpected status")
    if candidate["status"] != "stage6r_fixed_heartbeat_candidate_frozen":
        raise ValueError("heartbeat candidate has unexpected status")
    if candidate["candidate_id"] != config["reliability"]["heartbeat_candidate"]:
        raise ValueError("heartbeat candidate ID differs from protocol")
    expected_access = required_access_counts(config)
    for role, expected in expected_access.items():
        actual = int(access["roles"][role]["access_count"])
        if actual != expected:
            raise ValueError(
                f"access count mismatch for {role}: {actual} != {expected}"
            )
    sites = list(registry["split"]["roles"]["confirmation_lockbox"])
    if len(sites) != int(config["data_adapter"]["site_count"]):
        raise ValueError("lockbox site count differs from protocol")
    if paths["archive"].stat().st_size != registry["archive"]["size_bytes"]:
        raise ValueError("archive size differs from registered metadata")
    for site in sites:
        metadata = registry["selected_site_members"][site]
        if metadata["technology"] != config["data_adapter"]["technology"]:
            raise ValueError(f"technology mismatch for lockbox site {site}")
        if metadata["shape"][0] < 200 or metadata["shape"][1] < 64:
            raise ValueError(f"registered shape is too small for {site}")

    if list(map(int, config["task"]["n_channels"])) != sorted(
        map(int, parent["n_artifacts"])
    ):
        raise ValueError("N values differ from parent Final freeze")
    heartbeat_by_n = {
        str(key): int(value)
        for key, value in config["reliability"][
            "heartbeat_silence_scenes_by_n"
        ].items()
    }
    if heartbeat_by_n != candidate["frozen_change"][
        "heartbeat_silence_scenes_by_n"
    ]:
        raise ValueError("heartbeat map differs from selected candidate")

    n_artifacts = copy.deepcopy(parent["n_artifacts"])
    parent_heartbeat = {}
    for n_key, artifact in n_artifacts.items():
        semantic = artifact["workpoints"]["semantic"]
        parent_heartbeat[n_key] = int(
            semantic["heartbeat_silence_scenes"]
        )
        semantic["heartbeat_silence_scenes"] = heartbeat_by_n[n_key]

    snapshot = code_snapshot(PROJECT_DIR, config["code_snapshot_paths"])
    input_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "archive" and path.is_file()
    }
    result = {
        "version": "1.0",
        "status": "stage6r_confirmation_lockbox_frozen_unaccessed",
        "experiment_id": config["experiment_id"],
        "protocol_sha256": sha256_file(args.config),
        "input_hashes": input_hashes,
        "registered_archive": {
            "path": str(paths["archive"]),
            "size_bytes": registry["archive"]["size_bytes"],
            "sha256_from_original_registry": registry["archive"]["sha256"],
            "signal_bytes_rehashed_during_freeze": False
        },
        "registry_sha256": sha256_file(paths["registry"]),
        "sites": sites,
        "signal_values_accessed": False,
        "required_access_counts_before_run": expected_access,
        "heartbeat_change": {
            "parent_by_n": parent_heartbeat,
            "frozen_by_n": heartbeat_by_n,
            "only_changed_field": "heartbeat_silence_scenes"
        },
        "n_artifacts": n_artifacts,
        "code_snapshot": snapshot,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"]
    }
    atomic_write_json(args.output, result)
    print(json.dumps(
        {
            "output": str(args.output),
            "site_count": len(sites),
            "signal_values_accessed": False,
            "code_snapshot_sha256": snapshot["combined_sha256"],
            "required_access_counts": expected_access
        },
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
