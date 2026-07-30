#!/usr/bin/env python
"""Freeze development banks, workpoints, adapter, and code before Final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_few_shot_adaptation import _load_combined  # noqa: E402
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
    role_members,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot  # noqa: E402
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6r_context_adapter import (  # noqa: E402
    leave_one_context_novelty_threshold,
    robust_spectral_shape_features,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    code_snapshot,
    resolve_context_and_actions,
    task_queries,
)
from spectrum_semcom.stage6r_regret_codebook import (  # noqa: E402
    exhaustive_action_tuples,
    greedy_select_actions,
    regret_matrix,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_electrosense_external_final_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/electrosense_final_freeze_v1/result.json"
)


def _workpoints(scan: dict, n_channels: int, target: float) -> dict:
    block = scan["conditions"]["primary_fault"][str(n_channels)]
    matched = next(
        row
        for row in block["matched_clean"]
        if row["common_eligible"]
        and abs(float(row["target_clean_rate"]) - target) < 1e-12
    )
    semantic = next(
        row
        for row in block["semantic_workpoints"]
        if row["workpoint_id"] == matched["semantic_workpoint_id"]
    )
    exact = next(
        row
        for row in block["exact_workpoints"]
        if row["workpoint_id"] == matched["exact_workpoint_id"]
    )
    return {"matched": matched, "semantic": semantic, "exact": exact}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen Final snapshot")
    config = read_json(args.config)
    paths = {
        name: Path(value)
        if name == "archive"
        else PROJECT_DIR / value
        for name, value in config["inputs"].items()
        if name != "freeze_result"
    }
    registry = read_json(paths["registry"])
    access = read_json(paths["access_state"])
    if access["roles"]["pilot"]["access_count"] != 1:
        raise ValueError("Pilot must be consumed before freezing the adapter")
    if access["roles"]["stage6_final"]["access_count"] != 0:
        raise ValueError("Stage-6 Final values were accessed before freeze")
    if any(
        access["roles"][role]["access_count"] != 0
        for role in ("confirmation_lockbox", "reserve")
    ):
        raise ValueError("lockbox or reserve values were accessed")
    adaptation = read_json(paths["adaptation_config"])
    temporal = read_json(paths["temporal_gate_config"])
    scan_config = read_json(paths["matched_scan_config"])
    scan = read_json(paths["matched_scan_result"])
    if config["task"]["n_channels"] != scan_config["task"]["n_channels"]:
        raise ValueError("N values differ from frozen matched scan")
    if config["task"]["demand_ratios"] != adaptation["task"]["demand_ratios"]:
        raise ValueError("demand ratios differ from frozen adaptation")
    if abs(
        float(config["task"]["epsilon_db"])
        - float(scan_config["task"]["epsilon_db"])
    ) > 1e-12:
        raise ValueError("epsilon differs from frozen matched scan")
    for field in (
        "window_size_scenes",
        "required_consecutive_matching_windows",
        "checkpoints_scenes",
        "minimum_accumulated_bank_coverage",
        "maximum_calibration_scenes",
    ):
        if config["context_gate"][field] != temporal["task"][field]:
            raise ValueError(f"context gate field changed: {field}")

    pilot_metadata = role_members(registry, "pilot")
    pilot_arrays = load_npy_members(
        paths["archive"], [row["member"] for row in pilot_metadata]
    )
    n_artifacts = {}
    epsilon = float(config["task"]["epsilon_db"])
    k = int(config["task"]["primary_k"])
    for n_channels in map(int, config["task"]["n_channels"]):
        development_power, campaigns, _ = _load_combined(
            adaptation, n_channels
        )
        queries = task_queries(
            n_channels, config["task"]["demand_ratios"]
        )
        development_states = [
            build_task_state(row, queries, epsilon_db=epsilon)
            for row in development_power
        ]
        actions = exhaustive_action_tuples(development_states)
        regrets = regret_matrix(development_states, actions)
        prototypes = {}
        bank_actions = {}
        bank_ids = {}
        for bank_id, source in enumerate(
            config["context_gate"]["development_bank_sources"], start=1
        ):
            indices = np.flatnonzero(campaigns == source)
            prototypes[source] = robust_spectral_shape_features(
                development_power[indices]
            )
            selected = greedy_select_actions(
                regrets=regrets[indices],
                actions=actions,
                epsilon_db=epsilon,
                weights=np.ones(indices.size, dtype=np.float64),
                max_codewords=k,
            )
            bank_actions[source] = [list(row) for row in selected.actions]
            bank_ids[source] = bank_id
        novelty_threshold = leave_one_context_novelty_threshold(prototypes)
        pilot_rows = []
        for metadata in pilot_metadata:
            power = aggregate_frequency_bins(
                pilot_arrays[metadata["member"]], n_channels
            )
            resolution = resolve_context_and_actions(
                power=power,
                queries=queries,
                epsilon_db=epsilon,
                prototypes=prototypes,
                novelty_threshold=novelty_threshold,
                bank_actions=bank_actions,
                gate=config["context_gate"],
                primary_k=k,
            )
            resolution.pop("states")
            pilot_rows.append(
                {
                    "site": metadata["site"],
                    "scene_count": int(power.shape[0]),
                    **resolution,
                }
            )
        n_artifacts[str(n_channels)] = {
            "n_channels": n_channels,
            "queries": [
                {"demand_channels": query.demand_channels}
                for query in queries
            ],
            "prototypes": {
                source: values.tolist()
                for source, values in prototypes.items()
            },
            "novelty_threshold": float(novelty_threshold),
            "bank_actions": bank_actions,
            "bank_ids": bank_ids,
            "workpoints": _workpoints(
                scan,
                n_channels,
                float(config["reliability"]["matched_clean_target"]),
            ),
            "pilot_adapter_gate": pilot_rows,
        }
        print(f"Frozen development bank and Pilot gate N={n_channels}")
    snapshot = code_snapshot(PROJECT_DIR, config["code_snapshot_paths"])
    result = {
        "version": "1.0",
        "status": "stage6r_electrosense_final_frozen_unaccessed",
        "experiment_id": config["experiment_id"],
        "protocol_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        },
        "archive_sha256": registry["archive"]["sha256"],
        "registry_sha256": sha256_file(paths["registry"]),
        "final_sites": registry["split"]["roles"]["stage6_final"],
        "final_signal_values_accessed": False,
        "code_snapshot": snapshot,
        "n_artifacts": n_artifacts,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(
        {
            "output": str(args.output),
            "code_snapshot_sha256": snapshot["combined_sha256"],
            "final_site_count": len(result["final_sites"]),
            "final_signal_values_accessed": False,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
