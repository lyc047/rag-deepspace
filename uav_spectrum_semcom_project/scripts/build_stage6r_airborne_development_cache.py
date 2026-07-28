#!/usr/bin/env python
"""Build a development-only cache from the consumed 2024-2025 airborne catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

import run_stage5_external_final as stage5_final  # noqa: E402
from spectrum_semcom.aerpaw_helikite_repair import (  # noqa: E402
    load_helikite_zip_power_sweep_compatible,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json"
)
DEFAULT_CATALOG = (
    PROJECT_DIR / "results/stage6/external_final_catalog_v1/catalog.json"
)
DEFAULT_STATE = (
    PROJECT_DIR / "configs/stage6_external_final_access_state.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/airborne_development_cache_v1/cache.npz"
)
DEFAULT_RESULT = (
    PROJECT_DIR / "results/stage6r/airborne_development_cache_v1/result.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    if args.output.exists() or args.result.exists():
        raise FileExistsError("refusing to overwrite Stage-6R development cache")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if (
        state.get("status") != "access_consumed_final_execution_failed"
        or state.get("access_count") != 1
        or state.get("final_signal_values_accessed") is not True
        or state.get("reset_permitted") is not False
    ):
        raise ValueError("external data is not in the registered consumed state")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if len(catalog.get("scenes", [])) != 200:
        raise ValueError("Stage-6R airborne cache requires all registered 200 scenes")

    parent_loader = stage5_final.load_helikite_zip_power_sweep
    stage5_final.load_helikite_zip_power_sweep = (
        load_helikite_zip_power_sweep_compatible
    )
    try:
        records = stage5_final.load_scene_arrays(catalog, protocol)
    finally:
        stage5_final.load_helikite_zip_power_sweep = parent_loader
    if len(records) != 200:
        raise ValueError("airborne adapter did not load all 200 scenes")

    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(
            [scene["scene_id"] for scene in catalog["scenes"]]
        ),
        "campaign_ids": np.asarray(
            [record["campaign_id"] for record in records]
        ),
        "timestamps_local": np.asarray(
            [record["timestamp_local"] for record in records]
        ),
        "cluster_ids": np.asarray(
            [record["cluster_id"] for record in records]
        ),
    }
    for n_channels in n_values:
        arrays[f"channel_power_n{n_channels}"] = np.stack(
            [
                record["n_data"][str(n_channels)]["channel_power_dbm"]
                for record in records
            ]
        ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    campaign_ids, counts = np.unique(arrays["campaign_ids"], return_counts=True)
    result = {
        "version": "1.0",
        "status": "stage6r_airborne_development_cache_complete",
        "data_role": "consumed_external_data_now_development_only",
        "external_confirmatory_final": False,
        "source_access_count": 1,
        "source_access_state_sha256": sha256_file(args.state),
        "source_catalog_sha256": sha256_file(args.catalog),
        "source_protocol_sha256": sha256_file(args.protocol),
        "cache_path": str(args.output.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "cache_sha256": sha256_file(args.output),
        "scene_count": len(records),
        "campaign_counts": {
            str(key): int(value) for key, value in zip(campaign_ids, counts)
        },
        "n_values": n_values,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in arrays.items()
        },
        "governance": {
            "may_be_used_for_stage6r_training_and_selection": True,
            "may_be_relabelled_as_new_final": False,
            "future_final_requires_new_unseen_signal_data": True,
        },
    }
    atomic_write_json(args.result, result)
    print(
        json.dumps(
            {
                "cache": str(args.output),
                "result": str(args.result),
                "scene_count": len(records),
                "campaign_counts": result["campaign_counts"],
                "cache_sha256": result["cache_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
