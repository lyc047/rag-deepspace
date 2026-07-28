#!/usr/bin/env python
"""Run frozen Stage-5 spectrum-regime stress tests without loading Final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage5_dynamic_query_development import build_query_schedule  # noqa: E402
from run_stage5_event_semantics_development import load_json  # noqa: E402
from run_stage5_query_bundle_development import (  # noqa: E402
    load_npz,
    run_schedule,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_spectrum_stress import apply_spectrum_regime  # noqa: E402


def validate_inputs(protocol: dict) -> tuple[dict, dict, dict]:
    paths = {
        "cache_result": (
            PROJECT_DIR / protocol["development_source"]["cache_result"],
            protocol["development_source"]["cache_result_sha256"],
        ),
        "schedule": (
            PROJECT_DIR / protocol["schedule"]["protocol"],
            protocol["schedule"]["protocol_sha256"],
        ),
        "predecessor_protocol": (
            PROJECT_DIR / protocol["predecessor"]["protocol"],
            protocol["predecessor"]["protocol_sha256"],
        ),
        "predecessor_result": (
            PROJECT_DIR / protocol["predecessor"]["result"],
            protocol["predecessor"]["result_sha256"],
        ),
        "stage2": (
            PROJECT_DIR / protocol["link"]["stage2_config"],
            protocol["link"]["stage2_config_sha256"],
        ),
    }
    for path, expected in paths.values():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen stress-test input hash mismatch: {path}")
    governance = protocol["governance"]
    if (
        governance["stage4_final_measurements_may_be_loaded"]
        or governance["stage4_final_metrics_may_be_loaded"]
        or governance["fixed_policy_retuned"]
        or governance["stress_trajectories_are_measured_claims"]
    ):
        raise ValueError("spectrum-stress governance is invalid")
    cache_result = load_json(paths["cache_result"][0])
    split = protocol["development_source"]["split"]
    split_row = cache_result["splits"][split]
    if split_row["cache_sha256"] != protocol["development_source"]["cache_sha256"]:
        raise ValueError("stress-test cache declaration changed")
    cache_path = PROJECT_DIR / split_row["cache"]
    if sha256_file(cache_path) != split_row["cache_sha256"]:
        raise ValueError("stress-test cache hash mismatch")
    return (
        load_npz(cache_path),
        load_json(paths["stage2"][0]),
        load_json(paths["schedule"][0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_spectrum_stress_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/stage5/spectrum_stress_development_v1",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite spectrum-stress output")
    protocol = load_json(args.protocol)
    cache, stage2, schedule_protocol = validate_inputs(protocol)
    schedules = {
        name: build_query_schedule(cache, schedule_protocol, name)
        for name in protocol["schedule"]["names"]
    }
    regimes = {}
    for regime_index, (regime, parameters) in enumerate(
        protocol["spectrum_regimes"].items()
    ):
        transformed = dict(cache)
        transformed["channel_power_dbm"] = apply_spectrum_regime(
            cache["channel_power_dbm"],
            cache["site_ids"],
            regime=regime,
            parameters=parameters,
        )
        regimes[regime] = {}
        for schedule_index, (schedule_name, query) in enumerate(
            schedules.items()
        ):
            regimes[regime][schedule_name] = run_schedule(
                transformed,
                stage2,
                protocol,
                query,
                name=f"{regime}/{schedule_name}",
                seed_offset=(regime_index + 1) * 100_000
                + (schedule_index + 1) * 10_000,
            )
    result = {
        "version": "1.0",
        "status": "stage5_spectrum_stress_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "regimes": regimes,
        "governance": {
            "stage4_final_measurements_loaded": False,
            "stage4_final_metrics_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "stress_trajectories_are_new_measurements": False,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    output = args.out_dir / "spectrum_stress_result.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "regimes": list(regimes),
                "final_loaded": False,
                "fixed_policy_retuned": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

