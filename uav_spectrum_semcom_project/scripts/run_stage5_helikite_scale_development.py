#!/usr/bin/env python
"""Run fixed Stage-5 semantics on excluded airborne N=8/16/32/64 caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    normalize_cost,
    quantize_unit_interval,
    selected_block_start,
)
from spectrum_semcom.digital_link import nominal_transmitted_bits  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_event_semantics import (  # noqa: E402
    block_regret,
    contiguous_block_costs,
)
from spectrum_semcom.stage5_query_bundle import query_bundle_payload_width  # noqa: E402
from spectrum_semcom.stage5_scale_boundary import index_width  # noqa: E402


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def method_bits(
    n_channels: int,
    demands: tuple[int, ...],
    header: int,
    soft_bits: int,
) -> dict[str, int]:
    payload = query_bundle_payload_width(n_channels, demands)
    return {
        "soft_power_fixed4_always": header + n_channels * soft_bits,
        "separate_optimal_indices_always": len(demands) * header
        + sum(index_width(n_channels, demand) for demand in demands),
        "query_bundle_always": header + payload,
        "query_event_bundle_fixed": header + payload,
    }


def evaluate_n(
    powers: np.ndarray,
    timestamps: np.ndarray,
    clusters: np.ndarray,
    *,
    n_channels: int,
    ratios: tuple[float, ...],
    header: int,
    soft_bits: int,
    threshold_db: float,
    max_age_minutes: float,
    link,
) -> dict:
    demands = tuple(int(round(n_channels * value)) for value in ratios)
    if any(abs(demand / n_channels - ratio) > 1e-12 for demand, ratio in zip(demands, ratios)):
        raise ValueError("demand ratios must map exactly to channel counts")
    bits = method_bits(n_channels, demands, header, soft_bits)
    nominal = {
        method: nominal_transmitted_bits(value, link)
        for method, value in bits.items()
    }
    rows = []
    event_state: dict[int, int] = {}
    last_bundle_timestamp: str | None = None
    event_transmissions = 0
    for position, values in enumerate(powers):
        demand = demands[position % len(demands)]
        costs = contiguous_block_costs(values, demand)
        optimum = int(np.argmin(costs))
        normalized = normalize_cost(values)
        quantized = quantize_unit_interval(normalized, soft_bits)
        soft_start = selected_block_start(quantized, "channel", demand)
        soft_regret = block_regret(costs, soft_start)
        reason = "reuse_within_threshold"
        previous = event_state.get(demand)
        age = None
        if last_bundle_timestamp is not None:
            age = (
                datetime.fromisoformat(str(timestamps[position]))
                - datetime.fromisoformat(last_bundle_timestamp)
            ).total_seconds() / 60.0
        reuse_regret = (
            float("inf") if previous is None else block_regret(costs, previous)
        )
        transmits = (
            previous is None
            or age is None
            or age > max_age_minutes
            or reuse_regret > threshold_db
        )
        if transmits:
            reason = (
                "missing_state"
                if previous is None
                else "state_age"
                if age is not None and age > max_age_minutes
                else "reuse_regret"
            )
            for update_demand in demands:
                event_state[update_demand] = int(
                    np.argmin(contiguous_block_costs(values, update_demand))
                )
            last_bundle_timestamp = str(timestamps[position])
            event_transmissions += 1
        event_start = event_state[demand]
        rows.extend(
            [
                {
                    "scene_position": position,
                    "cluster_id": str(clusters[position]),
                    "demand_channels": demand,
                    "method": "soft_power_fixed4_always",
                    "regret_db": soft_regret,
                    "application_bits": bits["soft_power_fixed4_always"],
                    "nominal_transmitted_bits": nominal[
                        "soft_power_fixed4_always"
                    ],
                    "transmitted": True,
                },
                {
                    "scene_position": position,
                    "cluster_id": str(clusters[position]),
                    "demand_channels": demand,
                    "method": "separate_optimal_indices_always",
                    "regret_db": 0.0,
                    "application_bits": bits[
                        "separate_optimal_indices_always"
                    ],
                    "nominal_transmitted_bits": nominal[
                        "separate_optimal_indices_always"
                    ],
                    "transmitted": True,
                },
                {
                    "scene_position": position,
                    "cluster_id": str(clusters[position]),
                    "demand_channels": demand,
                    "method": "query_bundle_always",
                    "regret_db": 0.0,
                    "application_bits": bits["query_bundle_always"],
                    "nominal_transmitted_bits": nominal[
                        "query_bundle_always"
                    ],
                    "transmitted": True,
                },
                {
                    "scene_position": position,
                    "cluster_id": str(clusters[position]),
                    "demand_channels": demand,
                    "method": "query_event_bundle_fixed",
                    "regret_db": block_regret(costs, event_start),
                    "application_bits": bits["query_event_bundle_fixed"]
                    if transmits
                    else 0,
                    "nominal_transmitted_bits": nominal[
                        "query_event_bundle_fixed"
                    ]
                    if transmits
                    else 0,
                    "transmitted": transmits,
                    "event_reason": reason,
                    "predecision_reuse_regret_db": reuse_regret
                    if np.isfinite(reuse_regret)
                    else None,
                },
            ]
        )
    summaries = {}
    for method in bits:
        selected = [row for row in rows if row["method"] == method]
        regret = np.asarray([row["regret_db"] for row in selected])
        summaries[method] = {
            "scene_count": len(selected),
            "mean_regret_db": float(np.mean(regret)),
            "maximum_regret_db": float(np.max(regret)),
            "cvar_0_9_regret_db": empirical_cvar_numpy(regret, 0.9),
            "mean_application_bits_per_scene": float(
                np.mean([row["application_bits"] for row in selected])
            ),
            "mean_nominal_transmitted_bits_per_scene": float(
                np.mean(
                    [row["nominal_transmitted_bits"] for row in selected]
                )
            ),
            "transmission_rate": float(
                np.mean([row["transmitted"] for row in selected])
            ),
        }
    soft = summaries["soft_power_fixed4_always"]
    comparisons = {}
    for method in (
        "separate_optimal_indices_always",
        "query_bundle_always",
        "query_event_bundle_fixed",
    ):
        current = summaries[method]
        comparisons[method] = {
            "application_bit_reduction_pct_vs_soft": float(
                100.0
                * (
                    soft["mean_application_bits_per_scene"]
                    - current["mean_application_bits_per_scene"]
                )
                / soft["mean_application_bits_per_scene"]
            ),
            "nominal_transmitted_bit_reduction_pct_vs_soft": float(
                100.0
                * (
                    soft["mean_nominal_transmitted_bits_per_scene"]
                    - current["mean_nominal_transmitted_bits_per_scene"]
                )
                / soft["mean_nominal_transmitted_bits_per_scene"]
            ),
            "mean_regret_difference_db_vs_soft": float(
                current["mean_regret_db"] - soft["mean_regret_db"]
            ),
        }
    return {
        "n_channels": n_channels,
        "demands": list(demands),
        "application_bits_per_update": bits,
        "nominal_transmitted_bits_per_update": nominal,
        "summaries": summaries,
        "comparisons_vs_soft_power_fixed4": comparisons,
        "event_bundle_transmission_count": event_transmissions,
        "rows": rows,
    }


def benchmark(
    cache: dict[str, np.ndarray],
    n_channels: int,
    ratios: tuple[float, ...],
    repetitions: int,
) -> dict:
    powers = cache[f"channel_power_n{n_channels}"]
    demands = tuple(int(round(n_channels * value)) for value in ratios)
    tracemalloc.start()
    start = time.perf_counter()
    checksum = 0
    for _ in range(repetitions):
        for position, values in enumerate(powers):
            demand = demands[position % len(demands)]
            checksum += int(np.argmin(contiguous_block_costs(values, demand)))
    duration = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "repetitions": repetitions,
        "sweep_evaluations": repetitions * len(powers),
        "total_runtime_ms": float(duration * 1000.0),
        "mean_runtime_us_per_sweep": float(
            duration * 1_000_000.0 / (repetitions * len(powers))
        ),
        "peak_traced_memory_bytes": int(peak),
        "checksum": int(checksum),
        "machine_specific_secondary_evidence": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage5_helikite_scale_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage5/helikite_scale_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "helikite_scale_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite measured scale result v1")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if (
        governance["external_final_archives_may_be_opened"]
        or governance["external_final_signal_values_may_be_loaded"]
        or governance["fixed_policy_retuned"]
        or protocol["fixed_policy"]["parameters_may_be_retuned"]
    ):
        raise ValueError("measured scale governance is invalid")
    source = protocol["development_source"]
    pilot_result_path = PROJECT_DIR / source["pilot_result"]
    cache_path = PROJECT_DIR / source["cache"]
    stage2_path = PROJECT_DIR / protocol["link_accounting"]["stage2_config"]
    for path, expected in (
        (pilot_result_path, source["pilot_result_sha256"]),
        (cache_path, source["cache_sha256"]),
        (stage2_path, protocol["link_accounting"]["stage2_config_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"measured scale frozen input mismatch: {path}")
    pilot = json.loads(pilot_result_path.read_text(encoding="utf-8"))
    if not all(pilot["quality_gates"].values()):
        raise ValueError("helikite pilot quality gates are incomplete")
    cache = load_npz(cache_path)
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    link = replace(
        build_link_config(
            stage2, float(protocol["link_accounting"]["ebn0_db"])
        ),
        fec=protocol["link_accounting"]["fec"],
        channel=protocol["link_accounting"]["channel"],
    )
    results = {}
    engineering = {}
    for n_channels in grid["n_channels"]:
        n = int(n_channels)
        results[str(n)] = evaluate_n(
            cache[f"channel_power_n{n}"],
            cache["timestamps_local"],
            cache["cluster_ids"],
            n_channels=n,
            ratios=ratios,
            header=int(grid["application_header_bits"]),
            soft_bits=int(grid["soft_power_bits_per_channel"]),
            threshold_db=float(protocol["fixed_policy"]["regret_threshold_db"]),
            max_age_minutes=float(
                protocol["fixed_policy"]["max_state_age_minutes"]
            ),
            link=link,
        )
        engineering[str(n)] = benchmark(
            cache,
            n,
            ratios,
            int(protocol["engineering_benchmark"]["repetitions"]),
        )
    checks = {
        "query_bundle_clean_regret_equals_zero": all(
            results[str(n)]["summaries"]["query_bundle_always"][
                "maximum_regret_db"
            ]
            == 0.0
            for n in grid["n_channels"]
        ),
        "event_bundle_respects_frozen_regret_bound": all(
            results[str(n)]["summaries"]["query_event_bundle_fixed"][
                "maximum_regret_db"
            ]
            <= float(protocol["fixed_policy"]["regret_threshold_db"]) + 1e-12
            for n in grid["n_channels"]
        ),
        "bundle_application_bits_lower_than_soft_for_all_n": all(
            results[str(n)]["application_bits_per_update"][
                "query_bundle_always"
            ]
            < results[str(n)]["application_bits_per_update"][
                "soft_power_fixed4_always"
            ]
            for n in grid["n_channels"]
        ),
        "bundle_bit_reduction_increases_n8_to_n64": results["64"][
            "comparisons_vs_soft_power_fixed4"
        ]["query_bundle_always"]["application_bit_reduction_pct_vs_soft"]
        > results["8"]["comparisons_vs_soft_power_fixed4"][
            "query_bundle_always"
        ]["application_bit_reduction_pct_vs_soft"],
    }
    if not all(checks.values()):
        raise AssertionError(f"measured scale descriptive check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage5_helikite_measured_scale_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "pilot_result_sha256": source["pilot_result_sha256"],
            "cache_sha256": source["cache_sha256"],
            "stage2_config_sha256": protocol["link_accounting"][
                "stage2_config_sha256"
            ],
            "scene_count": len(cache["scene_ids"]),
            "cluster_count_15_minute": len(set(cache["cluster_ids"].astype(str))),
        },
        "descriptive_checks": checks,
        "n_results": results,
        "engineering_benchmark": engineering,
        "governance": {
            "pilot_permanently_excluded_from_final": True,
            "external_final_archives_opened": False,
            "external_final_signal_values_loaded": False,
            "fixed_policy_retuned": False,
            "confirmatory_final": False,
            "final_access_consumed": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "scene_count": result["inputs"]["scene_count"],
                "checks": checks,
                "bundle_application_bit_reduction_pct": {
                    n: results[n]["comparisons_vs_soft_power_fixed4"][
                        "query_bundle_always"
                    ]["application_bit_reduction_pct_vs_soft"]
                    for n in results
                },
                "final_access_consumed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
