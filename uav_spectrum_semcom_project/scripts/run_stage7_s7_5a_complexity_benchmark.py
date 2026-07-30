#!/usr/bin/env python
"""Benchmark the frozen task-codebook and compact codec on the current CPU."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file
from spectrum_semcom.stage6_context_codec import (
    decode_compact_update,
    encode_compact_update,
    encode_context_install,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage7_s7_5a_complexity_benchmark_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_5a_complexity_benchmark_v1/result.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _spectra(
    rng: np.random.Generator, scenes: int, n_channels: int
) -> np.ndarray:
    time_axis = np.arange(scenes, dtype=np.float64)[:, None]
    frequency_axis = np.linspace(0.0, 2.0 * np.pi, n_channels)[None, :]
    slow = 4.0 * np.sin(time_axis / 31.0 + frequency_axis)
    occupancy = 7.0 * (
        np.sin(time_axis / 17.0 - 2.0 * frequency_axis) > 0.72
    )
    return -95.0 + slow + occupancy + rng.normal(
        0.0, 1.5, size=(scenes, n_channels)
    )


def _latency_batches(
    state,
    session,
    warmup: int,
    batches: int,
    calls_per_batch: int,
) -> np.ndarray:
    packet, _ = encode_compact_update(
        state, session, node_id=1, update_epoch=1
    )
    for index in range(warmup):
        packet, _ = encode_compact_update(
            state,
            session,
            node_id=1,
            update_epoch=index % 256,
        )
        decode_compact_update(packet, session)
    rows = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for batch in range(batches):
            started = time.perf_counter_ns()
            for index in range(calls_per_batch):
                packet, _ = encode_compact_update(
                    state,
                    session,
                    node_id=1,
                    update_epoch=(batch + index) % 256,
                )
                decode_compact_update(packet, session)
            elapsed = time.perf_counter_ns() - started
            rows.append(elapsed / calls_per_batch / 1_000_000.0)
    finally:
        if gc_enabled:
            gc.enable()
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite benchmark result")
    config = _read(args.config)
    task = config["task"]
    benchmark = config["benchmark"]
    gates = config["development_gates"]
    started = time.perf_counter()
    n_results = {}
    passing = 0
    for n_position, n_channels in enumerate(task["n_channels"]):
        n_channels = int(n_channels)
        queries = tuple(
            SpectrumTaskQuery(
                max(1, min(n_channels, int(round(n_channels * ratio))))
            )
            for ratio in task["demand_ratios"]
        )
        rng = np.random.default_rng(
            int(task["random_seed"]) + n_position * 1009
        )
        tracemalloc.start()
        spectra = _spectra(
            rng, int(task["synthetic_scene_count"]), n_channels
        )
        states = [
            build_task_state(
                row, queries, epsilon_db=float(task["epsilon_db"])
            )
            for row in spectra
        ]
        fit_started = time.perf_counter()
        codebook = fit_greedy_task_codebook(states)
        fit_seconds = time.perf_counter() - fit_started
        session = install_codebook(codebook, epoch=2)
        install_bits = int(encode_context_install(session, node_id=1).size)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        latency = _latency_batches(
            states[-1],
            session,
            int(benchmark["warmup_calls"]),
            int(benchmark["timed_batches"]),
            int(benchmark["calls_per_batch"]),
        )
        packet, decision = encode_compact_update(
            states[-1], session, node_id=1, update_epoch=1
        )
        row = {
            "n_channels": n_channels,
            "query_demands": [
                int(query.demand_channels) for query in queries
            ],
            "training_scenes": len(states),
            "codeword_count": len(codebook.codewords),
            "symbol_width_bits": int(codebook.symbol_width_bits),
            "codebook_fit_seconds": float(fit_seconds),
            "compact_encode_decode_ms": {
                "median": float(np.median(latency)),
                "p95": float(np.quantile(latency, 0.95)),
                "maximum": float(np.max(latency)),
                "batch_count": int(latency.size),
                "calls_per_batch": int(benchmark["calls_per_batch"]),
            },
            "peak_python_allocation_mib": float(
                peak_bytes / (1024.0 * 1024.0)
            ),
            "pickle_codebook_bytes": len(
                pickle.dumps(codebook, protocol=5)
            ),
            "context_install_bits": install_bits,
            "compact_example_bits": int(packet.size),
            "compact_example_uses_escape": bool(decision.uses_fallback),
            "session_start_amortized_bits_per_scene": {
                str(length): float(
                    (
                        int(benchmark["startup_activation_bits"])
                        + int(benchmark["startup_ack_bits"])
                    )
                    / int(length)
                )
                for length in benchmark["session_lengths_scenes"]
            },
        }
        row["gates"] = {
            "latency_passed": (
                row["compact_encode_decode_ms"]["p95"]
                <= float(gates["maximum_compact_encode_decode_p95_ms"])
            ),
            "fit_passed": (
                fit_seconds <= float(gates["maximum_codebook_fit_seconds"])
            ),
            "memory_passed": (
                row["peak_python_allocation_mib"]
                <= float(gates["maximum_peak_python_allocation_mib"])
            ),
        }
        row["n_gate_passed"] = all(row["gates"].values())
        passing += int(row["n_gate_passed"])
        n_results[str(n_channels)] = row
        print(
            f"S7.5A N={n_channels} p95="
            f"{row['compact_encode_decode_ms']['p95']:.4f} ms "
            f"fit={fit_seconds:.3f} s "
            f"peak={row['peak_python_allocation_mib']:.2f} MiB",
            flush=True,
        )
    required = int(gates["minimum_passing_n_count"])
    result = {
        "version": "1.0",
        "status": "stage7_s7_5a_complexity_benchmark_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "n_results": n_results,
        "decision": {
            "passing_n_count": passing,
            "required_n_count": required,
            "development_gate_passed": passing >= required,
            "next_action": (
                "write_frozen_architecture_complexity_and_session_cost_report"
                if passing >= required
                else "profile_failed_complexity_dimension_before_claiming_feasibility"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
