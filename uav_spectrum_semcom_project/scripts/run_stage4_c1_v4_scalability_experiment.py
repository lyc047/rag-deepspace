#!/usr/bin/env python
"""Run the post-Final C1-v4 fixed-task scalability experiment on development sweeps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage2_digital_link import build_link_config  # noqa: E402
from spectrum_semcom.aerpaw_spectrum import AerpawZipPair  # noqa: E402
from spectrum_semcom.c1_temporal_statistics import (  # noqa: E402
    empirical_cvar_numpy,
    paired_cluster_bootstrap,
)
from spectrum_semcom.c1_v4_block_semantics import (  # noqa: E402
    block_regret_db,
    build_block_semantic_scene,
    load_aerpaw_zip_power_sweep,
    quantize_unit_interval,
    selected_block_start,
)
from spectrum_semcom.digital_link import transmit_payload_analytic  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402

METHODS = ("soft_power_fixed4", "best_block_onehot", "best_block_binary_index")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_from_stem(stem: str) -> str:
    return datetime.strptime(stem.removeprefix("results_"), "%Y%m%d_%H%M%S").isoformat()


def task_id(n_channels: int, demand_channels: int) -> str:
    return f"N{n_channels}_D{demand_channels}"


def method_application_bits(
    method: str,
    n_channels: int,
    demand_channels: int,
    overhead_bits: int,
) -> int:
    candidates = n_channels - demand_channels + 1
    if method == "soft_power_fixed4":
        payload = 4 * n_channels
    elif method == "best_block_onehot":
        payload = candidates
    elif method == "best_block_binary_index":
        payload = int(math.ceil(math.log2(candidates)))
    else:
        raise KeyError(method)
    return int(overhead_bits + payload)


def validate_inputs(protocol_path: Path, protocol: dict) -> tuple[dict, dict, dict]:
    source = protocol["development_source"]
    cache_result_path = PROJECT_DIR / source["cache_result"]
    provenance_path = PROJECT_DIR / source["provenance"]
    stage2_path = PROJECT_DIR / protocol["link"]["stage2_config"]
    for path in (cache_result_path, provenance_path, stage2_path):
        if "c1_v4_final" in path.as_posix().lower():
            raise ValueError("scalability experiment must not load a C1-v4 Final path")
    expected = (
        (cache_result_path, source["cache_result_sha256"]),
        (provenance_path, source["provenance_sha256"]),
        (stage2_path, protocol["link"]["stage2_config_sha256"]),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path}")
    if protocol["governance"]["final_measurements_or_final_metrics_may_be_loaded"]:
        raise ValueError("protocol must prohibit Final inputs")
    cache_result = load_json(cache_result_path)
    if cache_result.get("governance", {}).get("development_only") is not True:
        raise ValueError("source cache is not marked development-only")
    split = cache_result["splits"][source["split"]]
    cache_path = PROJECT_DIR / split["cache"]
    if sha256_file(cache_path) != split["cache_sha256"]:
        raise ValueError("development metadata cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as handle:
        metadata = {key: handle[key] for key in handle.files}
    provenance = load_json(provenance_path)
    if len(provenance["scenes"]) != source["scene_count"]:
        raise ValueError("provenance scene count differs from frozen protocol")
    if set(metadata["scene_ids"].astype(str)) != {
        row["scene_id"] for row in provenance["scenes"]
    }:
        raise ValueError("metadata cache and provenance scene IDs differ")
    return cache_result, provenance, metadata


def build_task_arrays(
    protocol: dict,
    cache_result: dict,
    provenance: dict,
    metadata: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    scene_position = {
        scene_id: index for index, scene_id in enumerate(metadata["scene_ids"].astype(str))
    }
    archive_paths = {
        site: Path(row["path"]).resolve()
        for site, row in cache_result["archive_audit"].items()
    }
    for site, path in archive_paths.items():
        declared = cache_result["archive_audit"][site]
        if not path.is_file() or path.stat().st_size != int(declared["size_bytes"]):
            raise FileNotFoundError(f"verified development archive is unavailable: {path}")
    tasks = []
    for n_channels in protocol["channel_counts"]:
        for ratio in protocol["demand_ratios"]:
            demand = int(round(n_channels * ratio))
            if abs(demand / n_channels - ratio) > 1e-12:
                raise ValueError("demand ratio is not exact for a selected channel count")
            tasks.append(
                {
                    "task_id": task_id(n_channels, demand),
                    "n_channels": int(n_channels),
                    "demand_channels": demand,
                    "demand_ratio": float(ratio),
                    "candidate_blocks": int(n_channels - demand + 1),
                }
            )
    scene_count = len(metadata["scene_ids"])
    clean_regret = np.zeros((len(tasks), len(METHODS), scene_count), dtype=np.float64)
    fallback_regret = np.zeros((len(tasks), scene_count), dtype=np.float64)
    archive_handles = {
        site: zipfile.ZipFile(path) for site, path in archive_paths.items()
    }
    try:
        for row_index, row in enumerate(provenance["scenes"]):
            position = scene_position[row["scene_id"]]
            site = row["site"]
            archive = archive_handles[site]
            meta_info = archive.getinfo(row["meta_member"])
            data_info = archive.getinfo(row["data_member"])
            if (
                f"{meta_info.CRC:08x}" != row["meta_crc32"]
                or meta_info.file_size != int(row["meta_size"])
                or f"{data_info.CRC:08x}" != row["data_crc32"]
                or data_info.file_size != int(row["data_size"])
            ):
                raise ValueError(f"ZIP member provenance mismatch: {row['scene_id']}")
            pair = AerpawZipPair(
                stem=row["source_stem"],
                archive_path=archive_paths[site],
                meta_member=row["meta_member"],
                data_member=row["data_member"],
                timestamp_local=timestamp_from_stem(row["source_stem"]),
            )
            sweep = load_aerpaw_zip_power_sweep(pair, archive)
            for task_index, task in enumerate(tasks):
                scene = build_block_semantic_scene(
                    sweep,
                    frequency_low_mhz=float(protocol["frequency_band_mhz"][0]),
                    frequency_high_mhz=float(protocol["frequency_band_mhz"][1]),
                    n_channels=task["n_channels"],
                    demand_channels=task["demand_channels"],
                )
                soft_values = quantize_unit_interval(scene.normalized_channel_power, 4)
                soft_start = selected_block_start(
                    soft_values, "channel", task["demand_channels"]
                )
                best_start = int(np.argmin(scene.block_cost_dbm))
                clean_regret[task_index, 0, position] = block_regret_db(
                    soft_start, scene.block_cost_dbm
                )
                clean_regret[task_index, 1, position] = block_regret_db(
                    best_start, scene.block_cost_dbm
                )
                clean_regret[task_index, 2, position] = clean_regret[
                    task_index, 1, position
                ]
                fallback_regret[task_index, position] = block_regret_db(
                    0, scene.block_cost_dbm
                )
            if (row_index + 1) % 30 == 0:
                print(
                    f"raw development sweeps: {row_index + 1}/{len(provenance['scenes'])}",
                    flush=True,
                )
    finally:
        for handle in archive_handles.values():
            handle.close()
    arrays = {
        "clean_regret": clean_regret,
        "fallback_regret": fallback_regret,
    }
    return tasks, arrays


def summarize_scene_rows(rows: list[dict], alpha: float) -> dict:
    regret = np.asarray([row["regret_db"] for row in rows], dtype=np.float64)
    return {
        "scene_count": len(rows),
        "mean_clean_regret_db": float(
            np.mean([row["clean_regret_db"] for row in rows])
        ),
        "mean_regret_db": float(np.mean(regret)),
        "cvar_0_9_regret_db": empirical_cvar_numpy(regret, alpha),
        "mean_actual_bits": float(np.mean([row["actual_bits"] for row in rows])),
        "mean_frame_success_rate": float(
            np.mean([row["frame_success_rate"] for row in rows])
        ),
        "mean_link_latency_ms": float(
            np.mean([row["link_latency_ms"] for row in rows])
        ),
        "application_bits": int(rows[0]["application_bits"]),
        "payload_bits": int(rows[0]["payload_bits"]),
    }


def run_link(
    protocol: dict,
    tasks: list[dict],
    arrays: dict,
    metadata: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    stage2 = load_json(PROJECT_DIR / protocol["link"]["stage2_config"])
    scene_count = len(metadata["scene_ids"])
    task_count = len(tasks)
    method_count = len(METHODS)
    channel_count = len(protocol["link"]["channel_models"])
    ebn0_count = len(protocol["link"]["ebn0_db"])
    scene_regret = np.zeros((task_count, method_count, scene_count), dtype=np.float64)
    scene_bits = np.zeros_like(scene_regret)
    scene_success = np.zeros_like(scene_regret)
    scene_latency = np.zeros_like(scene_regret)
    condition_regret = np.zeros(
        (task_count, method_count, channel_count, ebn0_count), dtype=np.float64
    )
    condition_bits = np.zeros_like(condition_regret)
    condition_success = np.zeros_like(condition_regret)
    condition_latency = np.zeros_like(condition_regret)
    app_bits = np.asarray(
        [
            [
                method_application_bits(
                    method,
                    task["n_channels"],
                    task["demand_channels"],
                    int(protocol["application_overhead_bits"]),
                )
                for method in METHODS
            ]
            for task in tasks
        ],
        dtype=int,
    )
    unique_lengths = sorted(set(int(value) for value in app_bits.reshape(-1)))
    condition_total = channel_count * ebn0_count
    repeats = int(protocol["link"]["repeats"])
    for channel_index, channel in enumerate(protocol["link"]["channel_models"]):
        for ebn0_index, ebn0 in enumerate(protocol["link"]["ebn0_db"]):
            link = replace(build_link_config(stage2, float(ebn0)), channel=channel)
            for position in range(scene_count):
                tx_by_length = {}
                for application_bits in unique_lengths:
                    bit_values, success_values, latency_values = [], [], []
                    for repetition in range(repeats):
                        seed = (
                            83260723
                            + position * 100_000
                            + channel_index * 1000
                            + ebn0_index * 100
                            + repetition
                        )
                        tx = transmit_payload_analytic(application_bits, link, seed)
                        bit_values.append(tx.transmitted_bits)
                        success_values.append(float(tx.frame_success))
                        latency_values.append(tx.duration_s * 1000.0)
                    tx_by_length[application_bits] = (
                        float(np.mean(bit_values)),
                        float(np.mean(success_values)),
                        float(np.mean(latency_values)),
                    )
                for task_index in range(task_count):
                    fallback = arrays["fallback_regret"][task_index, position]
                    for method_index in range(method_count):
                        bits, success, latency = tx_by_length[
                            int(app_bits[task_index, method_index])
                        ]
                        clean = arrays["clean_regret"][
                            task_index, method_index, position
                        ]
                        regret = success * clean + (1.0 - success) * fallback
                        scene_regret[task_index, method_index, position] += regret
                        scene_bits[task_index, method_index, position] += bits
                        scene_success[task_index, method_index, position] += success
                        scene_latency[task_index, method_index, position] += latency
                        condition_regret[
                            task_index, method_index, channel_index, ebn0_index
                        ] += regret
                        condition_bits[
                            task_index, method_index, channel_index, ebn0_index
                        ] += bits
                        condition_success[
                            task_index, method_index, channel_index, ebn0_index
                        ] += success
                        condition_latency[
                            task_index, method_index, channel_index, ebn0_index
                        ] += latency
            print(
                f"link grid: {channel} {ebn0:g} dB complete "
                f"({channel_index * ebn0_count + ebn0_index + 1}/{condition_total})",
                flush=True,
            )
    for value in (scene_regret, scene_bits, scene_success, scene_latency):
        value /= condition_total
    for value in (
        condition_regret,
        condition_bits,
        condition_success,
        condition_latency,
    ):
        value /= scene_count
    scene_rows = []
    for task_index, task in enumerate(tasks):
        for method_index, method in enumerate(METHODS):
            payload = int(app_bits[task_index, method_index]) - int(
                protocol["application_overhead_bits"]
            )
            for position in range(scene_count):
                scene_rows.append(
                    {
                        **task,
                        "method": method,
                        "scene_id": str(metadata["scene_ids"][position]),
                        "site": str(metadata["site_ids"][position]),
                        "cluster_id": str(metadata["cluster_ids"][position]),
                        "application_bits": int(app_bits[task_index, method_index]),
                        "payload_bits": payload,
                        "clean_regret_db": float(
                            arrays["clean_regret"][
                                task_index, method_index, position
                            ]
                        ),
                        "regret_db": float(
                            scene_regret[task_index, method_index, position]
                        ),
                        "actual_bits": float(
                            scene_bits[task_index, method_index, position]
                        ),
                        "frame_success_rate": float(
                            scene_success[task_index, method_index, position]
                        ),
                        "link_latency_ms": float(
                            scene_latency[task_index, method_index, position]
                        ),
                    }
                )
    condition_rows = []
    for task_index, task in enumerate(tasks):
        for method_index, method in enumerate(METHODS):
            for channel_index, channel in enumerate(
                protocol["link"]["channel_models"]
            ):
                for ebn0_index, ebn0 in enumerate(protocol["link"]["ebn0_db"]):
                    condition_rows.append(
                        {
                            **task,
                            "method": method,
                            "channel": channel,
                            "ebn0_db": float(ebn0),
                            "mean_regret_db": float(
                                condition_regret[
                                    task_index,
                                    method_index,
                                    channel_index,
                                    ebn0_index,
                                ]
                            ),
                            "mean_actual_bits": float(
                                condition_bits[
                                    task_index,
                                    method_index,
                                    channel_index,
                                    ebn0_index,
                                ]
                            ),
                            "mean_frame_success_rate": float(
                                condition_success[
                                    task_index,
                                    method_index,
                                    channel_index,
                                    ebn0_index,
                                ]
                            ),
                            "mean_link_latency_ms": float(
                                condition_latency[
                                    task_index,
                                    method_index,
                                    channel_index,
                                    ebn0_index,
                                ]
                            ),
                        }
                    )
    return scene_rows, condition_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage4_c1_v4_scalability_experiment.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage4/c1_v4_scalability_v1",
    )
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.out_dir = args.out_dir.resolve()
    output = args.out_dir / "scalability_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite scalability result v1")
    protocol = load_json(args.protocol)
    cache_result, provenance, metadata = validate_inputs(args.protocol, protocol)
    tasks, arrays = build_task_arrays(protocol, cache_result, provenance, metadata)
    scene_rows, condition_rows = run_link(protocol, tasks, arrays, metadata)
    alpha = float(protocol["statistics"]["cvar_alpha"])
    summary = {}
    comparisons = {}
    strong_targets = {}
    for task_index, task in enumerate(tasks):
        task_rows = [row for row in scene_rows if row["task_id"] == task["task_id"]]
        summary[task["task_id"]] = {}
        by_method = {
            method: [row for row in task_rows if row["method"] == method]
            for method in METHODS
        }
        for method, rows in by_method.items():
            summary[task["task_id"]][method] = summarize_scene_rows(rows, alpha)
        soft = by_method["soft_power_fixed4"]
        comparisons[task["task_id"]] = {}
        for method_index, method in enumerate(METHODS[1:]):
            proposed = by_method[method]
            inference = paired_cluster_bootstrap(
                np.asarray([row["regret_db"] for row in proposed]),
                np.asarray([row["regret_db"] for row in soft]),
                np.asarray([row["actual_bits"] for row in proposed]),
                np.asarray([row["actual_bits"] for row in soft]),
                [row["cluster_id"] for row in soft],
                repetitions=int(
                    protocol["statistics"]["paired_cluster_bootstrap_repetitions"]
                ),
                seed=int(protocol["statistics"]["bootstrap_seed"])
                + task_index * 10
                + method_index,
                cvar_alpha=alpha,
                one_sided_confidence=float(
                    protocol["statistics"]["one_sided_confidence"]
                ),
            )
            soft_bits = summary[task["task_id"]]["soft_power_fixed4"][
                "mean_actual_bits"
            ]
            proposed_bits = summary[task["task_id"]][method]["mean_actual_bits"]
            inference["actual_bit_reduction_pct"] = float(
                100.0 * (soft_bits - proposed_bits) / soft_bits
            )
            inference["mean_regret_noninferior"] = bool(
                inference["mean_regret_upper_bound"] <= 0.0
            )
            inference["reaches_20pct_strong_target"] = bool(
                inference["actual_bit_reduction_pct"]
                >= float(
                    protocol["descriptive_targets"][
                        "target_actual_bit_reduction_pct"
                    ]
                )
                and inference["mean_regret_noninferior"]
            )
            comparisons[task["task_id"]][method] = inference
        strong_targets[task["task_id"]] = {
            method: comparisons[task["task_id"]][method][
                "reaches_20pct_strong_target"
            ]
            for method in METHODS[1:]
        }
    result = {
        "version": "1.0",
        "status": "postfinal_development_scalability_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "task_grid_sha256": canonical_json_sha256(tasks),
        "inputs": {
            "scene_count": len(metadata["scene_ids"]),
            "cluster_count": len(set(metadata["cluster_ids"].astype(str))),
            "source_cache_result_sha256": protocol["development_source"][
                "cache_result_sha256"
            ],
            "source_provenance_sha256": protocol["development_source"][
                "provenance_sha256"
            ],
        },
        "tasks": tasks,
        "methods": list(METHODS),
        "summary": summary,
        "comparisons_vs_soft_power_fixed4": comparisons,
        "strong_target_by_task": strong_targets,
        "strong_target_task_count": {
            method: sum(strong_targets[task["task_id"]][method] for task in tasks)
            for method in METHODS[1:]
        },
        "condition_rows": condition_rows,
        "scene_rows": scene_rows,
        "governance": {
            "final_measurement_values_loaded": False,
            "final_metrics_loaded": False,
            "changes_final_algorithm_or_claim": False,
            "compact_binary_index_confirmed": False,
            "data_role": protocol["development_source"]["role"],
            "confirmatory": False,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "tasks": len(tasks),
                "scenes": len(metadata["scene_ids"]),
                "onehot_strong_targets": result["strong_target_task_count"][
                    "best_block_onehot"
                ],
                "binary_index_strong_targets": result["strong_target_task_count"][
                    "best_block_binary_index"
                ],
                "final_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
