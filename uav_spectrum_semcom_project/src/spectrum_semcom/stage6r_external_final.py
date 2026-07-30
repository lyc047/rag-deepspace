"""Frozen utilities for the Stage-6R ElectroSense external Final."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from spectrum_semcom.electrosense_psd import sha256_file
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    SpectrumTaskState,
    build_task_state,
)
from spectrum_semcom.stage6r_context_adapter import (
    context_signature,
    robust_spectral_shape_features,
    stable_signature,
)
from spectrum_semcom.stage6r_regret_codebook import (
    exhaustive_action_tuples,
    greedy_select_actions,
    regret_matrix,
)


def task_queries(
    n_channels: int, demand_ratios: Sequence[float]
) -> tuple[SpectrumTaskQuery, ...]:
    return tuple(
        SpectrumTaskQuery(int(round(n_channels * float(ratio))))
        for ratio in demand_ratios
    )


def build_states(
    power: np.ndarray,
    queries: Sequence[SpectrumTaskQuery],
    epsilon_db: float,
) -> list[SpectrumTaskState]:
    return [
        build_task_state(row, queries, epsilon_db=epsilon_db)
        for row in np.asarray(power, dtype=np.float64)
    ]


def synthetic_six_hour_timestamps(scene_count: int) -> np.ndarray:
    if scene_count < 2:
        raise ValueError("at least two scenes are required")
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    step_seconds = 6.0 * 60.0 * 60.0 / float(scene_count - 1)
    return np.asarray(
        [
            (start + timedelta(seconds=index * step_seconds)).isoformat()
            for index in range(scene_count)
        ]
    )


def resolve_context_and_actions(
    *,
    power: np.ndarray,
    queries: Sequence[SpectrumTaskQuery],
    epsilon_db: float,
    prototypes: dict[str, np.ndarray],
    novelty_threshold: float,
    bank_actions: dict[str, Sequence[Sequence[int]]],
    gate: dict,
    primary_k: int,
) -> dict:
    values = np.asarray(power, dtype=np.float64)
    states = build_states(values, queries, epsilon_db)
    all_actions = exhaustive_action_tuples(states)
    all_regrets = regret_matrix(states, all_actions)
    action_lookup = {action: index for index, action in enumerate(all_actions)}
    signatures: list[str] = []
    trace: list[dict] = []
    resolution: dict | None = None
    window = int(gate["window_size_scenes"])
    required = int(gate["required_consecutive_matching_windows"])
    minimum_coverage = float(gate["minimum_accumulated_bank_coverage"])
    for checkpoint in map(int, gate["checkpoints_scenes"]):
        if checkpoint > len(states):
            break
        signature, source, distance = context_signature(
            values[checkpoint - window : checkpoint],
            prototypes,
            novelty_threshold,
        )
        signatures.append(signature)
        stable = stable_signature(signatures, required)
        trace_row = {
            "checkpoint_scenes": checkpoint,
            "signature": signature,
            "nearest_source": source,
            "context_distance": float(distance),
            "novelty_threshold": float(novelty_threshold),
            "stable_signature": stable,
            "accumulated_bank_coverage": None,
            "accepted": False,
        }
        if stable == "OOD":
            calibration_regrets = all_regrets[:checkpoint]
            selected = greedy_select_actions(
                regrets=calibration_regrets,
                actions=all_actions,
                epsilon_db=epsilon_db,
                weights=np.ones(checkpoint, dtype=np.float64),
                max_codewords=primary_k,
            ).actions
            resolution = {
                "decision": "OOD",
                "adapted": True,
                "checkpoint": checkpoint,
                "selected_actions": selected,
            }
            trace_row["accepted"] = True
        elif isinstance(stable, str) and stable.startswith("BANK:"):
            bank_source = stable.removeprefix("BANK:")
            selected = tuple(
                tuple(map(int, row)) for row in bank_actions[bank_source]
            )
            selected_indices = np.asarray(
                [action_lookup[action] for action in selected], dtype=int
            )
            coverage = float(
                np.mean(
                    np.min(
                        all_regrets[:checkpoint, selected_indices], axis=1
                    )
                    <= epsilon_db + 1e-12
                )
            )
            trace_row["accumulated_bank_coverage"] = coverage
            if coverage >= minimum_coverage:
                resolution = {
                    "decision": stable,
                    "adapted": False,
                    "checkpoint": checkpoint,
                    "selected_actions": selected,
                }
                trace_row["accepted"] = True
        trace.append(trace_row)
        if resolution is not None:
            break
    if resolution is None:
        calibration_count = min(
            int(gate["maximum_calibration_scenes"]), len(states)
        )
        return {
            "decision": "UNRESOLVED_EXACT_FALLBACK",
            "adapted": False,
            "resolved": False,
            "calibration_scenes": calibration_count,
            "selected_actions": [],
            "decision_trace": trace,
            "states": states,
            "future_compact_coverage": 0.0,
            "future_escape_rate": 1.0,
        }
    calibration_count = int(resolution["checkpoint"])
    selected_actions = tuple(resolution["selected_actions"])
    selected_indices = np.asarray(
        [action_lookup[action] for action in selected_actions], dtype=int
    )
    future_regrets = all_regrets[calibration_count:, selected_indices]
    if future_regrets.shape[0] == 0:
        future_coverage = 0.0
    else:
        future_coverage = float(
            np.mean(
                np.min(future_regrets, axis=1) <= epsilon_db + 1e-12
            )
        )
    return {
        "decision": resolution["decision"],
        "adapted": bool(resolution["adapted"]),
        "resolved": True,
        "calibration_scenes": calibration_count,
        "selected_actions": [list(row) for row in selected_actions],
        "decision_trace": trace,
        "states": states,
        "future_compact_coverage": future_coverage,
        "future_escape_rate": 1.0 - future_coverage,
    }


def code_snapshot(
    project_dir: Path, relative_paths: Sequence[str]
) -> dict:
    hashes = {
        relative: sha256_file(project_dir / relative)
        for relative in relative_paths
    }
    payload = json.dumps(
        hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": hashes,
        "combined_sha256": hashlib.sha256(payload).hexdigest(),
    }


def hierarchical_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
    site_weights: np.ndarray | None = None,
) -> dict:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or min(rows.shape) < 1:
        raise ValueError("values must have shape [site, paired_trajectory]")
    weights = (
        np.ones(rows.shape[0], dtype=np.float64)
        if site_weights is None
        else np.asarray(site_weights, dtype=np.float64).reshape(-1)
    )
    if weights.shape != (rows.shape[0],) or np.any(weights <= 0):
        raise ValueError("invalid site weights")
    point = float(np.average(np.mean(rows, axis=1), weights=weights))
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        site_sample = rng.integers(0, rows.shape[0], size=rows.shape[0])
        trajectory_sample = rng.integers(
            0, rows.shape[1], size=rows.shape[1]
        )
        sampled = rows[np.ix_(site_sample, trajectory_sample)]
        boot[index] = np.average(
            np.mean(sampled, axis=1), weights=weights[site_sample]
        )
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": point,
        "ci_lower": float(np.quantile(boot, alpha)),
        "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
        "bootstrap_unit": "site_then_paired_trajectory",
    }

