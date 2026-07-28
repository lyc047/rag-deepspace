#!/usr/bin/env python
"""Evaluate context installation and compact event updates on pilot data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file  # noqa: E402
from spectrum_semcom.stage5_query_bundle import query_bundle_payload_width  # noqa: E402
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    ContextEventState,
    choose_context_event_update,
    decode_compact_update,
    decode_context_install,
    encode_context_install,
    next_context_event_state,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    SpectrumTaskState,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


@dataclass
class ExactState:
    actions: tuple[int, ...]
    success_timestamp: str


def _age_minutes(previous: str, current: str) -> float:
    return (
        datetime.fromisoformat(current) - datetime.fromisoformat(previous)
    ).total_seconds() / 60.0


def evaluate_n(
    powers: np.ndarray,
    timestamps: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    n_channels: int,
    ratios: tuple[float, ...],
    epsilon_db: float,
    max_age_minutes: float,
    stage5_header_bits: int,
    epoch: int,
) -> dict:
    demands = tuple(int(round(n_channels * ratio)) for ratio in ratios)
    queries = tuple(SpectrumTaskQuery(demand) for demand in demands)
    states = [
        build_task_state(values, queries, epsilon_db=epsilon_db)
        for values in powers
    ]
    codebook = fit_greedy_task_codebook(
        [states[int(index)] for index in train_indices]
    )
    sender_session = install_codebook(codebook, epoch=epoch)
    install_bits = encode_context_install(sender_session, node_id=1)
    receiver_install = decode_context_install(install_bits)
    receiver_session = receiver_install.session
    if receiver_session.manifest_sha256 != sender_session.manifest_sha256:
        raise AssertionError("installed sender and receiver codebooks differ")

    exact_packet_bits = (
        stage5_header_bits + query_bundle_payload_width(n_channels, demands)
    )
    semantic_state: ContextEventState | None = None
    exact_state: ExactState | None = None
    semantic_total_bits = 0
    exact_total_bits = 0
    semantic_updates = 0
    exact_updates = 0
    semantic_regrets = []
    exact_regrets = []
    fallback_updates = 0
    packet_round_trips = 0

    for index in evaluation_indices:
        task_state = states[int(index)]
        timestamp = str(timestamps[int(index)])
        semantic = choose_context_event_update(
            task_state,
            sender_session,
            cached_state=semantic_state,
            timestamp_local=timestamp,
            max_age_minutes=max_age_minutes,
            node_id=1,
        )
        if semantic.transmits:
            assert semantic.encoded_bits is not None
            decoded = decode_compact_update(
                semantic.encoded_bits, receiver_session
            )
            if decoded.decoder_actions != semantic.decoder_actions:
                raise AssertionError("compact event update round trip failed")
            semantic_state = next_context_event_state(
                semantic,
                timestamp_local=timestamp,
                codebook_epoch=sender_session.epoch,
            )
            semantic_total_bits += semantic.packet_bits
            semantic_updates += 1
            fallback_updates += int(decoded.uses_fallback)
            packet_round_trips += 1
        assert semantic_state is not None
        semantic_regrets.append(
            task_state.max_regret_db(semantic_state.decoder_actions)
        )

        exact_reuse = (
            float("inf")
            if exact_state is None
            else task_state.max_regret_db(exact_state.actions)
        )
        exact_age = (
            float("inf")
            if exact_state is None
            else _age_minutes(exact_state.success_timestamp, timestamp)
        )
        if (
            exact_state is None
            or exact_reuse > epsilon_db + 1e-12
            or exact_age > max_age_minutes
        ):
            exact_state = ExactState(task_state.optimal_actions, timestamp)
            exact_total_bits += exact_packet_bits
            exact_updates += 1
        exact_regrets.append(task_state.max_regret_db(exact_state.actions))

    scene_count = int(len(evaluation_indices))
    semantic_regret = np.asarray(semantic_regrets, dtype=np.float64)
    exact_regret = np.asarray(exact_regrets, dtype=np.float64)
    compact_regular_bits = 40 + codebook.symbol_width_bits
    per_update_saving = exact_packet_bits - compact_regular_bits
    break_even_updates = (
        int(math.ceil(install_bits.size / per_update_saving))
        if per_update_saving > 0
        else None
    )
    semantic_with_install = semantic_total_bits + int(install_bits.size)
    return {
        "n_channels": n_channels,
        "demands": list(demands),
        "evaluation_scene_count": scene_count,
        "codebook_size": len(codebook.codewords),
        "context_install_bits": int(install_bits.size),
        "compact_regular_update_bits": compact_regular_bits,
        "stage5_exact_update_bits": exact_packet_bits,
        "install_break_even_regular_update_count": break_even_updates,
        "semantic_update_count": semantic_updates,
        "semantic_update_rate": semantic_updates / scene_count,
        "semantic_fallback_update_count": fallback_updates,
        "stage5_update_count": exact_updates,
        "stage5_update_rate": exact_updates / scene_count,
        "packet_round_trip_count": packet_round_trips,
        "preprovisioned_semantic_total_bits": semantic_total_bits,
        "in_band_semantic_total_bits": semantic_with_install,
        "stage5_total_bits": exact_total_bits,
        "preprovisioned_mean_bits_per_scene": semantic_total_bits / scene_count,
        "in_band_mean_bits_per_scene": semantic_with_install / scene_count,
        "stage5_mean_bits_per_scene": exact_total_bits / scene_count,
        "preprovisioned_bit_reduction_pct_vs_stage5": float(
            100.0 * (exact_total_bits - semantic_total_bits) / exact_total_bits
        ),
        "in_band_bit_reduction_pct_vs_stage5": float(
            100.0
            * (exact_total_bits - semantic_with_install)
            / exact_total_bits
        ),
        "semantic_mean_regret_db": float(np.mean(semantic_regret)),
        "semantic_cvar_0_9_regret_db": empirical_cvar_numpy(
            semantic_regret, 0.9
        ),
        "semantic_maximum_regret_db": float(np.max(semantic_regret)),
        "stage5_mean_regret_db": float(np.mean(exact_regret)),
        "stage5_cvar_0_9_regret_db": empirical_cvar_numpy(exact_regret, 0.9),
        "stage5_maximum_regret_db": float(np.max(exact_regret)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR
        / "configs/stage6_context_event_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage6/context_event_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "context_event_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 context result")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    governance = protocol["governance"]
    if any(
        (
            governance["external_final_archives_may_be_opened"],
            governance["external_final_signal_values_may_be_loaded"],
            governance["external_final_access_may_be_consumed"],
            governance["output_is_confirmatory_final"],
        )
    ):
        raise ValueError("Stage-6 context governance is invalid")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("Stage-6 context cache hash mismatch")
    access_path = PROJECT_DIR / "configs/stage5_external_final_access_state.json"
    access_before = json.loads(access_path.read_text(encoding="utf-8"))
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    split = protocol["split"]
    train, evaluation, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            cache[split["group_field"]],
            seed=int(split["seed"]),
            train_fraction=float(split["train_group_fraction"]),
        )
    )
    grid = protocol["task_grid"]
    ratios = tuple(float(value) for value in grid["demand_ratios"])
    results = {
        str(n): evaluate_n(
            cache[f"channel_power_n{int(n)}"],
            cache["timestamps_local"],
            train,
            evaluation,
            n_channels=int(n),
            ratios=ratios,
            epsilon_db=float(grid["epsilon_db"]),
            max_age_minutes=float(grid["max_state_age_minutes"]),
            stage5_header_bits=int(grid["stage5_header_bits"]),
            epoch=int(grid["codebook_epoch"]),
        )
        for n in grid["n_channels"]
    }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_compact_packets_round_trip": all(
            value["packet_round_trip_count"] == value["semantic_update_count"]
            for value in results.values()
        ),
        "all_semantic_regret_respects_epsilon": all(
            value["semantic_maximum_regret_db"]
            <= float(grid["epsilon_db"]) + 1e-12
            for value in results.values()
        ),
        "all_stage5_regret_respects_epsilon": all(
            value["stage5_maximum_regret_db"]
            <= float(grid["epsilon_db"]) + 1e-12
            for value in results.values()
        ),
        "final_access_state_unchanged": access_before == access_after,
        "final_access_count_remains_zero": (
            access_after["access_count"] == 0
            and not access_after["final_signal_values_accessed"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Stage-6 context check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_context_event_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_sha256": source["cache_sha256"],
            "train_scene_count": int(train.size),
            "evaluation_scene_count": int(evaluation.size),
            "train_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
            "evaluation_groups_form_one_logical_session": bool(
                split["evaluation_groups_form_one_logical_session"]
            ),
        },
        "n_results": results,
        "checks": checks,
        "governance": {
            "development_only": True,
            "ack_faults_injected": False,
            "external_final_signal_values_loaded": False,
            "external_final_access_consumed": False,
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
                "checks": checks,
                "results": {
                    n: {
                        "install_bits": value["context_install_bits"],
                        "semantic_updates": value["semantic_update_count"],
                        "stage5_updates": value["stage5_update_count"],
                        "preprovisioned_reduction_pct": value[
                            "preprovisioned_bit_reduction_pct_vs_stage5"
                        ],
                        "in_band_reduction_pct": value[
                            "in_band_bit_reduction_pct_vs_stage5"
                        ],
                        "break_even_updates": value[
                            "install_break_even_regular_update_count"
                        ],
                        "max_regret_db": value[
                            "semantic_maximum_regret_db"
                        ],
                    }
                    for n, value in results.items()
                },
                "final_access_count": access_after["access_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
