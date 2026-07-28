#!/usr/bin/env python
"""Validate the Stage-6 task codebook with its real versioned bitstream."""

from __future__ import annotations

import argparse
import json
import sys
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
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import (  # noqa: E402
    decode_task_update,
    encode_task_update,
    install_codebook,
)


def evaluate_n(
    powers: np.ndarray,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    n_channels: int,
    ratios: tuple[float, ...],
    epsilon_db: float,
    stage5_header_bits: int,
    stage6_header_bits: int,
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
    session = install_codebook(codebook, epoch=epoch)
    packet_bits = []
    regrets = []
    fallback = []
    for index in evaluation_indices:
        state = states[int(index)]
        encoded = encode_task_update(
            state,
            session,
            node_id=1,
            application_header_bits=stage6_header_bits,
        )
        decoded = decode_task_update(
            encoded.bits,
            session,
            application_header_bits=stage6_header_bits,
        )
        if decoded.decoder_actions != encoded.decision.decoder_actions:
            raise AssertionError("task codec action round trip failed")
        packet_bits.append(int(encoded.bits.size))
        regrets.append(encoded.decision.max_regret_db)
        fallback.append(encoded.decision.uses_fallback)
    exact_payload = query_bundle_payload_width(n_channels, demands)
    stage5_bits = stage5_header_bits + exact_payload
    packet_array = np.asarray(packet_bits, dtype=np.float64)
    regret_array = np.asarray(regrets, dtype=np.float64)
    fallback_array = np.asarray(fallback, dtype=np.bool_)
    return {
        "n_channels": n_channels,
        "demands": list(demands),
        "codebook_size": len(codebook.codewords),
        "manifest_sha256": session.manifest_sha256,
        "packet_tag": session.packet_tag,
        "stage5_exact_bundle_application_bits": stage5_bits,
        "stage6_mean_real_packet_bits": float(np.mean(packet_array)),
        "stage6_minimum_real_packet_bits": int(np.min(packet_array)),
        "stage6_maximum_real_packet_bits": int(np.max(packet_array)),
        "real_packet_bit_reduction_pct_vs_stage5_exact_bundle": float(
            100.0 * (stage5_bits - np.mean(packet_array)) / stage5_bits
        ),
        "evaluation_codeword_coverage_rate": float(
            np.mean(~fallback_array)
        ),
        "evaluation_fallback_rate": float(np.mean(fallback_array)),
        "evaluation_mean_max_query_regret_db": float(np.mean(regret_array)),
        "evaluation_cvar_0_9_max_query_regret_db": empirical_cvar_numpy(
            regret_array, 0.9
        ),
        "evaluation_maximum_max_query_regret_db": float(np.max(regret_array)),
        "round_trip_scene_count": int(len(evaluation_indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_DIR / "configs/stage6_task_codec_development_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "results/stage6/task_codec_development_v1",
    )
    args = parser.parse_args()
    output = args.out_dir / "task_codec_result.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 codec result")
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
        raise ValueError("Stage-6 codec governance is invalid")
    source = protocol["development_source"]
    cache_path = PROJECT_DIR / source["cache"]
    if sha256_file(cache_path) != source["cache_sha256"]:
        raise ValueError("Stage-6 codec cache hash mismatch")
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
            train,
            evaluation,
            n_channels=int(n),
            ratios=ratios,
            epsilon_db=float(grid["epsilon_db"]),
            stage5_header_bits=int(grid["stage5_header_bits"]),
            stage6_header_bits=int(grid["stage6_codec_header_bits"]),
            epoch=int(grid["codebook_epoch"]),
        )
        for n in grid["n_channels"]
    }
    access_after = json.loads(access_path.read_text(encoding="utf-8"))
    checks = {
        "all_packets_round_trip": all(
            value["round_trip_scene_count"] == int(evaluation.size)
            for value in results.values()
        ),
        "all_decisions_respect_epsilon": all(
            value["evaluation_maximum_max_query_regret_db"]
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
        raise AssertionError(f"Stage-6 codec check failed: {checks}")
    result = {
        "version": "1.0",
        "status": "stage6_task_codec_development_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "inputs": {
            "cache_sha256": source["cache_sha256"],
            "train_scene_count": int(train.size),
            "evaluation_scene_count": int(evaluation.size),
            "train_groups": list(train_groups),
            "evaluation_groups": list(evaluation_groups),
        },
        "n_results": results,
        "checks": checks,
        "governance": {
            "development_only": True,
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
                        "stage5_bits": value[
                            "stage5_exact_bundle_application_bits"
                        ],
                        "stage6_bits": value["stage6_mean_real_packet_bits"],
                        "bit_reduction_pct": value[
                            "real_packet_bit_reduction_pct_vs_stage5_exact_bundle"
                        ],
                        "coverage": value[
                            "evaluation_codeword_coverage_rate"
                        ],
                        "max_regret_db": value[
                            "evaluation_maximum_max_query_regret_db"
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
