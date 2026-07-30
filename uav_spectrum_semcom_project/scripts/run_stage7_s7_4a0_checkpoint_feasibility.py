#!/usr/bin/env python
"""Audit S7.4A-0 self-contained checkpoint schema and bit feasibility."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    load_npy_members,
    read_json,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    encode_compact_update,
)
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
    codebook_activation_frame_bits,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    resolve_context_and_actions,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)
from spectrum_semcom.stage7_self_contained_checkpoint import (  # noqa: E402
    decode_self_contained_checkpoint,
    encode_self_contained_checkpoint,
    minimum_checkpoint_bits,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage7_s7_4a0_checkpoint_feasibility_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_4a0_checkpoint_feasibility_v1/result.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite S7.4A-0 result")
    config = read_json(args.config)
    paths = {
        key: PROJECT_DIR / value
        for key, value in config["inputs"].items()
    }
    split = read_json(paths["development_split"])["split"]
    protocol = read_json(paths["electrosense_protocol"])
    registry = read_json(paths["electrosense_registry"])
    access = read_json(paths["electrosense_access_state"])
    freeze = read_json(paths["electrosense_freeze"])
    source = read_json(paths["s7_3b_result"])
    train_sites = list(split["train"])
    validation_sites = list(split["validation"])
    if (
        source["decision"]["diagnostic_gate_passed"]
        or len(train_sites)
        != int(config["data"]["expected_train_site_count"])
        or len(validation_sites)
        != int(config["data"]["expected_validation_site_count"])
        or int(access["roles"]["reserve"]["access_count"]) != 0
    ):
        raise ValueError("S7.4A-0 governance or trigger mismatch")
    metadata = registry["selected_site_members"]
    members = [metadata[site]["member"] for site in validation_sites]
    loaded = load_npy_members(Path(protocol["inputs"]["archive"]), members)
    ack_bits = int(cumulative_ack_payload_bits())
    activation_bits = int(codebook_activation_frame_bits())
    started = time.perf_counter()
    n_results = {}
    passing = 0
    for n_channels in map(int, config["task"]["n_channels"]):
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        queries = task_queries(
            n_channels, config["task"]["demand_ratios"]
        )
        prototypes = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in frozen_n["prototypes"].items()
        }
        frame_sizes = []
        minimum_sizes = []
        compact_sizes = []
        separate_sizes = []
        reductions = []
        escape_count = 0
        resolved_bank_sites = 0
        site_rows = {}
        for site in validation_sites:
            power = aggregate_frequency_bins(
                loaded[metadata[site]["member"]], n_channels
            )
            resolution = resolve_context_and_actions(
                power=power,
                queries=queries,
                epsilon_db=float(config["task"]["epsilon_db"]),
                prototypes=prototypes,
                novelty_threshold=float(frozen_n["novelty_threshold"]),
                bank_actions=frozen_n["bank_actions"],
                gate=protocol["context_gate"],
                primary_k=int(config["task"]["primary_k"]),
            )
            states = resolution.pop("states")
            decision_name = str(resolution["decision"])
            row = {
                "resolved": bool(resolution["resolved"]),
                "decision": decision_name,
                "scene_count": len(states),
                "checkpoint_scene_count": 0,
            }
            if (
                not resolution["resolved"]
                or not decision_name.startswith("BANK:")
            ):
                site_rows[site] = row
                continue
            resolved_bank_sites += 1
            calibration = int(resolution["calibration_scenes"])
            codebook = codebook_from_selected_actions(
                states[:calibration],
                resolution["selected_actions"],
            )
            bank_source = decision_name.removeprefix("BANK:")
            bank_id = int(frozen_n["bank_ids"][bank_source])
            catalog = build_preinstalled_catalog(
                [(bank_id, codebook)], catalog_epoch=1
            )
            local_sizes = []
            local_escape = 0
            for epoch, state in enumerate(states[calibration:], start=1):
                checkpoint, checkpoint_decision = (
                    encode_self_contained_checkpoint(
                        state,
                        catalog,
                        node_id=1,
                        bank_id=bank_id,
                        codebook_epoch=2,
                        update_epoch=epoch % 256,
                    )
                )
                decoded = decode_self_contained_checkpoint(
                    checkpoint, catalog
                )
                compact, compact_decision = encode_compact_update(
                    state,
                    decoded.session,
                    node_id=1,
                    update_epoch=epoch % 256,
                )
                if (
                    decoded.decoder_actions
                    != checkpoint_decision.decoder_actions
                    or decoded.decoder_actions
                    != compact_decision.decoder_actions
                ):
                    raise ValueError("checkpoint action mismatch")
                checkpoint_size = int(checkpoint.size)
                compact_size = int(compact.size)
                separate_size = activation_bits + compact_size
                frame_sizes.append(checkpoint_size)
                minimum_sizes.append(
                    minimum_checkpoint_bits(catalog, bank_id)
                )
                compact_sizes.append(compact_size)
                separate_sizes.append(separate_size)
                reductions.append(
                    100.0
                    * (separate_size - checkpoint_size)
                    / separate_size
                )
                local_sizes.append(checkpoint_size)
                local_escape += int(checkpoint_decision.uses_fallback)
            escape_count += local_escape
            row.update(
                {
                    "checkpoint_scene_count": len(local_sizes),
                    "minimum_frame_bits": int(min(local_sizes)),
                    "maximum_frame_bits": int(max(local_sizes)),
                    "mean_frame_bits": float(np.mean(local_sizes)),
                    "escape_count": local_escape,
                }
            )
            site_rows[site] = row
        nonescape = [
            size
            for size, minimum in zip(frame_sizes, minimum_sizes)
            if size == minimum
        ]
        semantic_wp = frozen_n["workpoints"]["semantic"]
        attempts = int(semantic_wp["task_update_open_loop_attempts"])
        interval_cost = {}
        mean_frame = float(np.mean(frame_sizes))
        for interval in map(int, config["candidate_intervals_scenes"]):
            interval_cost[str(interval)] = {
                "checkpoint_plus_ack_bits_per_scene": float(
                    (attempts * mean_frame + ack_bits) / interval
                ),
                "ideal_request_response_heartbeat_bits_per_scene": float(
                    64.0 / interval
                ),
                "separate_activation_update_ack_bits_per_scene": float(
                    (
                        attempts * float(np.mean(separate_sizes))
                        + ack_bits
                    )
                    / interval
                ),
            }
        gate = config["feasibility_gate"]
        checks = {
            "all_validation_bank_frames_roundtrip": True,
            "maximum_non_escape_bits": int(max(nonescape))
            <= int(gate["maximum_non_escape_checkpoint_bits"]),
            "mean_reduction_vs_separate": float(np.mean(reductions))
            >= float(
                gate[
                    "minimum_reduction_vs_separate_activation_plus_update_percent"
                ]
            ),
            "escape_actions_preserved": True,
        }
        passed = all(checks.values())
        passing += int(passed)
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "resolved_bank_site_count": resolved_bank_sites,
            "site_results": site_rows,
            "frame_statistics": {
                "sample_count": len(frame_sizes),
                "non_escape_count": len(nonescape),
                "escape_count": escape_count,
                "escape_rate": float(escape_count / len(frame_sizes)),
                "minimum_non_escape_bits": int(min(nonescape)),
                "maximum_non_escape_bits": int(max(nonescape)),
                "mean_all_frame_bits": mean_frame,
                "maximum_all_frame_bits": int(max(frame_sizes)),
                "mean_compact_update_bits": float(
                    np.mean(compact_sizes)
                ),
                "mean_separate_activation_plus_update_bits": float(
                    np.mean(separate_sizes)
                ),
                "mean_reduction_vs_separate_percent": float(
                    np.mean(reductions)
                ),
            },
            "nominal_periodic_cost": interval_cost,
            "checks": checks,
            "feasibility_gate_passed": passed,
        }
        print(f"S7.4A-0 N={n_channels} pass={passed}", flush=True)
    required = int(
        config["feasibility_gate"]["minimum_passing_n_count"]
    )
    passed = passing >= required
    next_action = (
        config["feasibility_gate"]["if_gate_passes"]
        if passed
        else config["feasibility_gate"]["if_gate_fails"]
    )
    result = {
        "version": "1.0",
        "status": "stage7_s7_4a0_checkpoint_feasibility_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            key: sha256_file(path)
            for key, path in paths.items()
            if path.is_file()
        },
        "loaded_sites": {
            "train": [],
            "validation": validation_sites,
            "internal_development_test": [],
            "reserve": [],
        },
        "n_results": n_results,
        "decision": {
            "passing_n_count": passing,
            "required_n_count": required,
            "feasibility_gate_passed": passed,
            "next_action": next_action,
        },
        "governance_checks": {
            "internal_development_test_not_loaded": True,
            "reserve_remained_unread": True,
            "signal_values_used_only_for_frame_size_and_roundtrip": True,
            "full_protocol_effect_claim_forbidden": True,
            "external_final_claim_forbidden": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
