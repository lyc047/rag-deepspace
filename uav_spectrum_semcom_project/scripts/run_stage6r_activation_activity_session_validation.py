#!/usr/bin/env python
"""Run full activity/session validation for implemented Stage-6R activation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stage6r_activation_protocol_dry_run import (  # noqa: E402
    _compact_bits,
    _mean_ci,
    _metric_arrays,
    _paired_savings,
)
from run_stage6r_activity_session_diagnostic import _chunks  # noqa: E402
from run_stage6r_few_shot_adaptation import _load_combined, _queries  # noqa: E402
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
)
from spectrum_semcom.stage6_bit_accounting import (  # noqa: E402
    validate_breakdown_identity,
)
from spectrum_semcom.stage6_context_codec import (  # noqa: E402
    decode_context_install,
    encode_context_install,
)
from spectrum_semcom.stage6_context_heartbeat import (  # noqa: E402
    encode_context_probe_request,
    encode_context_probe_response,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_task_codebook import build_task_state  # noqa: E402
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6r_activation_reliability import (  # noqa: E402
    simulate_activation_semantic_trajectory,
)
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)


DEFAULT_CONFIG = (
    PROJECT_DIR
    / "configs/stage6r_activation_activity_session_validation_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/activation_activity_session_validation_v1/result.json"
)


def _summary(arrays: dict, *, seed: int, config: dict) -> dict:
    return {
        name: _mean_ci(
            values,
            seed=seed + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
        )
        for position, (name, values) in enumerate(arrays.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite activation validation")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        name: PROJECT_DIR / config["inputs"][name]
        for name in config["inputs"]
    }
    adaptation = json.loads(
        paths["adaptation_config"].read_text(encoding="utf-8")
    )
    temporal = json.loads(
        paths["temporal_gate_result"].read_text(encoding="utf-8")
    )
    scan_config = json.loads(
        paths["matched_scan_config"].read_text(encoding="utf-8")
    )
    scan = json.loads(
        paths["matched_scan_result"].read_text(encoding="utf-8")
    )
    full_diagnostic = json.loads(
        paths["full_manifest_activity_diagnostic"].read_text(
            encoding="utf-8"
        )
    )
    access = json.loads(
        paths["external_final_access_state"].read_text(encoding="utf-8")
    )
    target = float(config["frozen_selection"]["matched_clean_target"])
    primary_k = int(config["frozen_selection"]["primary_k"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    if trajectories != int(scan_config["monte_carlo"]["trajectories"]):
        raise ValueError("trajectory count must match the frozen scan")
    base_seed = int(scan_config["monte_carlo"]["seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    condition = scan_config["conditions"]["primary_fault"]
    codebook_epoch = int(config["frozen_selection"]["codebook_epoch"])
    catalog_epoch = int(config["frozen_selection"]["catalog_epoch"])
    ack_bits = int(cumulative_ack_payload_bits())
    heartbeat_request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=codebook_epoch,
            expected_update_epoch=0,
        ).size
    )
    heartbeat_response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=codebook_epoch,
            current_update_epoch=0,
        ).size
    )
    n_results = {}
    started = time.perf_counter()

    for n_position, n_channels in enumerate(
        map(int, scan_config["task"]["n_channels"])
    ):
        powers, campaigns, timestamps = _load_combined(
            adaptation, n_channels
        )
        queries = _queries(adaptation, n_channels)
        states = [
            build_task_state(
                row,
                queries,
                epsilon_db=float(scan_config["task"]["epsilon_db"]),
            )
            for row in powers
        ]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(
                scan_config["task"]["exact_action_header_bits"]
            ),
        )
        scan_n = scan["conditions"]["primary_fault"][str(n_channels)]
        matched = next(
            row
            for row in scan_n["matched_clean"]
            if row["common_eligible"]
            and abs(float(row["target_clean_rate"]) - target) < 1e-12
        )
        semantic_wp = next(
            row
            for row in scan_n["semantic_workpoints"]
            if row["workpoint_id"] == matched["semantic_workpoint_id"]
        )
        exact_wp = next(
            row
            for row in scan_n["exact_workpoints"]
            if row["workpoint_id"] == matched["exact_workpoint_id"]
        )
        temporal_rows = {
            row["holdout_campaign"]: row
            for row in temporal["n_results"][str(n_channels)]["rows"]
            if int(row["k"]) == primary_k
        }
        prepared = {}
        for campaign_position, campaign in enumerate(
            adaptation["splitting"]["campaign_ids"]
        ):
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            row = temporal_rows[campaign]
            calibration_count = int(row["calibration_scenes"])
            codebook = codebook_from_selected_actions(
                [
                    states[int(index)]
                    for index in indices[:calibration_count]
                ],
                row["selected_actions"],
            )
            sender = install_codebook(codebook, epoch=codebook_epoch)
            full_packet = encode_context_install(sender, node_id=1)
            receiver = decode_context_install(full_packet).session
            is_bank = str(row["decision"]).startswith("BANK:")
            if is_bank:
                source = str(row["decision"]).removeprefix("BANK:")
                bank_id = int(config["catalog_bank_ids"][source])
                catalog = build_preinstalled_catalog(
                    [(bank_id, codebook)],
                    catalog_epoch=catalog_epoch,
                )
            else:
                bank_id = None
                catalog = None
            prepared[campaign] = {
                "indices": indices,
                "calibration_count": calibration_count,
                "decision": row["decision"],
                "sender": sender,
                "receiver": receiver,
                "full_packet": full_packet,
                "catalog": catalog,
                "bank_id": bank_id,
                "compact_bits": _compact_bits(states, indices, sender),
                "random": np.random.default_rng(
                    base_seed
                    + n_position * 1_000_000
                    + campaign_position * 10_000
                ).random(
                    (trajectories, indices.size, RANDOM_STREAM_COUNT)
                ),
            }

        length_results = []
        for length_position, target_length in enumerate(
            config["session_length_scenes"]
        ):
            activity_internal = {}
            activity_output = {}
            for campaign_position, (campaign, value) in enumerate(
                prepared.items()
            ):
                chunks = _chunks(
                    value["indices"],
                    target_length=target_length,
                    calibration_count=value["calibration_count"],
                )
                activation_runs = []
                exact_runs = []
                activation_attempts = []
                fallbacks = []
                for trajectory in range(trajectories):
                    activation_parts = []
                    exact_parts = []
                    attempt_total = 0
                    fallback_total = 0
                    for chunk, random_slice in chunks:
                        calibration_count = value["calibration_count"]
                        calibration_indices = chunk[:calibration_count]
                        future_indices = chunk[calibration_count:]
                        random = value["random"][
                            trajectory, random_slice, :
                        ]
                        calibration = simulate_exact_trajectory(
                            states,
                            timestamps,
                            calibration_indices,
                            random_values=random[:calibration_count],
                            packet_loss_probability=float(
                                condition["task_loss_probability"]
                            ),
                            receiver_reset_probability=float(
                                condition[
                                    "receiver_context_reset_probability"
                                ]
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp[
                                    "calibration_open_loop_attempts"
                                ]
                            ),
                            packet_bits=exact_packet_bits,
                            epsilon_db=float(
                                scan_config["task"]["epsilon_db"]
                            ),
                            outage_penalty_db=float(
                                scan_config["task"][
                                    "outage_penalty_db"
                                ]
                            ),
                        )
                        suffix = simulate_activation_semantic_trajectory(
                            states,
                            timestamps,
                            future_indices,
                            sender_session=value["sender"],
                            receiver_session=value["receiver"],
                            full_install_packet=value["full_packet"],
                            sender_catalog=value["catalog"],
                            receiver_catalog=value["catalog"],
                            bank_id=value["bank_id"],
                            catalog_node_id=1,
                            maximum_activation_attempts=int(
                                config["frozen_selection"][
                                    "maximum_activation_attempts"
                                ]
                            ),
                            condition=condition,
                            random_values=random[calibration_count:],
                            epsilon_db=float(
                                scan_config["task"]["epsilon_db"]
                            ),
                            max_age_minutes=float(
                                semantic_wp[
                                    "maximum_state_age_minutes"
                                ]
                            ),
                            ack_frame_bits=ack_bits,
                            outage_penalty_db=float(
                                scan_config["task"][
                                    "outage_penalty_db"
                                ]
                            ),
                            heartbeat_interval_scenes=semantic_wp[
                                "heartbeat_silence_scenes"
                            ],
                            heartbeat_request_frame_bits=(
                                heartbeat_request_bits
                            ),
                            heartbeat_response_frame_bits=(
                                heartbeat_response_bits
                            ),
                            task_open_loop_attempts=int(
                                semantic_wp[
                                    "task_update_open_loop_attempts"
                                ]
                            ),
                            compact_codeword_frame_bits=value[
                                "compact_bits"
                            ],
                        )
                        activation_parts.extend(
                            [calibration, suffix.matched]
                        )
                        attempt_total += suffix.activation_attempt_count
                        fallback_total += suffix.full_install_fallback_count
                        exact_parts.append(
                            simulate_exact_trajectory(
                                states,
                                timestamps,
                                chunk,
                                random_values=random,
                                packet_loss_probability=float(
                                    condition["task_loss_probability"]
                                ),
                                receiver_reset_probability=float(
                                    condition[
                                        "receiver_context_reset_probability"
                                    ]
                                ),
                                task_open_loop_attempts=int(
                                    exact_wp[
                                        "task_packet_open_loop_attempts"
                                    ]
                                ),
                                packet_bits=exact_packet_bits,
                                epsilon_db=float(
                                    scan_config["task"]["epsilon_db"]
                                ),
                                outage_penalty_db=float(
                                    scan_config["task"][
                                        "outage_penalty_db"
                                    ]
                                ),
                            )
                        )
                    activation_runs.append(
                        combine_matched_trajectory_results(
                            activation_parts
                        )
                    )
                    exact_runs.append(
                        combine_matched_trajectory_results(exact_parts)
                    )
                    activation_attempts.append(attempt_total)
                    fallbacks.append(fallback_total)
                activation_arrays = _metric_arrays(activation_runs)
                exact_arrays = _metric_arrays(exact_runs)
                activity_internal[campaign] = {
                    "activation_runs": activation_runs,
                    "exact_runs": exact_runs,
                    "activation_arrays": activation_arrays,
                    "exact_arrays": exact_arrays,
                }
                activity_output[campaign] = {
                    "decision": value["decision"],
                    "evaluated_scene_count": len(
                        activation_runs[0].clean
                    ),
                    "session_count": len(chunks),
                    "activation": _summary(
                        activation_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                        ),
                        config=config,
                    ),
                    "exact": _summary(
                        exact_arrays,
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                            + 31
                        ),
                        config=config,
                    ),
                    "paired_savings_percentage": _mean_ci(
                        _paired_savings(
                            activation_arrays, exact_arrays
                        ),
                        seed=(
                            bootstrap_seed
                            + n_position * 100_000
                            + length_position * 10_000
                            + campaign_position * 100
                            + 67
                        ),
                        replicates=int(
                            config["monte_carlo"][
                                "bootstrap_replicates"
                            ]
                        ),
                        confidence=float(
                            config["monte_carlo"]["confidence_level"]
                        ),
                    ),
                    "mean_activation_attempt_count": float(
                        np.mean(activation_attempts)
                    ),
                    "mean_full_fallback_count": float(
                        np.mean(fallbacks)
                    ),
                }

            aggregate_activation = []
            aggregate_exact = []
            macro_activation_bits = []
            macro_exact_bits = []
            for trajectory in range(trajectories):
                aggregate_activation.append(
                    combine_matched_trajectory_results(
                        [
                            value["activation_runs"][trajectory]
                            for value in activity_internal.values()
                        ]
                    )
                )
                aggregate_exact.append(
                    combine_matched_trajectory_results(
                        [
                            value["exact_runs"][trajectory]
                            for value in activity_internal.values()
                        ]
                    )
                )
                macro_activation_bits.append(
                    np.mean(
                        [
                            value["activation_arrays"][
                                "total_bits_per_scene"
                            ][trajectory]
                            for value in activity_internal.values()
                        ]
                    )
                )
                macro_exact_bits.append(
                    np.mean(
                        [
                            value["exact_arrays"][
                                "total_bits_per_scene"
                            ][trajectory]
                            for value in activity_internal.values()
                        ]
                    )
                )
            activation_arrays = _metric_arrays(aggregate_activation)
            exact_arrays = _metric_arrays(aggregate_exact)
            macro_activation_bits = np.asarray(macro_activation_bits)
            macro_exact_bits = np.asarray(macro_exact_bits)
            macro_savings = (
                100.0
                * (macro_exact_bits - macro_activation_bits)
                / macro_exact_bits
            )
            reference = next(
                row
                for row in full_diagnostic["n_results"][str(n_channels)][
                    "session_length_results"
                ]
                if row["session_length_scenes"] == target_length
            )
            length_results.append(
                {
                    "session_length_scenes": target_length,
                    "activity_results": activity_output,
                    "scene_weighted": {
                        "activation": _summary(
                            activation_arrays,
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + length_position * 10_000
                                + 7001
                            ),
                            config=config,
                        ),
                        "exact": _summary(
                            exact_arrays,
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + length_position * 10_000
                                + 7002
                            ),
                            config=config,
                        ),
                        "paired_savings_percentage": _mean_ci(
                            _paired_savings(
                                activation_arrays, exact_arrays
                            ),
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + length_position * 10_000
                                + 7003
                            ),
                            replicates=int(
                                config["monte_carlo"][
                                    "bootstrap_replicates"
                                ]
                            ),
                            confidence=float(
                                config["monte_carlo"][
                                    "confidence_level"
                                ]
                            ),
                        ),
                    },
                    "equal_activity_macro_paired_savings_percentage": (
                        _mean_ci(
                            macro_savings,
                            seed=(
                                bootstrap_seed
                                + n_position * 100_000
                                + length_position * 10_000
                                + 8001
                            ),
                            replicates=int(
                                config["monte_carlo"][
                                    "bootstrap_replicates"
                                ]
                            ),
                            confidence=float(
                                config["monte_carlo"][
                                    "confidence_level"
                                ]
                            ),
                        )
                    ),
                    "full_manifest_reference": {
                        "clean_rate": reference["scene_weighted"][
                            "semantic"
                        ]["clean_rate"],
                        "paired_savings_percentage": reference[
                            "scene_weighted"
                        ]["paired_savings_percentage"],
                        "macro_paired_savings_percentage": reference[
                            "equal_activity_macro"
                        ]["paired_savings_percentage"],
                    },
                }
            )

        whole = next(
            row
            for row in length_results
            if row["session_length_scenes"] is None
        )
        shortest = [
            row["session_length_scenes"]
            for row in length_results
            if row["session_length_scenes"] is not None
            and row["scene_weighted"]["paired_savings_percentage"][
                "ci_lower"
            ]
            > 0.0
        ]
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "session_length_results": length_results,
            "checks": {
                "whole_scene_weighted_savings_lower_positive": (
                    whole["scene_weighted"][
                        "paired_savings_percentage"
                    ]["ci_lower"]
                    > 0.0
                ),
                "whole_macro_savings_lower_positive": (
                    whole[
                        "equal_activity_macro_paired_savings_percentage"
                    ]["ci_lower"]
                    > 0.0
                ),
                "whole_clean_loss_vs_full_manifest": float(
                    whole["full_manifest_reference"]["clean_rate"]["mean"]
                    - whole["scene_weighted"]["activation"][
                        "clean_rate"
                    ]["mean"]
                ),
                "shortest_positive_session_length": (
                    min(shortest) if shortest else None
                ),
                "all_wrong_codebook_decode_counts_zero": all(
                    row["scene_weighted"]["activation"][
                        "wrong_codebook_decode_count"
                    ]["mean"]
                    == 0.0
                    for row in length_results
                ),
            },
        }
        print(
            f"S6R activation activity/session N={n_channels} complete",
            flush=True,
        )

    macro_positive = sum(
        value["checks"]["whole_macro_savings_lower_positive"]
        for value in n_results.values()
    )
    checks = {
        "whole_scene_weighted_savings_lower_positive_all_n": all(
            value["checks"][
                "whole_scene_weighted_savings_lower_positive"
            ]
            for value in n_results.values()
        ),
        "whole_macro_savings_lower_positive_n_count": macro_positive,
        "whole_clean_loss_within_gate_all_n": all(
            value["checks"]["whole_clean_loss_vs_full_manifest"]
            <= float(
                config["gates"][
                    "whole_activity_clean_loss_vs_full_manifest_maximum"
                ]
            )
            for value in n_results.values()
        ),
        "all_wrong_codebook_decode_counts_zero": all(
            value["checks"]["all_wrong_codebook_decode_counts_zero"]
            for value in n_results.values()
        ),
        "primary_gate_passed": (
            all(
                value["checks"][
                    "whole_scene_weighted_savings_lower_positive"
                ]
                and value["checks"][
                    "all_wrong_codebook_decode_counts_zero"
                ]
                for value in n_results.values()
            )
            and macro_positive
            >= int(
                config["gates"][
                    "whole_activity_equal_activity_macro_savings_lower_positive_minimum_n_count"
                ]
            )
        ),
        "external_final_access_count_unchanged": (
            access.get("access_count") == 1
        ),
    }
    result = {
        "version": "1.0",
        "status": "stage6r_activation_activity_session_validation_complete",
        "verification_status": "ANALYZED",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(args.config),
        "input_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "external_access_status_before_run": access.get("status"),
        "external_access_count_before_run": access.get("access_count"),
        "n_results": n_results,
        "checks": checks,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
