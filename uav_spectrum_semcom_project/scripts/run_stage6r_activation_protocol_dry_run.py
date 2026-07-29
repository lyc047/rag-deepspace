#!/usr/bin/env python
"""Run the registered implemented-activation protocol dry-run."""

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
    encode_compact_update,
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
    simulate_calibrated_semantic_trajectory,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_activation_protocol_dry_run_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6r/activation_protocol_dry_run_v1/result.json"
)


def _mean_ci(values, *, seed: int, replicates: int, confidence: float) -> dict:
    rows = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    positions = rng.integers(
        0, rows.size, size=(int(replicates), rows.size)
    )
    boot = np.mean(rows[positions], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(rows)),
        "ci_lower": float(np.quantile(boot, alpha)),
        "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
    }


def _metric_arrays(results: list) -> dict[str, np.ndarray]:
    rows = []
    for result in results:
        validate_breakdown_identity(result.bit_breakdown)
        scenes = len(result.clean)
        rows.append(
            {
                "clean_rate": float(np.mean(result.clean)),
                "availability_rate": float(np.mean(result.available)),
                "total_bits_per_scene": float(
                    result.bit_breakdown.actual_total_application_bits
                    / scenes
                ),
                "context_bits_per_scene": float(
                    (
                        result.bit_breakdown.initial_install_bits
                        + result.bit_breakdown.recovery_install_bits
                    )
                    / scenes
                ),
                "wrong_codebook_decode_count": float(
                    result.wrong_codebook_decode_count
                ),
            }
        )
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _summaries(arrays: dict, *, seed: int, config: dict) -> dict:
    return {
        name: _mean_ci(
            values,
            seed=seed + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
        )
        for position, (name, values) in enumerate(arrays.items())
    }


def _paired_savings(semantic: dict, exact: dict) -> np.ndarray:
    return 100.0 * (
        exact["total_bits_per_scene"] - semantic["total_bits_per_scene"]
    ) / exact["total_bits_per_scene"]


def _compact_bits(states, indices, sender) -> int:
    for index in indices:
        packet, decision = encode_compact_update(
            states[int(index)],
            sender,
            node_id=1,
            update_epoch=1,
        )
        if not decision.uses_fallback:
            return int(packet.size)
    raise ValueError("selected codebook has no compact state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite activation dry-run")
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
    access = json.loads(
        paths["external_final_access_state"].read_text(encoding="utf-8")
    )
    target = float(config["frozen_selection"]["matched_clean_target"])
    primary_k = int(config["frozen_selection"]["primary_k"])
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(config["monte_carlo"]["random_seed"])
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
        activity_outputs = {}
        trajectory_full = [[] for _ in range(trajectories)]
        trajectory_activation = [[] for _ in range(trajectories)]
        trajectory_exact = [[] for _ in range(trajectories)]
        macro_arrays = {
            "full": [],
            "activation": [],
            "exact": [],
        }

        for campaign_position, campaign in enumerate(
            adaptation["splitting"]["campaign_ids"]
        ):
            indices = np.flatnonzero(campaigns == campaign)
            indices = indices[np.argsort(timestamps[indices], kind="stable")]
            row = temporal_rows[campaign]
            calibration_count = int(row["calibration_scenes"])
            calibration_indices = indices[:calibration_count]
            future_indices = indices[calibration_count:]
            codebook = codebook_from_selected_actions(
                [states[int(index)] for index in calibration_indices],
                row["selected_actions"],
            )
            sender = install_codebook(codebook, epoch=codebook_epoch)
            full_packet = encode_context_install(sender, node_id=1)
            receiver = decode_context_install(full_packet).session
            compact_bits = _compact_bits(states, indices, sender)
            is_bank = str(row["decision"]).startswith("BANK:")
            if is_bank:
                source = str(row["decision"]).removeprefix("BANK:")
                bank_id = int(config["catalog_bank_ids"][source])
                sender_catalog = build_preinstalled_catalog(
                    [(bank_id, codebook)],
                    catalog_epoch=catalog_epoch,
                )
                receiver_catalog = sender_catalog
            else:
                bank_id = None
                sender_catalog = None
                receiver_catalog = None
            random = np.random.default_rng(
                base_seed
                + n_position * 1_000_000
                + campaign_position * 10_000
            ).random(
                (trajectories, indices.size, RANDOM_STREAM_COUNT)
            )
            full_runs = []
            activation_runs = []
            exact_runs = []
            activation_counts = []
            full_fallback_counts = []
            for trajectory in range(trajectories):
                full = simulate_calibrated_semantic_trajectory(
                    states,
                    timestamps,
                    indices,
                    calibration_scene_count=calibration_count,
                    selected_actions=row["selected_actions"],
                    random_values=random[trajectory],
                    condition=condition,
                    exact_packet_bits=exact_packet_bits,
                    epsilon_db=float(scan_config["task"]["epsilon_db"]),
                    max_age_minutes=float(
                        semantic_wp["maximum_state_age_minutes"]
                    ),
                    ack_frame_bits=ack_bits,
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                    heartbeat_interval_scenes=semantic_wp[
                        "heartbeat_silence_scenes"
                    ],
                    heartbeat_request_frame_bits=heartbeat_request_bits,
                    heartbeat_response_frame_bits=heartbeat_response_bits,
                    calibration_open_loop_attempts=int(
                        semantic_wp["calibration_open_loop_attempts"]
                    ),
                    task_open_loop_attempts=int(
                        semantic_wp["task_update_open_loop_attempts"]
                    ),
                    codebook_epoch=codebook_epoch,
                ).combined
                calibration = simulate_exact_trajectory(
                    states,
                    timestamps,
                    calibration_indices,
                    random_values=random[trajectory, :calibration_count],
                    packet_loss_probability=float(
                        condition["task_loss_probability"]
                    ),
                    receiver_reset_probability=float(
                        condition["receiver_context_reset_probability"]
                    ),
                    task_open_loop_attempts=int(
                        semantic_wp["calibration_open_loop_attempts"]
                    ),
                    packet_bits=exact_packet_bits,
                    epsilon_db=float(scan_config["task"]["epsilon_db"]),
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                )
                activated_suffix = simulate_activation_semantic_trajectory(
                    states,
                    timestamps,
                    future_indices,
                    sender_session=sender,
                    receiver_session=receiver,
                    full_install_packet=full_packet,
                    sender_catalog=sender_catalog,
                    receiver_catalog=receiver_catalog,
                    bank_id=bank_id,
                    catalog_node_id=1,
                    maximum_activation_attempts=int(
                        config["frozen_selection"][
                            "maximum_activation_attempts"
                        ]
                    ),
                    condition=condition,
                    random_values=random[trajectory, calibration_count:],
                    epsilon_db=float(scan_config["task"]["epsilon_db"]),
                    max_age_minutes=float(
                        semantic_wp["maximum_state_age_minutes"]
                    ),
                    ack_frame_bits=ack_bits,
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                    heartbeat_interval_scenes=semantic_wp[
                        "heartbeat_silence_scenes"
                    ],
                    heartbeat_request_frame_bits=heartbeat_request_bits,
                    heartbeat_response_frame_bits=heartbeat_response_bits,
                    task_open_loop_attempts=int(
                        semantic_wp["task_update_open_loop_attempts"]
                    ),
                    compact_codeword_frame_bits=compact_bits,
                )
                activated = combine_matched_trajectory_results(
                    [calibration, activated_suffix.matched]
                )
                exact = simulate_exact_trajectory(
                    states,
                    timestamps,
                    indices,
                    random_values=random[trajectory],
                    packet_loss_probability=float(
                        condition["task_loss_probability"]
                    ),
                    receiver_reset_probability=float(
                        condition["receiver_context_reset_probability"]
                    ),
                    task_open_loop_attempts=int(
                        exact_wp["task_packet_open_loop_attempts"]
                    ),
                    packet_bits=exact_packet_bits,
                    epsilon_db=float(scan_config["task"]["epsilon_db"]),
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                )
                full_runs.append(full)
                activation_runs.append(activated)
                exact_runs.append(exact)
                activation_counts.append(
                    activated_suffix.activation_attempt_count
                )
                full_fallback_counts.append(
                    activated_suffix.full_install_fallback_count
                )
                trajectory_full[trajectory].append(full)
                trajectory_activation[trajectory].append(activated)
                trajectory_exact[trajectory].append(exact)
            full_arrays = _metric_arrays(full_runs)
            activation_arrays = _metric_arrays(activation_runs)
            exact_arrays = _metric_arrays(exact_runs)
            macro_arrays["full"].append(full_arrays)
            macro_arrays["activation"].append(activation_arrays)
            macro_arrays["exact"].append(exact_arrays)
            activity_outputs[campaign] = {
                "decision": row["decision"],
                "scene_count": int(indices.size),
                "full_manifest": _summaries(
                    full_arrays,
                    seed=base_seed + n_position * 100_000
                    + campaign_position * 100 + 1,
                    config=config,
                ),
                "activation": _summaries(
                    activation_arrays,
                    seed=base_seed + n_position * 100_000
                    + campaign_position * 100 + 2,
                    config=config,
                ),
                "exact": _summaries(
                    exact_arrays,
                    seed=base_seed + n_position * 100_000
                    + campaign_position * 100 + 3,
                    config=config,
                ),
                "activation_savings_vs_exact_percentage": _mean_ci(
                    _paired_savings(activation_arrays, exact_arrays),
                    seed=base_seed + n_position * 100_000
                    + campaign_position * 100 + 4,
                    replicates=int(
                        config["monte_carlo"]["bootstrap_replicates"]
                    ),
                    confidence=float(
                        config["monte_carlo"]["confidence_level"]
                    ),
                ),
                "mean_activation_attempt_count": float(
                    np.mean(activation_counts)
                ),
                "mean_full_fallback_count": float(
                    np.mean(full_fallback_counts)
                ),
            }

        aggregate_full = [
            combine_matched_trajectory_results(rows)
            for rows in trajectory_full
        ]
        aggregate_activation = [
            combine_matched_trajectory_results(rows)
            for rows in trajectory_activation
        ]
        aggregate_exact = [
            combine_matched_trajectory_results(rows)
            for rows in trajectory_exact
        ]
        full_arrays = _metric_arrays(aggregate_full)
        activation_arrays = _metric_arrays(aggregate_activation)
        exact_arrays = _metric_arrays(aggregate_exact)
        macro_activation_bits = np.mean(
            [
                arrays["total_bits_per_scene"]
                for arrays in macro_arrays["activation"]
            ],
            axis=0,
        )
        macro_exact_bits = np.mean(
            [
                arrays["total_bits_per_scene"]
                for arrays in macro_arrays["exact"]
            ],
            axis=0,
        )
        macro_savings = (
            100.0
            * (macro_exact_bits - macro_activation_bits)
            / macro_exact_bits
        )
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "activity_results": activity_outputs,
            "scene_weighted": {
                "full_manifest": _summaries(
                    full_arrays,
                    seed=base_seed + n_position * 100_000 + 7001,
                    config=config,
                ),
                "activation": _summaries(
                    activation_arrays,
                    seed=base_seed + n_position * 100_000 + 7002,
                    config=config,
                ),
                "exact": _summaries(
                    exact_arrays,
                    seed=base_seed + n_position * 100_000 + 7003,
                    config=config,
                ),
                "activation_savings_vs_exact_percentage": _mean_ci(
                    _paired_savings(activation_arrays, exact_arrays),
                    seed=base_seed + n_position * 100_000 + 7004,
                    replicates=int(
                        config["monte_carlo"]["bootstrap_replicates"]
                    ),
                    confidence=float(
                        config["monte_carlo"]["confidence_level"]
                    ),
                ),
            },
            "equal_activity_macro_activation_savings_vs_exact_percentage": (
                _mean_ci(
                    macro_savings,
                    seed=base_seed + n_position * 100_000 + 8001,
                    replicates=int(
                        config["monte_carlo"]["bootstrap_replicates"]
                    ),
                    confidence=float(
                        config["monte_carlo"]["confidence_level"]
                    ),
                )
            ),
            "checks": {
                "activation_bits_below_full_manifest": (
                    float(
                        np.mean(activation_arrays["total_bits_per_scene"])
                    )
                    < float(np.mean(full_arrays["total_bits_per_scene"]))
                ),
                "activation_clean_loss_vs_full_manifest": float(
                    np.mean(full_arrays["clean_rate"])
                    - np.mean(activation_arrays["clean_rate"])
                ),
                "all_wrong_codebook_decode_counts_zero": bool(
                    np.max(
                        activation_arrays["wrong_codebook_decode_count"]
                    )
                    == 0.0
                ),
                "ood_activation_attempt_count_zero": all(
                    output["mean_activation_attempt_count"] == 0.0
                    for output in activity_outputs.values()
                    if output["decision"] == "OOD"
                ),
            },
        }
        print(f"S6R activation dry-run N={n_channels} complete", flush=True)

    maximum_clean_loss = float(
        config["gates"]["activation_clean_loss_vs_full_manifest_maximum"]
    )
    macro_positive_count = sum(
        value[
            "equal_activity_macro_activation_savings_vs_exact_percentage"
        ]["mean"]
        > 0.0
        for value in n_results.values()
    )
    weighted_lower_positive_count = sum(
        value["scene_weighted"][
            "activation_savings_vs_exact_percentage"
        ]["ci_lower"]
        > 0.0
        for value in n_results.values()
    )
    checks = {
        "all_bit_identities_valid": True,
        "activation_total_bits_below_full_manifest_all_n": all(
            value["checks"]["activation_bits_below_full_manifest"]
            for value in n_results.values()
        ),
        "activation_clean_loss_within_gate_all_n": all(
            value["checks"]["activation_clean_loss_vs_full_manifest"]
            <= maximum_clean_loss
            for value in n_results.values()
        ),
        "all_wrong_codebook_decode_counts_zero": all(
            value["checks"][
                "all_wrong_codebook_decode_counts_zero"
            ]
            for value in n_results.values()
        ),
        "all_ood_activation_attempt_counts_zero": all(
            value["checks"]["ood_activation_attempt_count_zero"]
            for value in n_results.values()
        ),
        "macro_positive_n_count": macro_positive_count,
        "weighted_savings_lower_positive_n_count": (
            weighted_lower_positive_count
        ),
        "full_revalidation_entry_passed": (
            macro_positive_count
            >= int(
                config["gates"][
                    "equal_activity_macro_savings_positive_minimum_n_count"
                ]
            )
            and weighted_lower_positive_count
            >= int(
                config["gates"][
                    "scene_weighted_savings_lower_positive_minimum_n_count"
                ]
            )
            and all(
                value["checks"]["activation_bits_below_full_manifest"]
                and value["checks"][
                    "all_wrong_codebook_decode_counts_zero"
                ]
                and value["checks"]["ood_activation_attempt_count_zero"]
                and value["checks"][
                    "activation_clean_loss_vs_full_manifest"
                ]
                <= maximum_clean_loss
                for value in n_results.values()
            )
        ),
        "external_final_access_count_unchanged": (
            access.get("access_count") == 1
        ),
    }
    result = {
        "version": "1.0",
        "status": "stage6r_activation_protocol_dry_run_complete",
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
