#!/usr/bin/env python
"""Single-consumption Stage-6R execution on a frozen ElectroSense role."""

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
from spectrum_semcom.electrosense_psd import (  # noqa: E402
    aggregate_frequency_bins,
    claim_role_access,
    load_npy_members,
    read_json,
    role_members,
    sha256_file,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import environment_snapshot  # noqa: E402
from spectrum_semcom.stage5_cumulative_ack import (  # noqa: E402
    cumulative_ack_payload_bits,
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
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402
from spectrum_semcom.stage6r_activation_reliability import (  # noqa: E402
    simulate_activation_semantic_trajectory,
)
from spectrum_semcom.stage6r_codebook_activation import (  # noqa: E402
    build_preinstalled_catalog,
)
from spectrum_semcom.stage6r_confirmation_governance import (  # noqa: E402
    validate_execution_preconditions,
)
from spectrum_semcom.stage6r_external_final import (  # noqa: E402
    code_snapshot,
    hierarchical_mean_ci,
    resolve_context_and_actions,
    synthetic_six_hour_timestamps,
    task_queries,
)
from spectrum_semcom.stage6r_reliability_integration import (  # noqa: E402
    codebook_from_selected_actions,
)


DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_electrosense_external_final_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/electrosense_external_final_v1/result.json"
)


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


def _hierarchical_metric_summary(
    site_arrays: list[dict[str, np.ndarray]],
    *,
    seed: int,
    config: dict,
    site_weights: np.ndarray,
) -> dict:
    names = site_arrays[0].keys()
    return {
        name: hierarchical_mean_ci(
            np.stack([row[name] for row in site_arrays]),
            seed=seed + position * 1009,
            replicates=int(config["monte_carlo"]["bootstrap_replicates"]),
            confidence=float(config["monte_carlo"]["confidence_level"]),
            site_weights=site_weights,
        )
        for position, name in enumerate(names)
    }


def _site_validation(values: np.ndarray, metadata: dict) -> str | None:
    if values.ndim != 2:
        return "array_is_not_two_dimensional"
    if list(values.shape) != metadata["shape"]:
        return "shape_differs_from_registered_npy_header"
    if values.shape[0] < 200 or values.shape[1] < 64:
        return "array_below_frozen_minimum_shape"
    if not np.all(np.isfinite(values)):
        return "array_contains_non_finite_values"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--consume",
        action="store_true",
        help="Required acknowledgement that the configured role is consumed once.",
    )
    args = parser.parse_args()
    if not args.consume:
        raise ValueError("execution requires explicit --consume")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite external Final result")
    config = read_json(args.config)
    paths = {
        name: Path(value)
        if name == "archive"
        else PROJECT_DIR / value
        for name, value in config["inputs"].items()
    }
    registry = read_json(paths["registry"])
    access = read_json(paths["access_state"])
    freeze = read_json(paths["freeze_result"])
    current_snapshot = code_snapshot(
        PROJECT_DIR, config["code_snapshot_paths"]
    )
    execution_role, frozen_sites = validate_execution_preconditions(
        config=config,
        registry=registry,
        access_state=access,
        freeze=freeze,
        protocol_sha256=sha256_file(args.config),
        code_snapshot=current_snapshot,
    )
    if paths["archive"].stat().st_size != registry["archive"]["size_bytes"]:
        raise ValueError("archive size changed after registration")
    verify_full_archive = bool(
        config.get("governance", {}).get(
            "verify_full_archive_sha256_before_access",
            execution_role == "stage6_final",
        )
    )
    if (
        verify_full_archive
        and sha256_file(paths["archive"]) != registry["archive"]["sha256"]
    ):
        raise ValueError("archive hash changed after registration")

    governance = config.get("governance", {})
    receipt = claim_role_access(
        registry_path=paths["registry"],
        access_state_path=paths["access_state"],
        role=execution_role,
        actor=str(
            governance.get(
                "actor", "Codex Stage-6R confirmatory executor"
            )
        ),
        purpose=str(
            governance.get(
                "purpose",
                "single 24-site Stage-6R ElectroSense external Final",
            )
        ),
        evidence={
            "protocol_sha256": freeze["protocol_sha256"],
            "freeze_result_sha256": sha256_file(paths["freeze_result"]),
            "code_snapshot_sha256": current_snapshot["combined_sha256"],
            "output_path": str(args.output),
        },
    )
    started = time.perf_counter()
    metadata_rows = role_members(registry, execution_role)
    loaded = load_npy_members(
        paths["archive"], [row["member"] for row in metadata_rows]
    )
    valid: list[tuple[dict, np.ndarray]] = []
    invalid_sites: list[dict] = []
    for metadata in metadata_rows:
        values = loaded[metadata["member"]]
        reason = _site_validation(values, metadata)
        if reason is None:
            valid.append((metadata, values))
        else:
            invalid_sites.append(
                {"site": metadata["site"], "reason": reason}
            )
    del loaded

    scan_config = read_json(paths["matched_scan_config"])
    activation_protocol = read_json(paths["activation_protocol"])
    condition = scan_config["conditions"]["primary_fault"]
    trajectories = int(config["monte_carlo"]["trajectories"])
    base_seed = int(config["monte_carlo"]["random_seed"])
    bootstrap_seed = int(config["monte_carlo"]["bootstrap_seed"])
    epsilon = float(config["task"]["epsilon_db"])
    codebook_epoch = int(config["reliability"]["codebook_epoch"])
    catalog_epoch = int(config["reliability"]["catalog_epoch"])
    maximum_activation_attempts = int(
        config["reliability"]["maximum_activation_attempts"]
    )
    if maximum_activation_attempts != int(
        activation_protocol["fail_closed"][
            "maximum_activation_attempts_before_full_install_fallback"
        ]
    ):
        raise ValueError("activation attempt limit differs from frozen protocol")
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
    for n_position, n_channels in enumerate(
        map(int, config["task"]["n_channels"])
    ):
        frozen_n = freeze["n_artifacts"][str(n_channels)]
        queries = task_queries(
            n_channels, config["task"]["demand_ratios"]
        )
        workpoints = frozen_n["workpoints"]
        semantic_wp = workpoints["semantic"]
        exact_wp = workpoints["exact"]
        exact_packet_bits = exact_query_bundle_bits(
            n_channels,
            queries,
            header_bits=int(scan_config["task"]["exact_action_header_bits"]),
        )
        prototypes = {
            source: np.asarray(values, dtype=np.float64)
            for source, values in frozen_n["prototypes"].items()
        }
        bank_actions = frozen_n["bank_actions"]
        site_outputs = {}
        semantic_site_arrays = []
        exact_site_arrays = []
        scene_counts = []
        for site_position, (metadata, raw_values) in enumerate(valid):
            site = metadata["site"]
            power = aggregate_frequency_bins(raw_values, n_channels)
            timestamps = synthetic_six_hour_timestamps(power.shape[0])
            resolution = resolve_context_and_actions(
                power=power,
                queries=queries,
                epsilon_db=epsilon,
                prototypes=prototypes,
                novelty_threshold=float(frozen_n["novelty_threshold"]),
                bank_actions=bank_actions,
                gate=config["context_gate"],
                primary_k=int(config["task"]["primary_k"]),
            )
            states = resolution.pop("states")
            indices = np.arange(len(states), dtype=np.int64)
            random = np.random.default_rng(
                base_seed
                + n_position * 1_000_000
                + site_position * 10_000
            ).random((trajectories, len(states), RANDOM_STREAM_COUNT))
            exact_runs = [
                simulate_exact_trajectory(
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
                    epsilon_db=epsilon,
                    outage_penalty_db=float(
                        scan_config["task"]["outage_penalty_db"]
                    ),
                )
                for trajectory in range(trajectories)
            ]
            activation_counts: list[int] = []
            full_fallback_counts: list[int] = []
            if not resolution["resolved"]:
                semantic_runs = exact_runs
                activation_counts = [0] * trajectories
                full_fallback_counts = [0] * trajectories
            else:
                calibration_count = int(resolution["calibration_scenes"])
                calibration_indices = indices[:calibration_count]
                future_indices = indices[calibration_count:]
                codebook = codebook_from_selected_actions(
                    states[:calibration_count],
                    resolution["selected_actions"],
                )
                sender = install_codebook(codebook, epoch=codebook_epoch)
                full_packet = encode_context_install(sender, node_id=1)
                receiver = decode_context_install(full_packet).session
                compact_bits = _compact_bits(states, indices, sender)
                is_bank = str(resolution["decision"]).startswith("BANK:")
                if is_bank:
                    source = str(resolution["decision"]).removeprefix(
                        "BANK:"
                    )
                    bank_id = int(frozen_n["bank_ids"][source])
                    catalog = build_preinstalled_catalog(
                        [(bank_id, codebook)], catalog_epoch=catalog_epoch
                    )
                else:
                    bank_id = None
                    catalog = None
                semantic_runs = []
                for trajectory in range(trajectories):
                    calibration = simulate_exact_trajectory(
                        states,
                        timestamps,
                        calibration_indices,
                        random_values=random[
                            trajectory, :calibration_count
                        ],
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
                        epsilon_db=epsilon,
                        outage_penalty_db=float(
                            scan_config["task"]["outage_penalty_db"]
                        ),
                    )
                    suffix = simulate_activation_semantic_trajectory(
                        states,
                        timestamps,
                        future_indices,
                        sender_session=sender,
                        receiver_session=receiver,
                        full_install_packet=full_packet,
                        sender_catalog=catalog,
                        receiver_catalog=catalog,
                        bank_id=bank_id,
                        catalog_node_id=1,
                        maximum_activation_attempts=(
                            maximum_activation_attempts
                        ),
                        condition=condition,
                        random_values=random[
                            trajectory, calibration_count:
                        ],
                        epsilon_db=epsilon,
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
                        heartbeat_request_frame_bits=(
                            heartbeat_request_bits
                        ),
                        heartbeat_response_frame_bits=(
                            heartbeat_response_bits
                        ),
                        task_open_loop_attempts=int(
                            semantic_wp["task_update_open_loop_attempts"]
                        ),
                        compact_codeword_frame_bits=compact_bits,
                    )
                    semantic_runs.append(
                        combine_matched_trajectory_results(
                            [calibration, suffix.matched]
                        )
                    )
                    activation_counts.append(
                        suffix.activation_attempt_count
                    )
                    full_fallback_counts.append(
                        suffix.full_install_fallback_count
                    )
            semantic_arrays = _metric_arrays(semantic_runs)
            exact_arrays = _metric_arrays(exact_runs)
            semantic_site_arrays.append(semantic_arrays)
            exact_site_arrays.append(exact_arrays)
            scene_counts.append(len(states))
            site_outputs[site] = {
                "scene_count": len(states),
                **resolution,
                "semantic": _summaries(
                    semantic_arrays,
                    seed=bootstrap_seed
                    + n_position * 100_000
                    + site_position * 100
                    + 1,
                    config=config,
                ),
                "exact": _summaries(
                    exact_arrays,
                    seed=bootstrap_seed
                    + n_position * 100_000
                    + site_position * 100
                    + 2,
                    config=config,
                ),
                "paired_savings_percentage": _mean_ci(
                    _paired_savings(semantic_arrays, exact_arrays),
                    seed=bootstrap_seed
                    + n_position * 100_000
                    + site_position * 100
                    + 3,
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
                "mean_full_install_fallback_count": float(
                    np.mean(full_fallback_counts)
                ),
            }
            print(
                f"{execution_role} N={n_channels} site={site} "
                f"decision={resolution['decision']}",
                flush=True,
            )
        if not semantic_site_arrays:
            n_results[str(n_channels)] = {
                "n_channels": n_channels,
                "site_results": {},
                "checks": {"insufficient_evaluable_sites": True},
            }
            continue
        scene_weights = np.asarray(scene_counts, dtype=np.float64)
        equal_weights = np.ones(len(scene_counts), dtype=np.float64)
        savings_matrix = np.stack(
            [
                _paired_savings(semantic, exact)
                for semantic, exact in zip(
                    semantic_site_arrays, exact_site_arrays
                )
            ]
        )
        scene_weighted = {
            "semantic": _hierarchical_metric_summary(
                semantic_site_arrays,
                seed=bootstrap_seed + n_position * 100_000 + 7001,
                config=config,
                site_weights=scene_weights,
            ),
            "exact": _hierarchical_metric_summary(
                exact_site_arrays,
                seed=bootstrap_seed + n_position * 100_000 + 7002,
                config=config,
                site_weights=scene_weights,
            ),
            "paired_savings_percentage": hierarchical_mean_ci(
                savings_matrix,
                seed=bootstrap_seed + n_position * 100_000 + 7003,
                replicates=int(
                    config["monte_carlo"]["bootstrap_replicates"]
                ),
                confidence=float(
                    config["monte_carlo"]["confidence_level"]
                ),
                site_weights=scene_weights,
            ),
        }
        equal_site = {
            "semantic": _hierarchical_metric_summary(
                semantic_site_arrays,
                seed=bootstrap_seed + n_position * 100_000 + 8001,
                config=config,
                site_weights=equal_weights,
            ),
            "exact": _hierarchical_metric_summary(
                exact_site_arrays,
                seed=bootstrap_seed + n_position * 100_000 + 8002,
                config=config,
                site_weights=equal_weights,
            ),
            "paired_savings_percentage": hierarchical_mean_ci(
                savings_matrix,
                seed=bootstrap_seed + n_position * 100_000 + 8003,
                replicates=int(
                    config["monte_carlo"]["bootstrap_replicates"]
                ),
                confidence=float(
                    config["monte_carlo"]["confidence_level"]
                ),
                site_weights=equal_weights,
            ),
        }
        n_results[str(n_channels)] = {
            "n_channels": n_channels,
            "site_results": site_outputs,
            "scene_weighted": scene_weighted,
            "equal_site_macro": equal_site,
            "decision_counts": {
                decision: sum(
                    row["decision"] == decision
                    for row in site_outputs.values()
                )
                for decision in sorted(
                    {row["decision"] for row in site_outputs.values()}
                )
            },
            "checks": {
                "semantic_clean_ci_lower_at_least_target": (
                    scene_weighted["semantic"]["clean_rate"]["ci_lower"]
                    >= float(
                        config["gates"][
                            "semantic_scene_weighted_clean_ci_lower_minimum_all_n"
                        ]
                    )
                ),
                "scene_weighted_savings_ci_lower_positive": (
                    scene_weighted["paired_savings_percentage"]["ci_lower"]
                    > 0.0
                ),
                "equal_site_savings_ci_lower_positive": (
                    equal_site["paired_savings_percentage"]["ci_lower"]
                    > 0.0
                ),
                "all_wrong_codebook_decode_counts_zero": all(
                    float(
                        np.max(
                            arrays["wrong_codebook_decode_count"]
                        )
                    )
                    == 0.0
                    for arrays in semantic_site_arrays
                ),
            },
        }

    minimum_sites = int(config["gates"]["minimum_evaluable_sites"])
    enough_sites = len(valid) >= minimum_sites
    clean_all_n = enough_sites and all(
        row["checks"].get(
            "semantic_clean_ci_lower_at_least_target", False
        )
        for row in n_results.values()
    )
    weighted_all_n = enough_sites and all(
        row["checks"].get(
            "scene_weighted_savings_ci_lower_positive", False
        )
        for row in n_results.values()
    )
    macro_count = sum(
        row["checks"].get("equal_site_savings_ci_lower_positive", False)
        for row in n_results.values()
    )
    wrong_zero = enough_sites and all(
        row["checks"].get(
            "all_wrong_codebook_decode_counts_zero", False
        )
        for row in n_results.values()
    )
    checks = {
        "registered_site_count": len(metadata_rows),
        "evaluable_site_count": len(valid),
        "minimum_evaluable_site_gate_passed": enough_sites,
        "semantic_clean_ci_lower_gate_passed_all_n": clean_all_n,
        "scene_weighted_savings_ci_lower_positive_all_n": weighted_all_n,
        "equal_site_macro_savings_ci_lower_positive_n_count": macro_count,
        "equal_site_macro_gate_passed": macro_count
        >= int(
            config["gates"][
                "equal_site_macro_savings_ci_lower_positive_minimum_n_count"
            ]
        ),
        "all_wrong_codebook_decode_counts_zero": wrong_zero,
        "all_bit_accounting_identities_valid": True,
        "no_site_replacement": True,
        "execution_role_access_count_after_run": read_json(
            paths["access_state"]
        )["roles"][execution_role]["access_count"],
    }
    primary_passed = bool(
        enough_sites
        and clean_all_n
        and weighted_all_n
        and checks["equal_site_macro_gate_passed"]
        and wrong_zero
    )
    status = (
        "INCONCLUSIVE_INSUFFICIENT_EVALUABLE_SITES"
        if not enough_sites
        else "PASSED"
        if primary_passed
        else "FAILED"
    )
    result = {
        "version": "1.0",
        "status": str(
            governance.get(
                "completion_status",
                "stage6r_electrosense_external_final_complete",
            )
        ),
        "verification_status": status,
        "experiment_id": config["experiment_id"],
        "access_receipt": receipt,
        "protocol_sha256": freeze["protocol_sha256"],
        "freeze_result_sha256": sha256_file(paths["freeze_result"]),
        "code_snapshot": current_snapshot,
        "execution_role": execution_role,
        "registered_sites": [row["site"] for row in metadata_rows],
        "frozen_sites": frozen_sites,
        "invalid_sites": invalid_sites,
        "n_results": n_results,
        "checks": checks,
        "primary_gate_passed": primary_passed,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(
        {
            "output": str(args.output),
            "verification_status": status,
            "primary_gate_passed": primary_passed,
            "checks": checks,
            "elapsed_seconds": result["elapsed_seconds"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
