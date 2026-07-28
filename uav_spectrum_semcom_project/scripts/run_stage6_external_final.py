#!/usr/bin/env python
"""Run the frozen single-access Stage-6 external Final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage5_external_final import load_scene_arrays  # noqa: E402
from run_stage6_grouped_validation import _group_random  # noqa: E402
from run_stage6_matched_reliability_coarse_grid import (  # noqa: E402
    _find_compact_frame_bits,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from run_stage6_task_scale_sensitivity import (  # noqa: E402
    _decision_summary,
    _exact_trajectory_metric,
    _summary,
    _trajectory_metric,
)
from spectrum_semcom.final_holdout import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
)
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
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
from spectrum_semcom.stage6_final_governance import (  # noqa: E402
    validate_external_final_catalog,
    verify_stage6_code_snapshot,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    combine_matched_trajectory_results,
    exact_query_bundle_bits,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_reliability_protocol_r1 import (  # noqa: E402
    combine_r1_trajectory_results,
    simulate_r1_trajectory,
)
from spectrum_semcom.stage6_task_codebook import (  # noqa: E402
    SpectrumTaskQuery,
    build_task_state,
    encode_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_task_codec import install_codebook  # noqa: E402


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_external_final_protocol_v1.json"
)
DEFAULT_REGISTRY = (
    PROJECT_DIR / "configs/stage5_external_final_registry_v1.json"
)
DEFAULT_STATE = (
    PROJECT_DIR / "configs/stage6_external_final_access_state.json"
)
DEFAULT_SNAPSHOT = (
    PROJECT_DIR / "results/stage6/external_final_freeze_v1/code_snapshot.json"
)
DEFAULT_CATALOG = (
    PROJECT_DIR / "results/stage6/external_final_catalog_v1/catalog.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "results/stage6/external_final_v1/final_raw.json"
)


def verify_frozen_evidence(protocol: dict) -> list[str]:
    errors = []
    for label, entry in protocol["frozen_development_evidence"].items():
        if sha256_file(PROJECT_DIR / entry["path"]) != entry["sha256"]:
            errors.append(f"frozen development evidence changed: {label}")
    return errors


def verify_inputs(
    protocol: dict,
    registry: dict,
    state: dict,
    snapshot: dict,
    catalog: dict,
    *,
    preflight: bool,
) -> list[str]:
    errors = verify_frozen_evidence(protocol)
    errors.extend(verify_stage6_code_snapshot(PROJECT_DIR, snapshot))
    errors.extend(validate_external_final_catalog(catalog, protocol, registry))
    if catalog.get("code_snapshot_sha256") != snapshot.get(
        "executable_snapshot_sha256"
    ):
        errors.append("catalog and Stage-6 code snapshot are not bound")
    if preflight:
        if (
            state.get("status") != "downloaded_integrity_verified_not_accessed"
            or state.get("access_count") != 0
            or state.get("final_signal_values_accessed") is not False
        ):
            errors.append("preflight requires pristine Stage-6 access state")
    else:
        if (
            state.get("status") != "access_consumed"
            or state.get("access_count") != 1
            or state.get("reset_permitted") is not False
            or state.get("final_signal_values_accessed") is not True
        ):
            errors.append("Final execution requires atomic access receipt")
        receipt = state.get("receipt", {})
        if receipt.get("code_snapshot_sha256") != snapshot.get(
            "executable_snapshot_sha256"
        ):
            errors.append("access receipt is not bound to snapshot")
        if receipt.get("catalog_sha256") != canonical_json_sha256(catalog):
            errors.append("access receipt is not bound to catalog")
    for entry in registry["final_candidates"]:
        archive = Path(registry["local_root"]) / entry["archive"]
        if (
            not archive.is_file()
            or archive.stat().st_size != int(entry["archive_size_bytes"])
            or sha256_file(archive) != entry["archive_sha256"]
        ):
            errors.append(f"archive integrity failed: {archive}")
    return errors


def load_pilot(protocol: dict) -> dict[str, np.ndarray]:
    entry = protocol["frozen_development_evidence"][
        "excluded_2023_pilot_cache"
    ]
    with np.load(PROJECT_DIR / entry["path"], allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def task_queries(protocol: dict, n_channels: int) -> tuple[SpectrumTaskQuery, ...]:
    demands = tuple(
        int(round(n_channels * float(ratio)))
        for ratio in protocol["resource_task"]["demand_ratios"]
    )
    return tuple(SpectrumTaskQuery(value) for value in demands)


def prepare_n(
    n_channels: int,
    protocol: dict,
    pilot: dict[str, np.ndarray],
    records: list[dict],
) -> dict:
    epsilon = float(protocol["resource_task"]["regret_threshold_db"])
    queries = task_queries(protocol, n_channels)
    pilot_states = [
        build_task_state(values, queries, epsilon_db=epsilon)
        for values in pilot[f"channel_power_n{n_channels}"]
    ]
    architecture = protocol["frozen_architecture"]
    train_indices, evaluation_indices, train_groups, evaluation_groups = (
        grouped_train_evaluation_indices(
            pilot[architecture["codebook_fit_group_field"]],
            seed=int(architecture["codebook_fit_seed"]),
            train_fraction=float(
                architecture["codebook_fit_train_group_fraction"]
            ),
        )
    )
    training_states = [pilot_states[int(index)] for index in train_indices]
    codebook = fit_greedy_task_codebook(training_states)
    final_states = [
        build_task_state(
            record["n_data"][str(n_channels)]["channel_power_dbm"],
            queries,
            epsilon_db=epsilon,
        )
        for record in records
    ]
    pilot_decisions = [
        encode_task_state(codebook, pilot_states[int(index)])
        for index in evaluation_indices
    ]
    final_decisions = [
        encode_task_state(codebook, state) for state in final_states
    ]
    training_actions = {state.optimal_actions for state in training_states}
    representation = {
        "n_channels": int(n_channels),
        "epsilon_db": epsilon,
        "query_set_id": protocol["resource_task"]["query_set_id"],
        "demands": [int(value.demand_channels) for value in queries],
        "pilot_train_scene_count": len(train_indices),
        "pilot_evaluation_scene_count": len(evaluation_indices),
        "pilot_train_group_count": len(train_groups),
        "pilot_evaluation_group_count": len(evaluation_groups),
        "training_exact_action_tuple_count": int(
            codebook.exact_action_tuple_count
        ),
        "codeword_count": len(codebook.codewords),
        "symbol_width_bits": int(codebook.symbol_width_bits),
        "training_coverage_rate": float(
            codebook.training_covered_count / codebook.training_scene_count
        ),
        "pilot_evaluation": _decision_summary(
            pilot_decisions, codebook
        ),
        "external_final": _decision_summary(final_decisions, codebook),
        "new_exact_action_tuple_rate_external": float(
            np.mean(
                [
                    state.optimal_actions not in training_actions
                    for state in final_states
                ]
            )
        ),
        "decoder_actions": [
            list(value.decoder_actions) for value in codebook.codewords
        ],
    }
    if not 1 <= len(codebook.codewords) <= 255:
        return {
            "deployable": False,
            "states": final_states,
            "representation": representation
            | {
                "deployable_in_current_context_codec": False,
                "non_deployable_reason": "codeword_count_outside_1_to_255",
            },
        }
    sender = install_codebook(
        codebook, epoch=int(protocol["resource_task"]["codebook_epoch"])
    )
    install_packet = encode_context_install(sender, node_id=1)
    receiver = decode_context_install(install_packet).session
    compact_bits = _find_compact_frame_bits(
        pilot_states, train_indices, sender
    )
    exact_bits = exact_query_bundle_bits(
        n_channels,
        queries,
        header_bits=int(protocol["resource_task"]["task_header_bits"]),
    )
    request_bits = int(
        encode_context_probe_request(
            node_id=1,
            codebook_epoch=sender.epoch,
            expected_update_epoch=0,
        ).size
    )
    response_bits = int(
        encode_context_probe_response(
            node_id=1,
            codebook_epoch=sender.epoch,
            current_update_epoch=0,
        ).size
    )
    representation |= {
        "deployable_in_current_context_codec": True,
        "non_deployable_reason": None,
        "context_install_bits": int(install_packet.size),
        "compact_frame_bits": int(compact_bits),
        "exact_action_frame_bits": int(exact_bits),
        "codebook_manifest_sha256": sender.manifest_sha256,
    }
    return {
        "deployable": True,
        "states": final_states,
        "representation": representation,
        "sender": sender,
        "receiver": receiver,
        "install_packet": install_packet,
        "compact_frame_bits": int(compact_bits),
        "exact_packet_bits": int(exact_bits),
        "ack_bits": int(cumulative_ack_payload_bits()),
        "heartbeat_request_bits": request_bits,
        "heartbeat_response_bits": response_bits,
    }


def evaluate_n(
    n_channels: int,
    n_position: int,
    protocol: dict,
    records: list[dict],
    prepared: dict,
) -> dict:
    architecture = protocol["frozen_architecture"]
    evaluation = protocol["system_evaluation"]
    fault = protocol["fault_model"]
    timestamps = np.asarray(
        [record["timestamp_local"] for record in records], dtype="U32"
    )
    campaigns = sorted({record["campaign_id"] for record in records})
    outer_groups: dict[str, dict] = {}
    for group_position, campaign in enumerate(campaigns):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record["campaign_id"] == campaign
            ],
            dtype=np.int64,
        )
        random = _group_random(
            trajectory_count=int(evaluation["link_trajectories"]),
            scene_count=int(indices.size),
            base_seed=int(evaluation["random_seed"]),
            n_position=n_position,
            group_position=group_position,
        )
        semantic_metrics = []
        exact_metrics = {
            int(attempts): []
            for attempts in evaluation["exact_reference_open_loop_attempts"]
        }
        for trajectory in range(int(evaluation["link_trajectories"])):
            semantic = simulate_r1_trajectory(
                prepared["states"],
                timestamps,
                indices,
                variant=architecture["semantic_variant"],
                sender_session=prepared["sender"],
                receiver_session=prepared["receiver"],
                install_packet=prepared["install_packet"],
                deployment_mode=architecture["deployment_mode"],
                condition=fault,
                random_values=random[trajectory],
                epsilon_db=float(protocol["resource_task"]["regret_threshold_db"]),
                max_age_minutes=float(
                    architecture["controller_parameters"][
                        "maximum_state_age_minutes"
                    ]
                ),
                ack_frame_bits=prepared["ack_bits"],
                outage_penalty_db=float(
                    protocol["resource_task"]["outage_penalty_db"]
                ),
                heartbeat_interval_scenes=int(
                    architecture["controller_parameters"][
                        "heartbeat_silence_scenes"
                    ]
                ),
                heartbeat_request_frame_bits=prepared[
                    "heartbeat_request_bits"
                ],
                heartbeat_response_frame_bits=prepared[
                    "heartbeat_response_bits"
                ],
                task_open_loop_attempts=int(
                    architecture["controller_parameters"][
                        "task_update_open_loop_attempts"
                    ]
                ),
                compact_codeword_frame_bits=prepared["compact_frame_bits"],
                exact_action_frame_bits=prepared["exact_packet_bits"],
            )
            semantic_metrics.append(_trajectory_metric(semantic))
            for attempts in exact_metrics:
                exact = simulate_exact_trajectory(
                    prepared["states"],
                    timestamps,
                    indices,
                    random_values=random[trajectory],
                    packet_loss_probability=float(
                        fault["task_loss_probability"]
                    ),
                    receiver_reset_probability=float(
                        fault["receiver_context_reset_probability"]
                    ),
                    task_open_loop_attempts=attempts,
                    packet_bits=prepared["exact_packet_bits"],
                    epsilon_db=float(
                        protocol["resource_task"]["regret_threshold_db"]
                    ),
                    outage_penalty_db=float(
                        protocol["resource_task"]["outage_penalty_db"]
                    ),
                )
                exact_metrics[attempts].append(
                    _exact_trajectory_metric(exact)
                )
        outer_groups[campaign] = {
            "outer_group_id": campaign,
            "site_id": campaign,
            "scene_count": int(indices.size),
            "status": "evaluated",
            "session_count": 1,
            "evaluated_scene_count": int(indices.size),
            "semantic_workpoint": {
                "variant_id": architecture["semantic_variant"],
                "deployment_mode": architecture["deployment_mode"],
                "summary": _summary(semantic_metrics),
                "trajectory_metrics": semantic_metrics,
            },
            "exact_workpoints": [
                {
                    "task_packet_open_loop_attempts": attempts,
                    "packet_bits": prepared["exact_packet_bits"],
                    "summary": _summary(rows),
                    "trajectory_metrics": rows,
                }
                for attempts, rows in sorted(exact_metrics.items())
            ],
        }
        print(
            f"Stage-6 Final N={n_channels} campaign={campaign} complete",
            flush=True,
        )
    return {
        "n_channels": int(n_channels),
        "status": "evaluated",
        "configured_outer_group_ids": campaigns,
        "representation": prepared["representation"],
        "outer_groups": outer_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pilot-smoke", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.preflight and args.pilot_smoke:
        raise SystemExit("select at most one of --preflight and --pilot-smoke")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.pilot_smoke:
        pilot = load_pilot(protocol)
        rows = []
        for n_channels in (
            int(protocol["resource_task"]["primary_n_channels"]),
            *map(int, protocol["resource_task"]["secondary_n_channels"]),
        ):
            queries = task_queries(protocol, n_channels)
            states = [
                build_task_state(
                    values,
                    queries,
                    epsilon_db=float(
                        protocol["resource_task"]["regret_threshold_db"]
                    ),
                )
                for values in pilot[f"channel_power_n{n_channels}"]
            ]
            train, evaluation, _, _ = grouped_train_evaluation_indices(
                pilot[
                    protocol["frozen_architecture"][
                        "codebook_fit_group_field"
                    ]
                ],
                seed=int(
                    protocol["frozen_architecture"]["codebook_fit_seed"]
                ),
                train_fraction=float(
                    protocol["frozen_architecture"][
                        "codebook_fit_train_group_fraction"
                    ]
                ),
            )
            codebook = fit_greedy_task_codebook(
                [states[int(index)] for index in train]
            )
            decisions = [
                encode_task_state(codebook, states[int(index)])
                for index in evaluation
            ]
            rows.append(
                {
                    "n_channels": n_channels,
                    "codeword_count": len(codebook.codewords),
                    "pilot_evaluation": _decision_summary(
                        decisions, codebook
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "status": "pilot_smoke_pass",
                    "signal_values_loaded": False,
                    "final_access_consumed": False,
                    "rows": rows,
                },
                indent=2,
            )
        )
        return
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    errors = verify_inputs(
        protocol,
        registry,
        state,
        snapshot,
        catalog,
        preflight=args.preflight,
    )
    if errors:
        raise ValueError("Stage-6 Final input audit failed: " + "; ".join(errors))
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "ready_for_atomic_single_access",
                    "scene_count": len(catalog["scenes"]),
                    "snapshot_sha256": snapshot[
                        "executable_snapshot_sha256"
                    ],
                    "catalog_sha256": catalog["catalog_sha256"],
                    "signal_values_loaded": False,
                    "final_access_consumed": False,
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage-6 Final output")
    records = load_scene_arrays(catalog, protocol)
    pilot = load_pilot(protocol)
    n_values = [
        int(protocol["resource_task"]["primary_n_channels"]),
        *map(int, protocol["resource_task"]["secondary_n_channels"]),
    ]
    n_results = {}
    wrong_actions = 0.0
    for n_position, n_channels in enumerate(n_values):
        prepared = prepare_n(n_channels, protocol, pilot, records)
        if not prepared["deployable"]:
            n_results[str(n_channels)] = {
                "n_channels": n_channels,
                "status": "not_deployable_in_current_codec",
                "representation": prepared["representation"],
                "outer_groups": {},
            }
            continue
        n_result = evaluate_n(
            n_channels, n_position, protocol, records, prepared
        )
        n_results[str(n_channels)] = n_result
        wrong_actions += sum(
            float(
                group["semantic_workpoint"]["summary"][
                    "wrong_codebook_decode_count"
                ]
            )
            for group in n_result["outer_groups"].values()
        )
    result = {
        "version": "1.0",
        "status": "stage6_external_final_execution_complete",
        "protocol_sha256": sha256_file(args.protocol),
        "registry_sha256": sha256_file(args.registry),
        "catalog_sha256": canonical_json_sha256(catalog),
        "access_receipt_sha256": state["receipt_sha256"],
        "code_snapshot_sha256": snapshot["executable_snapshot_sha256"],
        "scene_count": len(records),
        "campaign_ids": sorted(
            {record["campaign_id"] for record in records}
        ),
        "n_results": n_results,
        "checks": {
            "all_registered_n_retained": set(n_results)
            == {str(value) for value in n_values},
            "wrong_codebook_actions_must_be_zero": wrong_actions == 0.0,
            "external_final_access_count": 1,
            "all_methods_used_identical_scene_catalog": True,
        },
        "environment": environment_snapshot(["numpy", "scipy"]),
        "governance": {
            "access_count": 1,
            "post_final_parameter_tuning_on_this_final_permitted": False,
            "ack_faults_are_measured_claims": False,
            "multi_node_or_mimo_claim": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    updated_state = {
        **state,
        "status": "access_consumed_final_execution_complete",
        "final_method_outputs_accessed": True,
        "raw_result_sha256": sha256_file(args.output),
    }
    atomic_write_json(args.state, updated_state)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_count": len(records),
                "n_values": n_values,
                "access_count": 1,
                "post_final_tuning_permitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
