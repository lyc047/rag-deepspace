#!/usr/bin/env python
"""Run an excluded-pilot smoke test for Stage-6 R1 protocols."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_grouped_validation import _prepare_n  # noqa: E402
from run_stage6_matched_reliability_coarse_grid import (  # noqa: E402
    _session_random,
)
from run_stage6_task_codebook_development import (  # noqa: E402
    grouped_train_evaluation_indices,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import (  # noqa: E402
    environment_snapshot,
    sha256_file,
)
from spectrum_semcom.stage6_matched_reliability import (  # noqa: E402
    RANDOM_STREAM_COUNT,
    build_grouped_sessions,
    combine_matched_trajectory_results,
    simulate_exact_trajectory,
)
from spectrum_semcom.stage6_reliability_protocol_r1 import (  # noqa: E402
    R1_VARIANTS,
    combine_r1_trajectory_results,
    simulate_r1_trajectory,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs/stage6_reliability_protocol_diagnostic_v1.json"
)


def _metric(value) -> dict[str, float]:
    scenes = len(value.clean)
    bits = value.bit_breakdown
    result = {
        "clean_rate": float(np.mean(value.clean)),
        "availability_rate": float(np.mean(value.available)),
        "actual_bits_per_scene": float(
            bits.actual_total_application_bits / scenes
        ),
        "forward_bits_per_scene": float(
            bits.actual_forward_bits / scenes
        ),
        "feedback_bits_per_scene": float(
            bits.actual_feedback_bits / scenes
        ),
        "task_frame_bits_per_scene": float(
            bits.task_frame_bits / scenes
        ),
        "initial_install_bits_per_scene": float(
            bits.initial_install_bits / scenes
        ),
        "recovery_install_bits_per_scene": float(
            bits.recovery_install_bits / scenes
        ),
        "heartbeat_bits_per_scene": float(
            (
                bits.heartbeat_request_bits
                + bits.heartbeat_response_bits
            )
            / scenes
        ),
        "wrong_codebook_decode_count": float(
            value.wrong_codebook_decode_count
        ),
        "rejected_compact_count": float(value.rejected_compact_count),
    }
    for field in (
        "initial_install_count",
        "recovery_install_count",
        "compact_refresh_count",
        "exact_refresh_count",
        "actual_action_reset_count",
        "actual_codebook_reset_count",
        "missing_update_ack_uncertainty_count",
        "failed_heartbeat_uncertainty_count",
    ):
        if hasattr(value, field):
            result[field] = float(getattr(value, field))
    return result


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    protocol_path = DEFAULT_PROTOCOL.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = PROJECT_DIR / protocol["outputs"]["r1_smoke"]
    if output.exists():
        raise FileExistsError("refusing to overwrite R1 smoke result")
    frozen = protocol["frozen_inputs"]
    for key in (
        "grouped_candidate_protocol",
        "fixed_site_cache",
        "excluded_pilot_cache",
    ):
        entry = frozen[key]
        if sha256_file(PROJECT_DIR / entry["path"]) != entry["sha256"]:
            raise ValueError(f"{key} changed")
    candidates = json.loads(
        (PROJECT_DIR / frozen["grouped_candidate_protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    parent = json.loads(
        (
            PROJECT_DIR / candidates["parent_protocol"]["path"]
        ).read_text(encoding="utf-8")
    )
    with np.load(
        PROJECT_DIR / frozen["excluded_pilot_cache"]["path"],
        allow_pickle=False,
    ) as handle:
        pilot = {key: handle[key] for key in handle.files}
    split = parent["coarse_split"]
    _, evaluation_indices, _, _ = grouped_train_evaluation_indices(
        pilot[split["group_field"]],
        seed=int(split["seed"]),
        train_fraction=float(split["train_group_fraction"]),
    )
    sessions = build_grouped_sessions(
        evaluation_indices,
        pilot[split["group_field"]],
        target_scene_count=20,
        minimum_remainder_scenes=2,
    )
    positions = {
        int(index): position
        for position, index in enumerate(evaluation_indices)
    }
    trajectories = 12
    results = {}
    for n_position, n_channels in enumerate(
        protocol["evaluation_scope"]["n_channels"]
    ):
        prepared = _prepare_n(
            int(n_channels), parent, pilot, pilot
        )
        controller = candidates["controller_candidates"][
            str(n_channels)
        ][0]
        common_random = np.random.default_rng(
            20260728 + 30_000_000 + n_position * 100_000
        ).random(
            (
                trajectories,
                evaluation_indices.size,
                RANDOM_STREAM_COUNT,
            )
        )
        variant_rows = []
        for deployment in protocol["evaluation_scope"][
            "deployment_modes"
        ]:
            for variant in R1_VARIANTS:
                trajectory_rows = []
                for trajectory in range(trajectories):
                    session_rows = [
                        simulate_r1_trajectory(
                            prepared["states"],
                            pilot["timestamps_local"],
                            session,
                            variant=variant,
                            sender_session=prepared["sender"],
                            receiver_session=prepared["receiver"],
                            install_packet=prepared["install_packet"],
                            deployment_mode=deployment,
                            condition=parent["primary_fault_condition"],
                            random_values=_session_random(
                                common_random[trajectory],
                                session,
                                positions,
                            ),
                            epsilon_db=float(
                                parent["task_grid"]["epsilon_db"]
                            ),
                            max_age_minutes=float(
                                controller["maximum_state_age_minutes"]
                            ),
                            ack_frame_bits=prepared["ack_bits"],
                            outage_penalty_db=float(
                                parent["task_grid"]["outage_penalty_db"]
                            ),
                            heartbeat_interval_scenes=int(
                                controller["heartbeat_silence_scenes"]
                            ),
                            heartbeat_request_frame_bits=prepared[
                                "heartbeat_request_bits"
                            ],
                            heartbeat_response_frame_bits=prepared[
                                "heartbeat_response_bits"
                            ],
                            task_open_loop_attempts=int(
                                controller[
                                    "task_update_open_loop_attempts"
                                ]
                            ),
                            compact_codeword_frame_bits=prepared[
                                "compact_frame_bits"
                            ],
                            exact_action_frame_bits=prepared[
                                "exact_packet_bits"
                            ],
                        )
                        for session in sessions
                    ]
                    trajectory_rows.append(
                        _metric(
                            combine_r1_trajectory_results(session_rows)
                        )
                    )
                variant_rows.append(
                    {
                        "variant_id": variant,
                        "deployment_mode": deployment,
                        "requested_session_scene_count": 20,
                        "evaluated_scene_count": int(
                            sum(len(session) for session in sessions)
                        ),
                        "trajectory_metrics": trajectory_rows,
                        "summary": _summary(trajectory_rows),
                    }
                )
        exact_trajectories = []
        for trajectory in range(trajectories):
            session_rows = [
                simulate_exact_trajectory(
                    prepared["states"],
                    pilot["timestamps_local"],
                    session,
                    random_values=_session_random(
                        common_random[trajectory], session, positions
                    ),
                    packet_loss_probability=float(
                        parent["primary_fault_condition"][
                            "task_loss_probability"
                        ]
                    ),
                    receiver_reset_probability=float(
                        parent["primary_fault_condition"][
                            "receiver_context_reset_probability"
                        ]
                    ),
                    task_open_loop_attempts=1,
                    packet_bits=prepared["exact_packet_bits"],
                    epsilon_db=float(
                        parent["task_grid"]["epsilon_db"]
                    ),
                    outage_penalty_db=float(
                        parent["task_grid"]["outage_penalty_db"]
                    ),
                )
                for session in sessions
            ]
            exact_trajectories.append(
                _metric(combine_matched_trajectory_results(session_rows))
            )
        results[str(n_channels)] = {
            "controller": controller,
            "variant_workpoints": variant_rows,
            "exact_one_attempt": {
                "trajectory_metrics": exact_trajectories,
                "summary": _summary(exact_trajectories),
            },
        }
    wrong = sum(
        row["summary"]["wrong_codebook_decode_count"]
        for value in results.values()
        for row in value["variant_workpoints"]
    )
    result = {
        "version": "1.0",
        "status": "r1_excluded_pilot_smoke_complete",
        "protocol_sha256": sha256_file(protocol_path),
        "source": {
            "path": frozen["excluded_pilot_cache"]["path"],
            "sha256": frozen["excluded_pilot_cache"]["sha256"],
            "role": frozen["excluded_pilot_cache"]["role"],
            "evaluation_scene_count": int(evaluation_indices.size),
        },
        "trajectory_count": trajectories,
        "results": results,
        "checks": {
            "all_variants_complete": True,
            "wrong_codebook_actions_must_be_zero": wrong == 0.0,
            "external_final_signal_values_not_loaded": True,
            "external_final_access_count_remains_zero": True,
        },
        "environment": environment_snapshot(["numpy"]),
        "claim_boundary": (
            "Instrumentation smoke on the excluded 2023 pilot. It cannot "
            "select or confirm an R1 protocol."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
