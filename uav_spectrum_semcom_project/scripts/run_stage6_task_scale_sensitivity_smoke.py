#!/usr/bin/env python
"""Small deterministic execution check for the registered S6.7c runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
for item in (PROJECT_DIR / "src", PROJECT_DIR / "scripts"):
    sys.path.insert(0, str(item))

from run_stage6_task_scale_sensitivity import (  # noqa: E402
    DEFAULT_PROTOCOL,
    _prepare_grid_point,
    _run_group,
    _verified_inputs,
)
from spectrum_semcom.final_holdout import atomic_write_json  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402
from spectrum_semcom.stage6_task_scale_sensitivity import (  # noqa: E402
    registered_grid_points,
)


DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6/task_scale_sensitivity_smoke_v1/result.json"
)


def main() -> None:
    protocol, parent, candidates, fixed, pilot = _verified_inputs(
        DEFAULT_PROTOCOL
    )
    smoke_protocol = json.loads(json.dumps(protocol))
    smoke_protocol["system_evaluation"]["link_trajectories"] = 2
    smoke_protocol["frozen_architecture"]["session_scene_count"] = 20
    point = [
        value
        for value in registered_grid_points(protocol)
        if value.n_channels == 8
        and abs(value.epsilon_db - 0.2) < 1e-12
        and value.query_set_id == "multi_025_050_075"
    ][0]
    prepared = _prepare_grid_point(
        point, protocol, parent, pilot, fixed
    )
    if not prepared["deployable"]:
        raise ValueError("registered primary point is not deployable")
    group_ids = sorted(set(fixed["outer_group_ids"].astype(str)))
    group_id = next(
        value
        for value in group_ids
        if int((fixed["outer_group_ids"].astype(str) == value).sum()) >= 20
    )
    group_position = group_ids.index(group_id)
    group = _run_group(
        prepared,
        group_id,
        group_position,
        0,
        smoke_protocol,
        parent,
        candidates,
        fixed,
    )
    semantic = group["semantic_workpoint"]
    value = {
        "version": "1.0",
        "status": "pass",
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "grid_point_id": point.grid_point_id,
        "outer_group_id": group_id,
        "trajectory_count": 2,
        "session_scene_count": 20,
        "representation": prepared["representation"],
        "semantic_summary": semantic["summary"],
        "exact_summaries": [
            {
                "attempts": row["task_packet_open_loop_attempts"],
                "summary": row["summary"],
            }
            for row in group["exact_workpoints"]
        ],
        "checks": {
            "wrong_codebook_actions_must_be_zero": semantic["summary"][
                "wrong_codebook_decode_count"
            ]
            == 0.0,
            "all_bit_metrics_are_non_negative": all(
                row["actual_bits_per_scene"] >= 0.0
                for row in semantic["trajectory_metrics"]
            ),
            "all_clean_rates_are_probabilities": all(
                0.0 <= row["clean_rate"] <= 1.0
                for row in semantic["trajectory_metrics"]
            ),
            "external_final_signal_values_loaded": False,
            "external_final_access_count": 0,
        },
        "claim_boundary": "Execution smoke only; not a sensitivity result.",
    }
    if not (
        value["checks"]["wrong_codebook_actions_must_be_zero"]
        and value["checks"]["all_bit_metrics_are_non_negative"]
        and value["checks"]["all_clean_rates_are_probabilities"]
        and not value["checks"]["external_final_signal_values_loaded"]
        and value["checks"]["external_final_access_count"] == 0
    ):
        raise ValueError("S6.7c smoke check failed")
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(DEFAULT_OUTPUT, value)


if __name__ == "__main__":
    main()
