import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from run_stage6_fault_probability_stress import (  # noqa: E402
    FAULT_FIELDS,
    _condition_key,
)


def test_condition_key_tracks_only_scanned_fault_dimensions() -> None:
    condition = {
        "install_loss_probability": 0.1,
        "task_loss_probability": 0.2,
        "ack_loss_probability": 0.3,
        "receiver_context_reset_probability": 0.04,
        "delayed_duplicate_probability": 0.9,
    }
    assert _condition_key(condition) == (0.1, 0.2, 0.3, 0.04)


def test_fault_stress_protocol_is_one_factor_and_final_closed() -> None:
    protocol = json.loads(
        (
            PROJECT
            / "configs/stage6_fault_probability_stress_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(protocol["one_factor_scans"]) == set(FAULT_FIELDS)
    assert protocol["analysis_rules"][
        "multiple_dimension_results_must_not_be_combined_as_if_jointly_observed"
    ]
    assert not protocol["analysis_rules"][
        "results_may_modify_frozen_architecture_or_parameters"
    ]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]
