import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads(
        (PROJECT_DIR / relative).read_text(encoding="utf-8")
    )


def test_s7_2_keeps_internal_test_and_reserve_closed() -> None:
    result = _read(
        "results/stage7/s7_2_lightweight_models_v1/result.json"
    )
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]
    assert not result["governance_checks"][
        "site_identifier_used_as_feature"
    ]


def test_s7_2_negative_result_is_preserved() -> None:
    result = _read(
        "results/stage7/s7_2_lightweight_models_v1/result.json"
    )
    assert result["decision"]["passing_n_count"] == 0
    assert not result["decision"]["candidate_freeze_gate_passed"]
    assert result["decision"]["next_action"] == (
        "stop_model_escalation_and_reformulate_task_events_or_protocol"
    )
    assert all(
        not row["freeze_gate_passed"]
        for row in result["n_results"].values()
    )
    assert all(
        row["s7_1_comparison"]["s7_2_macro_within_site_auc"]
        < row["s7_1_comparison"]["s7_1_macro_within_site_auc"]
        for row in result["n_results"].values()
    )


def test_s7_2_selected_models_are_calibrated_but_not_freezable() -> None:
    result = _read(
        "results/stage7/s7_2_lightweight_models_v1/result.json"
    )
    for row in result["n_results"].values():
        selected = row["models"][row["selected_model"]]
        metrics = selected["validation_aggregate"]
        assert metrics["brier_skill_over_constant"] > 0.0
        assert metrics["expected_calibration_error"] <= 0.08
        assert metrics["recall_at_frozen_threshold"] >= 0.8
