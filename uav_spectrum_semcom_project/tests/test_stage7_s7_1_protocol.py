import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads(
        (PROJECT_DIR / relative).read_text(encoding="utf-8")
    )


def test_s7_1_did_not_open_internal_test_or_reserve() -> None:
    result = _read(
        "results/stage7/s7_1_strong_baselines_v1/result.json"
    )
    split = _read("configs/stage7_development_split_v1.json")["split"]
    assert set(result["loaded_sites"]["train"]) == set(split["train"])
    assert set(result["loaded_sites"]["validation"]) == set(
        split["validation"]
    )
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]


def test_s7_1_registered_aggregate_and_posthoc_decisions_are_distinct() -> None:
    aggregate = _read(
        "results/stage7/s7_1_strong_baselines_v1/result.json"
    )
    robustness = _read(
        "results/stage7/s7_1b_site_robustness_v1/result.json"
    )
    assert aggregate["decision"]["task_structure_passing_n_count"] == 4
    assert aggregate["decision"]["full_prediction_gate_passing_n_count"] == 3
    assert robustness["analysis_timing"].startswith("posthoc_")
    assert robustness["decision"][
        "registered_s7_1_aggregate_gate_result_unchanged"
    ]
    assert robustness["decision"]["any_robustness_caution"]
    assert not robustness["decision"][
        "internal_development_test_may_be_opened"
    ]
    assert all(
        row["dominant_positive_event_share"] > 0.95
        for row in robustness["n_results"].values()
    )


def test_iid_context_risk_is_empirically_unpredictable() -> None:
    result = _read(
        "results/stage7/s7_1_strong_baselines_v1/result.json"
    )
    for row in result["n_results"].values():
        audit = row["context_iid_audit"]["1"]
        assert abs(
            audit["empirical_validation_prevalence"]
            - audit["analytic_probability"]
        ) <= 0.01
        assert 0.45 <= audit["action_duration_lookup"]["roc_auc"] <= 0.55
        assert 0.45 <= audit["heartbeat_phase_lookup"]["roc_auc"] <= 0.55
