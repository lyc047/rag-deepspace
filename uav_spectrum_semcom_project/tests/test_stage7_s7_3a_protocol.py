import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads((PROJECT_DIR / relative).read_text(encoding="utf-8"))


def test_s7_3a_keeps_closed_splits_unread() -> None:
    result = _read(
        "results/stage7/s7_3a_event_value_labels_v1/result.json"
    )
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]


def test_s7_3a_preserves_concentration_failure() -> None:
    result = _read(
        "results/stage7/s7_3a_event_value_labels_v1/result.json"
    )
    assert result["decision"]["passing_n_count"] == 0
    assert not result["decision"]["label_viability_gate_passed"]
    assert result["decision"]["next_action"] == (
        "retain_event_decomposition_but_require_more_temporally_diverse_sites_"
        "before_value_prediction"
    )
    for row in result["n_results"].values():
        assert not row["viability_checks"][
            "persistent_not_overconcentrated"
        ]
        assert not row["viability_checks"][
            "beneficial_not_overconcentrated"
        ]
        assert not row["label_viability_gate_passed"]


def test_s7_3a_label_has_value_but_not_full_protocol_evidence() -> None:
    result = _read(
        "results/stage7/s7_3a_event_value_labels_v1/result.json"
    )
    for row in result["n_results"].values():
        assert row["viability_checks"]["beneficial_predictability"]
        assert row["viability_checks"]["avoided_dirty_scenes"]
        assert row["forced_refresh_effect"][
            "median_avoided_dirty_scenes"
        ] >= 3.0
    assert result["governance_checks"][
        "full_protocol_benefit_claim_forbidden"
    ]
    assert result["governance_checks"]["external_final_claim_forbidden"]
