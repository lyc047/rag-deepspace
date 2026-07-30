import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    PROJECT_DIR
    / "results/stage7/s7_4b0_event_piggyback_feasibility_v1"
)


def _load(name: str = "result.json") -> dict:
    return json.loads((RESULT_DIR / name).read_text(encoding="utf-8"))


def test_s7_4b_keeps_closed_splits_unread() -> None:
    result = _load()
    assert len(result["loaded_sites"]["validation"]) == 7
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]


def test_s7_4b_fails_common_policy_gate() -> None:
    decision = _load()["decision"]
    assert not decision["system_gate_passed"]
    assert all(
        value == 0
        for value in decision["policy_passing_n_count"].values()
    )
    assert decision["next_action"] == (
        "stop_adding_stateful_context_components_and_audit_semi_stateless_"
        "headers"
    )


def test_s7_4b_has_zero_wrong_codebook_actions() -> None:
    for row in _load()["n_results"].values():
        assert all(
            value == 0
            for value in row["wrong_codebook_decode_total"].values()
        )


def test_s7_4b_reproduction_matches_except_elapsed() -> None:
    original = _load()
    reproduction = _load("reproduction.json")
    original.pop("elapsed_seconds")
    reproduction.pop("elapsed_seconds")
    assert reproduction == original
