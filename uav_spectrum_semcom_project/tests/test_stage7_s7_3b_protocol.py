import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _result() -> dict:
    path = (
        PROJECT_DIR
        / "results/stage7/s7_3b_full_protocol_oracle_v1/result.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_s7_3b_keeps_closed_splits_unread() -> None:
    result = _result()
    assert len(result["loaded_sites"]["validation"]) == 7
    assert result["loaded_sites"]["train"] == []
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]


def test_s7_3b_oracle_fails_registered_system_gate() -> None:
    result = _result()
    assert result["decision"]["passing_n_count"] == 0
    assert not result["decision"]["diagnostic_gate_passed"]
    assert result["decision"]["next_action"] == (
        "stop_task_forecast_protection_and_focus_on_analytic_context_belief_"
        "or_periodically_self_contained_protocols"
    )
    assert all(
        not row["gate"]["n_gate_passed"]
        for row in result["n_results"].values()
    )


def test_s7_3b_oracle_is_mostly_redundant_and_cost_increasing() -> None:
    result = _result()
    oracle = "fixed10_plus_forced_refresh_oracle"
    for row in result["n_results"].values():
        trigger = row["trigger_diagnostics"][oracle]
        comparison = row["paired_comparisons_vs_fixed10"][oracle]
        assert trigger["redundant_fraction"] > 0.8
        assert (
            comparison["paired_clean_gain_percentage_points"]["mean"]
            < 0.1
        )
        assert (
            comparison["paired_total_bit_reduction_percentage"]["mean"]
            < 0.0
        )


def test_s7_3b_never_executes_wrong_codebook_action() -> None:
    result = _result()
    assert result["governance_checks"][
        "full_packet_ack_reset_recovery_accounting_used"
    ]
    assert result["governance_checks"]["oracle_is_noncausal"]
    for row in result["n_results"].values():
        assert all(
            count == 0
            for count in row["wrong_codebook_decode_total"].values()
        )
