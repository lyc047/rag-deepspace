import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _result(name: str = "result.json") -> dict:
    path = (
        PROJECT_DIR
        / "results/stage7/s7_4a1_checkpoint_protocol_scan_v1"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_s7_4a_keeps_closed_splits_unread() -> None:
    result = _result()
    assert len(result["loaded_sites"]["validation"]) == 7
    assert result["loaded_sites"]["train"] == []
    assert result["loaded_sites"]["internal_development_test"] == []
    assert result["loaded_sites"]["reserve"] == []
    assert result["governance_checks"][
        "internal_development_test_not_loaded"
    ]
    assert result["governance_checks"]["reserve_remained_unread"]


def test_s7_4a_periodic_checkpoint_fails_registered_gate() -> None:
    result = _result()
    decision = result["decision"]
    assert not decision["system_gate_passed"]
    assert decision["required_n_count"] == 3
    assert all(
        count == 0
        for count in decision["policy_passing_n_count"].values()
    )
    assert decision["next_action"] == (
        "reject_periodic_checkpoint_and_move_to_event_update_context_"
        "digest_piggyback"
    )


def test_s7_4a_checkpoint10_has_no_cross_n_bit_advantage() -> None:
    result = _result()
    reductions = {
        int(n): row["paired_comparisons_vs_fixed10"]["checkpoint10"][
            "paired_total_bit_reduction_percentage"
        ]["mean"]
        for n, row in result["n_results"].items()
    }
    assert reductions[8] > 0.0
    assert reductions[64] > 0.0
    assert reductions[16] < 0.0
    assert reductions[32] < 0.0


def test_s7_4a_never_executes_wrong_codebook_action() -> None:
    result = _result()
    assert result["governance_checks"][
        "full_packet_ack_reset_recovery_accounting_used"
    ]
    for row in result["n_results"].values():
        assert all(
            count == 0
            for count in row["wrong_codebook_decode_total"].values()
        )


def test_s7_4a_reproduction_matches_except_elapsed_time() -> None:
    original = _result()
    reproduction = _result("reproduction.json")
    original.pop("elapsed_seconds")
    reproduction.pop("elapsed_seconds")
    assert reproduction == original
