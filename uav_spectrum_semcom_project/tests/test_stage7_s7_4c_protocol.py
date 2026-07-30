import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/stage7/s7_4c0_semi_stateless_header_audit_v1"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_s7_4c_zero_bit_upper_bound_does_not_pass() -> None:
    result = _load("result.json")
    assert result["raw_signal_files_loaded"] == 0
    assert not result["decision"]["gate_passed"]
    for key, value in result["decision"]["passing_n_count"].items():
        assert value == 0
        if key.endswith("header0"):
            for n_result in result["n_results"].values():
                policy, _ = key.rsplit("_header", 1)
                assert not n_result[policy]["0"]["point_gate_passed"]


def test_s7_4c_freezes_fixed10_boundary() -> None:
    result = _load("result.json")
    assert result["decision"]["next_action"] == (
        "freeze_fixed10_heartbeat_as_stage7_reliability_boundary"
    )


def test_s7_4c_reproduces_except_elapsed() -> None:
    result = _load("result.json")
    reproduction = _load("reproduction.json")
    result.pop("elapsed_seconds")
    reproduction.pop("elapsed_seconds")
    assert result == reproduction
