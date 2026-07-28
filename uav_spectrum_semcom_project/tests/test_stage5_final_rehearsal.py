import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_rehearsal_reproduces_frozen_results_without_final_data() -> None:
    primary_rehearsal = (
        "results/stage5/final_rehearsal_v1/primary/query_bundle_result.json"
    )
    primary_frozen = (
        "results/stage5/query_bundle_development_v1/query_bundle_result.json"
    )
    robustness_rehearsal = (
        "results/stage5/final_rehearsal_v1/robustness/"
        "ack_state_recovery_result.json"
    )
    robustness_frozen = (
        "results/stage5/ack_state_recovery_development_v1/"
        "ack_state_recovery_result.json"
    )

    assert _sha256(primary_rehearsal) == _sha256(primary_frozen)
    assert _sha256(robustness_rehearsal) == _sha256(robustness_frozen)

    for result_path in (primary_rehearsal, robustness_rehearsal):
        governance = _load(result_path)["governance"]
        assert governance["stage4_final_measurements_loaded"] is False
        assert governance["stage4_final_metrics_loaded"] is False
        assert governance["fixed_policy_retuned"] is False
        assert governance["confirmatory_final"] is False


def test_rehearsal_preserves_expected_gate_boundary() -> None:
    primary = _load(
        "results/stage5/final_rehearsal_v1/primary/query_bundle_result.json"
    )
    robustness = _load(
        "results/stage5/final_rehearsal_v1/robustness/"
        "ack_state_recovery_result.json"
    )

    primary_passes = [
        row["comparisons_vs_query_absolute_index_always"][
            "query_event_bundle_fixed"
        ]["passes_descriptive_gate"]
        for row in primary["schedules"].values()
    ]
    recovery_passes = [
        row["comparisons"]["uncertainty_recovery_vs_naive"][
            "passes_recovery_gate"
        ]
        for row in robustness["conditions"].values()
    ]

    assert primary_passes == [True, True]
    assert sum(recovery_passes) == 4
    assert len(recovery_passes) == 8


def test_historical_final_access_counts_are_not_incremented() -> None:
    state_files = (
        "configs/stage4_c1_temporal_access_state.json",
        "configs/stage4_c1_v4_final_access_state.json",
        "configs/stage4_final_access_state.json",
    )

    for state_file in state_files:
        state = _load(state_file)
        assert state["status"] == "access_consumed"
        assert state["access_count"] == 1
