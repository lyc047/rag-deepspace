import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage5_event_semantics_development import (  # noqa: E402
    Policy,
    calibration_candidate_is_eligible,
    candidate_name,
    select_candidate,
)


def test_stage5_protocol_is_development_only_and_final_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage5_event_semantics_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    governance = protocol["governance"]
    assert governance["stage4_final_measurements_may_be_loaded"] is False
    assert governance["stage4_final_metrics_may_be_loaded"] is False
    assert governance["may_modify_stage4_final_algorithm_or_claim"] is False
    assert governance["output_is_confirmatory_final"] is False
    assert "final" not in protocol["development_source"]["cache_result"].lower()
    assert protocol["resource_task"] == {
        "n_channels": 8,
        "demand_channels": 4,
        "candidate_blocks": 5,
        "query_mode": "fixed_demand_mechanism_first",
    }


def test_candidate_names_are_stable() -> None:
    assert candidate_name(0.0, 60.0) == "event_delta_tau0_age60"
    assert candidate_name(0.02, 120.0) == "event_delta_tau0p02_age120"


def test_frozen_selection_rule_filters_and_orders_candidates() -> None:
    protocol = {
        "calibration_selection_rule": {
            "minimum_actual_bit_reduction_pct": 30.0,
            "maximum_mean_regret_upper_bound_db": 0.02,
            "maximum_cvar_upper_bound_db": 0.05,
        }
    }
    good = {
        "actual_bit_reduction_pct": 35.0,
        "mean_regret_upper_bound": 0.01,
        "cvar_upper_bound": 0.04,
    }
    bad = {**good, "cvar_upper_bound": 0.051}
    assert calibration_candidate_is_eligible(protocol, good)
    assert not calibration_candidate_is_eligible(protocol, bad)
    policies = [
        Policy("a", "event_delta", 0.0, 60.0),
        Policy("b", "event_delta", 0.1, 60.0),
    ]
    comparisons = {"a": good, "b": good}
    summaries = {
        "a": {
            "mean_actual_bits": 100.0,
            "cvar_0_9_regret_db": 0.1,
            "mean_regret_db": 0.01,
            "mean_transmission_rate": 0.5,
        },
        "b": {
            "mean_actual_bits": 90.0,
            "cvar_0_9_regret_db": 0.2,
            "mean_regret_db": 0.02,
            "mean_transmission_rate": 0.4,
        },
    }
    assert select_candidate(protocol, policies, comparisons, summaries).name == "b"


def test_stage5_fixed_demand_result_meets_frozen_development_gate() -> None:
    result_path = (
        ROOT
        / "results/stage5/event_semantics_development_v1/event_semantics_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == (
        "stage5_fixed_demand_development_validation_complete"
    )
    assert result["calibration"]["selected_policy"] == {
        "name": "event_delta_tau0p2_age120",
        "regret_threshold_db": 0.2,
        "max_age_minutes": 120.0,
    }
    comparison = result["validation"][
        "comparisons_vs_absolute_index_always"
    ]["event_delta_selected"]
    assert comparison["actual_bit_reduction_pct"] >= 30.0
    assert comparison["mean_regret_upper_bound"] <= 0.02
    assert comparison["cvar_upper_bound"] <= 0.05
    governance = result["governance"]
    assert governance["stage4_final_measurements_loaded"] is False
    assert governance["stage4_final_metrics_loaded"] is False
    assert governance["confirmatory_final"] is False
