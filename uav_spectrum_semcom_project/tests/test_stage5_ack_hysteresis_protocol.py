import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ack_hysteresis_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_ack_hysteresis_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(protocol["methods"]) == {
        "ideal_ack",
        "epoch_guard",
        "immediate_recovery",
        "hysteresis2",
        "hysteresis2_age80",
    }
    assert (
        protocol["methods"]["hysteresis2"]["force_after_unconfirmed_count"]
        == 2
    )
    assert protocol["methods"]["hysteresis2_age80"]["age_guard_fraction"] == 0.8
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["fixed_policy_retuned"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False
    assert protocol["governance"]["ack_faults_are_measured_claims"] is False


def test_ack_hysteresis_result_preserves_failed_fault_conditions() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/ack_hysteresis_development_v1"
            / "ack_hysteresis_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_ack_hysteresis_development_complete"
    assert len(result["conditions"]) == 8
    for method in ("hysteresis2", "hysteresis2_age80"):
        passes = [
            condition["comparisons"][
                f"{method}_vs_immediate_recovery"
            ]["passes_hysteresis_gate"]
            for condition in result["conditions"].values()
        ]
        assert sum(passes) == 2
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["stage4_final_metrics_loaded"] is False
    assert result["governance"]["fixed_policy_retuned"] is False
    assert result["governance"]["confirmatory_final"] is False
    assert result["governance"]["ack_faults_are_real_measurements"] is False
