import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ack_recovery_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_ack_state_recovery_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(protocol["methods"]) == {
        "ideal_ack",
        "naive_ack",
        "epoch_guard",
        "uncertainty_recovery",
    }
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["stress_predecessor"]["regime"] == "abrupt_band_hop"
    assert protocol["ack_fault_grid"]["ack_loss_probability"] == [
        0.0,
        0.05,
        0.1,
        0.2,
    ]
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["fixed_policy_retuned"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False
    assert protocol["governance"]["ack_faults_are_measured_claims"] is False


def test_ack_recovery_result_retains_all_fault_conditions() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/ack_state_recovery_development_v1"
            / "ack_state_recovery_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_ack_state_recovery_development_complete"
    assert len(result["conditions"]) == 8
    passes = []
    for condition in result["conditions"].values():
        assert set(condition["summary"]) == {
            "ideal_ack",
            "naive_ack",
            "epoch_guard",
            "uncertainty_recovery",
        }
        recovery = condition["comparisons"][
            "uncertainty_recovery_vs_naive"
        ]
        passes.append(recovery["passes_recovery_gate"])
    assert any(passes)
    assert not all(passes)
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["stage4_final_metrics_loaded"] is False
    assert result["governance"]["fixed_policy_retuned"] is False
    assert result["governance"]["confirmatory_final"] is False
    assert result["governance"]["ack_faults_are_real_measurements"] is False
