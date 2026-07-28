import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cumulative_ack_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_cumulative_ack_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["ack_control"]["payload_bits"] == 24
    assert protocol["ack_control"]["count_all_feedback_bits"] is True
    assert protocol["methods"]["cumack1_then_data"]["cumulative_ack_repeats"] == 1
    assert protocol["methods"]["cumack2_then_data"]["cumulative_ack_repeats"] == 2
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["fixed_policy_retuned"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False
    assert protocol["governance"]["ack_faults_are_measured_claims"] is False


def test_cumulative_ack_result_is_complete_and_retains_negative_result() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/cumulative_ack_development_v1/cumulative_ack_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_cumulative_ack_development_complete"
    assert len(result["conditions"]) == 8
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["stage4_final_metrics_loaded"] is False
    assert result["governance"]["feedback_bits_counted"] is True

    expected_methods = {
        "ideal_ack",
        "epoch_guard_no_recovery",
        "immediate_data_recovery",
        "cumack1_then_data",
        "cumack2_then_data",
    }
    pass_counts = {
        "cumack1_then_data": 0,
        "cumack2_then_data": 0,
    }
    for condition in result["conditions"].values():
        assert condition["ack_channel_bits_per_feedback_frame"] == 98
        assert set(condition["summary"]) == expected_methods
        for method in pass_counts:
            comparison = condition["comparisons"][
                f"{method}_vs_immediate_data_recovery"
            ]
            pass_counts[method] += int(
                comparison["passes_cumulative_ack_gate"]
            )
            assert comparison["actual_bit_reduction_pct"] < 0.0

    assert pass_counts == {
        "cumack1_then_data": 0,
        "cumack2_then_data": 0,
    }
