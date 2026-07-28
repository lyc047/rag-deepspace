import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sparse_ack_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_sparse_ack_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["status"] == "frozen_before_sparse_ack_metrics"
    assert protocol["control_link"]["ack_payload_bits"] == 24
    assert protocol["control_link"]["request_payload_bits"] == 1
    assert protocol["control_link"]["piggyback_epoch_increment_bits"] == 8
    assert protocol["control_link"]["count_all_incremental_bits"] is True
    assert protocol["methods"]["request_ack_then_data"][
        "request_on_uncertainty"
    ] is True
    assert protocol["methods"]["piggyback_ack_p25"][
        "piggyback_opportunity_probability"
    ] == 0.25
    assert protocol["methods"]["piggyback_ack_p50"][
        "piggyback_opportunity_probability"
    ] == 0.5
    assert protocol["methods"]["piggyback_ack_p100"][
        "piggyback_opportunity_probability"
    ] == 1.0
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    governance = protocol["governance"]
    assert governance["stage4_final_measurements_may_be_loaded"] is False
    assert governance["stage4_final_metrics_may_be_loaded"] is False
    assert governance["fixed_policy_retuned"] is False
    assert governance["output_is_confirmatory_final"] is False
    assert governance["control_faults_are_measured_claims"] is False
    assert governance["reverse_business_traffic_is_measured"] is False


def test_sparse_ack_result_is_complete_and_preserves_conditional_boundary() -> None:
    result = json.loads(
        (
            ROOT / "results/stage5/sparse_ack_development_v1/sparse_ack_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_sparse_ack_development_complete"
    assert len(result["conditions"]) == 8
    governance = result["governance"]
    assert governance["stage4_final_measurements_loaded"] is False
    assert governance["stage4_final_metrics_loaded"] is False
    assert governance["incremental_control_bits_counted"] is True
    assert governance["reverse_business_traffic_is_measured"] is False

    candidates = (
        "request_ack_then_data",
        "piggyback_ack_p25",
        "piggyback_ack_p50",
        "piggyback_ack_p100",
    )
    pass_counts = {method: 0 for method in candidates}
    for name, condition in result["conditions"].items():
        assert condition["control_costs"] == {
            "ack_response_bits": 98,
            "ack_request_bits": 70,
            "piggyback_increment_bits": 14,
        }
        for method in candidates:
            comparison = condition["comparisons"][
                f"{method}_vs_immediate_data_recovery"
            ]
            pass_counts[method] += int(comparison["passes_sparse_ack_gate"])
            if name.startswith("control_loss_0.2_") and method.startswith(
                "piggyback"
            ):
                assert comparison["passes_sparse_ack_gate"] is True

    assert pass_counts == {
        "request_ack_then_data": 0,
        "piggyback_ack_p25": 2,
        "piggyback_ack_p50": 2,
        "piggyback_ack_p100": 2,
    }
