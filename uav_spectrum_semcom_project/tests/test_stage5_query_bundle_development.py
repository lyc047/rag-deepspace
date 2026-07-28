import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_query_bundle_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage5_query_bundle_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["query_bundle"]["demand_order"] == [2, 4, 6]
    assert protocol["query_bundle"]["payload_bits"] == 8
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["fixed_policy_retuned"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False


def test_query_bundle_result_resolves_both_dynamic_schedules() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/query_bundle_development_v1/query_bundle_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_query_bundle_development_complete"
    for schedule in ("cyclic", "markov"):
        row = result["schedules"][schedule]
        proposed = row["comparisons_vs_query_absolute_index_always"][
            "query_event_bundle_fixed"
        ]
        vs_per_query = row["comparisons_vs_per_query_event_delta"][
            "query_event_bundle_fixed"
        ]
        assert proposed["passes_descriptive_gate"] is True
        assert proposed["actual_bit_reduction_pct"] > 45.0
        assert proposed["mean_regret_upper_bound"] <= 0.02
        assert proposed["cvar_upper_bound"] <= 0.05
        assert vs_per_query["actual_bit_reduction_pct"] > 0.0
        assert vs_per_query["mean_regret_upper_bound"] < 0.0
        assert vs_per_query["cvar_upper_bound"] < 0.0
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["fixed_policy_retuned"] is False
