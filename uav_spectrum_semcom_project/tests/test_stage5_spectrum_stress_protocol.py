import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spectrum_stress_protocol_is_frozen_and_final_closed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_spectrum_stress_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(protocol["spectrum_regimes"]) == {
        "observed",
        "gradual_drift",
        "abrupt_band_hop",
        "burst_block_interference",
    }
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["fixed_policy_retuned"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False
    assert protocol["governance"]["stress_trajectories_are_measured_claims"] is False


def test_spectrum_stress_result_keeps_all_regimes_and_negative_outcomes() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/spectrum_stress_development_v1"
            / "spectrum_stress_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_spectrum_stress_development_complete"
    assert set(result["regimes"]) == {
        "observed",
        "gradual_drift",
        "abrupt_band_hop",
        "burst_block_interference",
    }
    for schedules in result["regimes"].values():
        assert set(schedules) == {"cyclic", "markov"}
        for row in schedules.values():
            proposed = row["comparisons_vs_query_absolute_index_always"][
                "query_event_bundle_fixed"
            ]
            assert "passes_descriptive_gate" in proposed
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["stage4_final_metrics_loaded"] is False
    assert result["governance"]["fixed_policy_retuned"] is False
    assert result["governance"]["confirmatory_final"] is False
    assert result["governance"]["stress_trajectories_are_new_measurements"] is False
