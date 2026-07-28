import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_helikite_scale_protocol_is_final_closed_and_fixed() -> None:
    protocol = json.loads(
        (
            ROOT / "configs/stage5_helikite_scale_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["task_grid"]["n_channels"] == [8, 16, 32, 64]
    assert protocol["task_grid"]["demand_ratios"] == [0.25, 0.5, 0.75]
    assert protocol["fixed_policy"]["regret_threshold_db"] == 0.2
    assert protocol["fixed_policy"]["parameters_may_be_retuned"] is False
    assert protocol["governance"]["external_final_archives_may_be_opened"] is False
    assert (
        protocol["governance"]["external_final_signal_values_may_be_loaded"]
        is False
    )


def test_helikite_scale_result_has_all_n_and_passes_checks() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/helikite_scale_development_v1"
            / "helikite_scale_result.json"
        ).read_text(encoding="utf-8")
    )
    assert set(result["n_results"]) == {"8", "16", "32", "64"}
    assert all(result["descriptive_checks"].values())
    assert result["inputs"]["scene_count"] == 364
    assert result["governance"]["external_final_signal_values_loaded"] is False
    assert result["governance"]["final_access_consumed"] is False
