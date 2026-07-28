import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage5_dynamic_query_development import build_query_schedule  # noqa: E402


def test_dynamic_query_protocol_is_fixed_and_final_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage5_dynamic_query_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["fixed_policy_from_s5_1"]["parameters_may_be_retuned"] is False
    assert protocol["resource_queries"]["demand_channels"] == [2, 4, 6]
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False
    assert "final" not in protocol["development_source"]["cache_result"].lower()


def test_query_schedules_are_deterministic_and_valid() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage5_dynamic_query_development_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cache = {
        "scene_ids": np.asarray([f"s{i}" for i in range(12)]),
        "site_ids": np.asarray(["A"] * 6 + ["B"] * 6),
        "timestamps_local": np.asarray(
            [f"2022-01-01T00:{i:02d}:00" for i in range(6)] * 2
        ),
    }
    cyclic = build_query_schedule(cache, protocol, "cyclic")
    np.testing.assert_array_equal(cyclic[:6], [2, 4, 6, 2, 4, 6])
    np.testing.assert_array_equal(cyclic[6:], [4, 6, 2, 4, 6, 2])
    markov_a = build_query_schedule(cache, protocol, "markov")
    markov_b = build_query_schedule(cache, protocol, "markov")
    np.testing.assert_array_equal(markov_a, markov_b)
    assert set(markov_a).issubset({2, 4, 6})


def test_dynamic_query_result_preserves_positive_and_negative_schedules() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/dynamic_query_development_v1/dynamic_query_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_dynamic_query_development_complete"
    cyclic = result["schedules"]["cyclic"][
        "comparisons_vs_query_absolute_index_always"
    ]["query_event_delta_fixed"]
    markov = result["schedules"]["markov"][
        "comparisons_vs_query_absolute_index_always"
    ]["query_event_delta_fixed"]
    assert cyclic["passes_descriptive_gate"] is False
    assert markov["passes_descriptive_gate"] is True
    assert cyclic["actual_bit_reduction_pct"] < 30.0
    assert markov["actual_bit_reduction_pct"] > 30.0
    assert result["governance"]["fixed_policy_retuned"] is False
    assert result["governance"]["stage4_final_measurements_loaded"] is False
