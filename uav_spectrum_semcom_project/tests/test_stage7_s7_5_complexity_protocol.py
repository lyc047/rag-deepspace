import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_s7_5a_online_codec_passes_but_unbounded_fit_fails() -> None:
    result = _load(
        "results/stage7/s7_5a_complexity_benchmark_v1/result.json"
    )
    reproduction = _load(
        "results/stage7/s7_5a_complexity_benchmark_v1/reproduction.json"
    )
    assert not result["decision"]["development_gate_passed"]
    assert not reproduction["decision"]["development_gate_passed"]
    for n in ("8", "16", "32", "64"):
        assert result["n_results"][n]["gates"]["latency_passed"]
        assert result["n_results"][n]["gates"]["memory_passed"]
        assert reproduction["n_results"][n]["gates"]["latency_passed"]
        assert reproduction["n_results"][n]["gates"]["memory_passed"]
    assert not result["n_results"]["32"]["gates"]["fit_passed"]
    assert not result["n_results"]["64"]["gates"]["fit_passed"]


def test_s7_5b_bounded_k3_fit_passes_both_runs() -> None:
    result = _load(
        "results/stage7/s7_5b_offline_fit_scaling_v1/result.json"
    )
    reproduction = _load(
        "results/stage7/s7_5b_offline_fit_scaling_v1/reproduction.json"
    )
    assert result["decision"]["gate_passed"]
    assert reproduction["decision"]["gate_passed"]
    assert result["decision"]["passing_n_count"] == 4
    assert reproduction["decision"]["passing_n_count"] == 4
    assert all(row["n_gate_passed"] for row in result["n_checks"].values())
    assert all(
        row["n_gate_passed"] for row in reproduction["n_checks"].values()
    )


def test_s7_5_session_start_amortization_is_monotone() -> None:
    result = _load(
        "results/stage7/s7_5a_complexity_benchmark_v1/result.json"
    )
    for row in result["n_results"].values():
        values = row["session_start_amortized_bits_per_scene"]
        assert values["10"] == 8.8
        assert values["10000"] == 0.0088
        assert values["10"] > values["100"] > values["1000"] > values["10000"]
