import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage4_c1_v4_scalability_experiment import (  # noqa: E402
    method_application_bits,
    task_id,
)


def test_scalability_protocol_is_development_only() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage4_c1_v4_scalability_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["governance"]["final_measurements_or_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["may_modify_final_algorithm_or_claim"] is False
    assert protocol["governance"]["compact_binary_index_is_confirmed"] is False
    assert "final" not in protocol["development_source"]["cache_result"].lower()
    assert "final" not in protocol["development_source"]["provenance"].lower()
    assert protocol["channel_counts"] == [8, 16, 32, 64]
    assert protocol["demand_ratios"] == [0.25, 0.5, 0.75]


def test_scalability_payload_formulas() -> None:
    overhead = 152
    for n_channels in (8, 16, 32, 64):
        for ratio in (0.25, 0.5, 0.75):
            demand = int(n_channels * ratio)
            candidates = n_channels - demand + 1
            assert method_application_bits(
                "soft_power_fixed4", n_channels, demand, overhead
            ) == overhead + 4 * n_channels
            assert method_application_bits(
                "best_block_onehot", n_channels, demand, overhead
            ) == overhead + candidates
            assert method_application_bits(
                "best_block_binary_index", n_channels, demand, overhead
            ) == overhead + math.ceil(math.log2(candidates))
            assert task_id(n_channels, demand) == f"N{n_channels}_D{demand}"


def test_scalability_result_meets_frozen_development_claim() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage4/c1_v4_scalability_v1/scalability_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["inputs"]["scene_count"] == 180
    assert result["inputs"]["cluster_count"] == 72
    assert len(result["tasks"]) == 12
    assert result["strong_target_task_count"]["best_block_onehot"] == 6
    assert result["strong_target_task_count"]["best_block_binary_index"] == 6
    assert result["governance"]["final_measurement_values_loaded"] is False
    assert result["governance"]["final_metrics_loaded"] is False
    assert result["governance"]["confirmatory"] is False
    for task in result["tasks"]:
        comparisons = result["comparisons_vs_soft_power_fixed4"][task["task_id"]]
        assert comparisons["best_block_onehot"]["mean_regret_upper_bound"] < 0
        assert comparisons["best_block_binary_index"]["mean_regret_upper_bound"] < 0
    n64 = result["comparisons_vs_soft_power_fixed4"]["N64_D32"]
    assert n64["best_block_onehot"]["actual_bit_reduction_pct"] > 40
    assert n64["best_block_binary_index"]["actual_bit_reduction_pct"] > 49
