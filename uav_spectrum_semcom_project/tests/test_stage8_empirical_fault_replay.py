import numpy as np

from scripts.run_stage8_empirical_fault_replay import (
    heartbeat_key,
    paired_group_summary,
)


def test_heartbeat_key_is_stable_for_no_heartbeat() -> None:
    assert heartbeat_key(None) == "none"
    assert heartbeat_key(10) == "10"


def test_paired_summary_uses_candidate_minus_baseline_clean() -> None:
    candidate = {
        "clean_rate": np.array([0.90, 0.92]),
        "availability_rate": np.array([0.95, 0.96]),
        "total_bits_per_scene": np.array([8.0, 8.0]),
    }
    baseline = {
        "clean_rate": np.array([0.89, 0.91]),
        "availability_rate": np.array([0.94, 0.95]),
        "total_bits_per_scene": np.array([10.0, 10.0]),
    }
    config = {
        "monte_carlo": {
            "bootstrap_replicates": 100,
            "confidence_level": 0.95,
        }
    }
    result = paired_group_summary(
        [candidate], [baseline], [100], seed=1, config=config
    )
    assert abs(result["clean_difference_percentage_points"]["mean"] - 1.0) < 1e-9
    assert (
        abs(result["relative_total_bit_saving_percentage"]["mean"] - 20.0)
        < 1e-9
    )
