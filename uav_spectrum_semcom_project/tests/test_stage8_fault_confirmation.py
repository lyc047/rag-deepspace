from scripts.run_stage8_fault_confirmation import (
    dispersion_statistic,
    overdispersion_monte_carlo,
)


def test_overdispersion_rejects_strong_between_flight_heterogeneity() -> None:
    rows = [
        {"internal_gap_count": 0, "internal_frame_span": 100},
        {"internal_gap_count": 0, "internal_frame_span": 100},
        {"internal_gap_count": 80, "internal_frame_span": 100},
        {"internal_gap_count": 80, "internal_frame_span": 100},
    ]
    result = overdispersion_monte_carlo(rows, replicates=2000, seed=1)
    assert result["monte_carlo_p_value"] < 0.01
    assert dispersion_statistic(rows, 0.4) > 100
