from scripts.analyze_stage8_alfa_development import gap_summary, merge_intervals


def test_merge_intervals_avoids_double_counting_processed_clips() -> None:
    assert merge_intervals([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]


def test_gap_summary_uses_robust_dynamic_threshold() -> None:
    timestamps_ns = [0, 1_000_000_000, 2_000_000_000, 20_000_000_000]
    result = gap_summary(timestamps_ns, 5.0)
    assert result["median_interval_seconds"] == 1.0
    assert result["large_gap_threshold_seconds"] == 10.0
    assert result["large_gap_count"] == 1
