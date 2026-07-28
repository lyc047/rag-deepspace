from spectrum_semcom.resource_task_audit import select_non_saturated_configuration


def row(split: str, gap: float, ambiguous: float, clean: float, channels: int = 8) -> dict:
    return {
        "split_role": split,
        "sources_per_scene": 2,
        "n_channels": channels,
        "demand_channels": 1,
        "oracle_clean_rate": clean,
        "mean_random_oracle_gap": gap,
        "mean_oracle_margin": gap / 2,
        "ambiguous_oracle_rate": ambiguous,
    }


def test_selection_uses_validation_and_frozen_non_saturation_filters() -> None:
    rows = [
        row("train", 0.9, 0.0, 0.5),
        row("validation", 0.01, 0.0, 0.5),
        row("validation", 0.4, 0.9, 0.5),
        row("validation", 0.3, 0.1, 0.5, channels=8),
        row("validation", 0.2, 0.1, 0.5, channels=4),
    ]
    selected = select_non_saturated_configuration(
        rows,
        selection_split="validation",
        minimum_mean_random_oracle_gap=0.02,
        maximum_ambiguous_oracle_rate=0.8,
        oracle_clean_rate_range=(0.2, 0.98),
    )
    assert selected is not None
    assert selected["mean_random_oracle_gap"] == 0.3
    assert selected["n_channels"] == 8


def test_selection_returns_none_when_task_is_saturated() -> None:
    selected = select_non_saturated_configuration(
        [row("validation", 0.0, 1.0, 1.0)],
        selection_split="validation",
        minimum_mean_random_oracle_gap=0.02,
        maximum_ambiguous_oracle_rate=0.8,
        oracle_clean_rate_range=(0.2, 0.98),
    )
    assert selected is None
