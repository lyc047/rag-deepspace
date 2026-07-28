"""Selection rules for a non-saturated Gate A development task."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def select_non_saturated_configuration(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_split: str,
    minimum_mean_random_oracle_gap: float,
    maximum_ambiguous_oracle_rate: float,
    oracle_clean_rate_range: tuple[float, float],
) -> dict[str, Any] | None:
    """Apply preregistered truth-only difficulty filters on validation rows."""

    low_clean, high_clean = oracle_clean_rate_range
    if not 0.0 <= low_clean < high_clean <= 1.0:
        raise ValueError("oracle_clean_rate_range must be ordered within [0, 1]")
    eligible: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("split_role") != selection_split:
            continue
        if float(row["mean_random_oracle_gap"]) < minimum_mean_random_oracle_gap:
            continue
        if float(row["ambiguous_oracle_rate"]) > maximum_ambiguous_oracle_rate:
            continue
        if not low_clean <= float(row["oracle_clean_rate"]) <= high_clean:
            continue
        eligible.append(row)
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_random_oracle_gap"]),
            float(row["mean_oracle_margin"]),
            -float(row["ambiguous_oracle_rate"]),
            -int(row["n_channels"]),
            -int(row["sources_per_scene"]),
        ),
    )
    keys = (
        "sources_per_scene",
        "n_channels",
        "demand_channels",
        "oracle_clean_rate",
        "mean_random_oracle_gap",
        "mean_oracle_margin",
        "ambiguous_oracle_rate",
    )
    return {key: selected[key] for key in keys}
