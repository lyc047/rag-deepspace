from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectrum_semcom.stage6_frozen_risk_ranker import (
    fit_frozen_risk_ranking,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_frozen_risk_candidate_sets_are_nested() -> None:
    cache_path = (
        PROJECT
        / "results/stage5/helikite_pilot_v1/helikite_pilot_cache.npz"
    )
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    groups = np.asarray(cache["cluster_ids"]).astype(str)
    unique_groups = sorted(set(groups.tolist()))
    train_groups = unique_groups[:4]
    evaluation_groups = unique_groups[4:]
    training = np.isin(groups, np.asarray(train_groups))
    queries = tuple(SpectrumTaskQuery(value) for value in (2, 4, 6))
    states = [
        build_task_state(values, queries, epsilon_db=0.2)
        for values in cache["channel_power_n8"]
    ]
    codebook = fit_greedy_task_codebook(
        [state for state, selected in zip(states, training) if selected]
    )
    selection = json.loads(
        (
            PROJECT
            / (
                "results/stage6/temporal_hazard_development_v1/"
                "temporal_hazard_result.json"
            )
        ).read_text(encoding="utf-8")
    )
    value = fit_frozen_risk_ranking(
        cache["channel_power_n8"],
        states,
        cache["timestamps_local"],
        cache["cluster_ids"],
        codebook,
        train_groups=train_groups,
        evaluation_groups=evaluation_groups,
        selected_c=float(selection["results"]["8"]["selected_c"]),
        reservation_fractions=(0.0, 0.05, 0.1, 0.2),
    )
    c0 = value.candidates_by_fraction[0.0]
    c5 = value.candidates_by_fraction[0.05]
    c10 = value.candidates_by_fraction[0.1]
    c20 = value.candidates_by_fraction[0.2]
    assert not np.any(c0)
    assert not np.any(c5 & ~c10)
    assert not np.any(c10 & ~c20)
    assert value.thresholds_by_fraction[0.0] is None
