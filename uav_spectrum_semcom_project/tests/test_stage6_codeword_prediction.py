from __future__ import annotations

import numpy as np

from spectrum_semcom.stage6_codeword_prediction import (
    build_next_codeword_datasets,
    fit_conditional_transition_destination,
    fit_first_order_markov,
    fit_frequency_prior,
    predict_first_order_markov,
)
from spectrum_semcom.stage6_task_codebook import (
    SpectrumTaskQuery,
    build_task_state,
    fit_greedy_task_codebook,
)
from spectrum_semcom.stage6_markov_scheduler import (
    codeword_run_lengths,
    fit_markov_scheduler,
    online_token_bucket_candidates,
    score_markov_scheduler,
)


def _fixture():
    powers = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.1, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 0.9, 0.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.9, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.1],
        ],
        dtype=np.float64,
    )
    queries = (SpectrumTaskQuery(2),)
    states = [
        build_task_state(row, queries, epsilon_db=0.2)
        for row in powers
    ]
    codebook = fit_greedy_task_codebook(states)
    timestamps = np.asarray(
        [f"2026-01-01T00:{index:02d}:00" for index in range(8)]
    )
    groups = np.asarray(["a"] * 4 + ["b"] * 4)
    return powers, states, timestamps, groups, codebook


def test_next_codeword_windows_are_causal_aligned_and_group_safe() -> None:
    powers, states, timestamps, groups, codebook = _fixture()
    datasets = build_next_codeword_datasets(
        powers,
        states,
        timestamps,
        groups,
        codebook,
        history_lengths=(1, 2),
    )
    np.testing.assert_array_equal(
        datasets["source_indices"], np.asarray([1, 2, 5, 6])
    )
    assert datasets["features_by_window"]["1"].shape[0] == 4
    assert datasets["features_by_window"]["2"].shape[0] == 4
    assert datasets["groups"].tolist() == ["a", "a", "b", "b"]

    one = datasets["features_by_window"]["1"].copy()
    two = datasets["features_by_window"]["2"].copy()
    changed = powers.copy()
    changed[datasets["source_indices"] + 1] += 1000.0
    changed_datasets = build_next_codeword_datasets(
        changed,
        states,
        timestamps,
        groups,
        codebook,
        history_lengths=(1, 2),
    )
    # Row zero forecasts index 2 from information through index 1. Altering
    # index 2 must therefore change its label, never its feature vector.
    np.testing.assert_allclose(
        changed_datasets["features_by_window"]["1"][0], one[0]
    )
    np.testing.assert_allclose(
        changed_datasets["features_by_window"]["2"][0], two[0]
    )


def test_frequency_and_markov_baselines_are_deterministic() -> None:
    current = np.asarray([0, 0, 0, 1, 1, 2])
    following = np.asarray([1, 1, 0, 2, 2, 2])
    assert fit_frequency_prior(following, class_count=3) == 2
    table = fit_first_order_markov(
        current,
        following,
        class_count=3,
        laplace_alpha=1.0,
    )
    np.testing.assert_array_equal(table, np.asarray([1, 2, 2]))
    np.testing.assert_array_equal(
        predict_first_order_markov(table, np.asarray([2, 0, 1])),
        np.asarray([2, 1, 2]),
    )
    destination = fit_conditional_transition_destination(
        current,
        following,
        class_count=3,
        laplace_alpha=0.1,
    )
    np.testing.assert_array_equal(destination, np.asarray([1, 2, 1]))


def test_next_codeword_protocol_keeps_final_closed() -> None:
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project
            / "configs/stage6_next_codeword_prediction_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["history_windows"] == [1, 2, 4, 8]
    assert not any(protocol["governance"].values())
    assert protocol["safety_boundary"][
        "prediction_may_not_execute_or_select_a_future_spectrum_action"
    ]


def test_selective_threshold_uses_confidence_and_abstains() -> None:
    from scripts.run_stage6_selective_codeword_prediction_development import (
        _select_threshold,
    )

    selected = _select_threshold(
        np.asarray([0.1, 0.9, 0.2, 0.8]),
        np.asarray([1, 1, 0, 0]),
        np.asarray([0, 1, 1, 0]),
        np.asarray([0, 0, 1, 1]),
        class_count=2,
        config={
            "threshold_grid_start": 0.5,
            "threshold_grid_stop": 0.5,
            "threshold_grid_step": 0.01,
            "maximum_inner_false_switch_rate": 0.0,
            "minimum_inner_transition_recall": 1.0,
        },
    )
    assert selected["feasible"]
    assert selected["threshold"] == 0.5
    assert selected["metrics"]["exact_class_accuracy"] == 1.0


def test_selective_protocol_uses_nested_groups_and_keeps_final_closed() -> None:
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project
            / "configs/stage6_selective_codeword_prediction_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["validation"]["scheme"] == "nested_leave_one_cluster_out"
    assert protocol["validation"]["history_length"] == 8
    assert not any(protocol["governance"].values())
    assert protocol["safety_boundary"][
        "low_confidence_prediction_must_abstain_to_current_codeword"
    ]

    corrected = json.loads(
        (
            project
            / "configs/stage6_selective_codeword_prediction_development_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert len(corrected["fixed_codebook"]["fit_groups"]) == 4
    assert len(corrected["validation"]["outer_evaluation_groups"]) == 3
    assert not (
        set(corrected["fixed_codebook"]["fit_groups"])
        & set(corrected["validation"]["outer_evaluation_groups"])
    )


def test_markov_scheduler_probabilities_are_causal_and_normalized() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 0, 0, 1])
    groups = np.asarray(["a"] * 5 + ["b"] * 3)
    np.testing.assert_array_equal(
        codeword_run_lengths(labels, groups),
        np.asarray([1, 2, 1, 2, 3, 1, 2, 1]),
    )
    model = fit_markov_scheduler(
        labels,
        groups,
        training_groups=("a",),
        class_count=2,
        duration_bin_lower_bounds=(1, 2, 4),
        laplace_alpha=1.0,
        duration_shrinkage_strength=5.0,
        entropy_weight=0.25,
    )
    np.testing.assert_allclose(
        np.sum(model.first_order_probabilities, axis=1), 1.0
    )
    np.testing.assert_allclose(
        np.sum(model.duration_probabilities, axis=2), 1.0
    )
    scores = score_markov_scheduler(model, labels, groups)
    for name in (
        "first_order_change_probability",
        "duration_change_probability",
        "duration_normalized_entropy",
    ):
        assert np.all((scores[name] >= 0.0) & (scores[name] <= 1.0))

    changed_future = labels.copy()
    changed_future[4] = 0
    changed_scores = score_markov_scheduler(
        model, changed_future, groups
    )
    # Scores through index 3 use no information from the altered index 4.
    np.testing.assert_allclose(
        changed_scores["duration_entropy_priority"][:4],
        scores["duration_entropy_priority"][:4],
    )


def test_markov_scheduler_protocol_keeps_prediction_out_of_action_loop() -> None:
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project
            / "configs/stage6_markov_probability_scheduler_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["controller"][
        "maximum_update_reservation_fraction_of_evaluation_scenes"
    ] == 0.1
    assert not protocol["governance"][
        "external_final_signal_values_may_be_loaded"
    ]
    assert protocol["safety_boundary"][
        "markov_probability_may_not_select_or_execute_a_future_spectrum_action"
    ]


def test_token_bucket_scheduler_is_causal_and_spreads_candidates() -> None:
    scores = np.ones(12, dtype=np.float64)
    indices = np.arange(12, dtype=np.int64)
    groups = np.asarray(["a"] * 6 + ["b"] * 6)
    candidates = online_token_bucket_candidates(
        scores,
        indices,
        groups,
        threshold=0.5,
        state_count=12,
        token_accrual_per_scene=0.25,
        bucket_capacity=1.0,
        initial_tokens=1.0,
        reset_at_group_boundary=True,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(candidates), np.asarray([0, 4, 6, 10])
    )
    altered_future = scores.copy()
    altered_future[8:] = 0.0
    altered = online_token_bucket_candidates(
        altered_future,
        indices,
        groups,
        threshold=0.5,
        state_count=12,
        token_accrual_per_scene=0.25,
        bucket_capacity=1.0,
        initial_tokens=1.0,
        reset_at_group_boundary=True,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(altered)[:3], np.asarray([0, 4, 6])
    )


def test_token_bucket_protocol_is_terminal_and_final_closed() -> None:
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            project
            / "configs/stage6_markov_token_bucket_scheduler_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["token_bucket"]["token_accrual_per_scene"] == 0.1
    assert not protocol["token_bucket"][
        "future_scores_or_future_candidate_counts_may_be_used"
    ]
    assert "terminate_markov_controller_research" in protocol[
        "retention_rule"
    ]["failure_action"]
    assert not protocol["governance"][
        "external_final_access_may_be_consumed"
    ]


def test_candidate_architecture_freeze_forbids_posthoc_tuning() -> None:
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            project
            / "configs/stage6_candidate_architecture_freeze_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert not manifest["freeze_governance"][
        "algorithm_modules_may_be_added_after_freeze"
    ]
    assert not manifest["freeze_governance"][
        "existing_parameters_may_be_tuned_after_freeze"
    ]
    assert manifest["freeze_governance"][
        "defect_correction_requires_new_freeze_version"
    ]
    assert not (
        set(manifest["retained_components"])
        & set(manifest["excluded_components"])
    )
