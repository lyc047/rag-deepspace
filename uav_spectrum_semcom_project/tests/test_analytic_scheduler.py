import numpy as np

from spectrum_semcom.analytic_scheduler import CandidateAction, analytic_raw_score, exact_multiple_choice_knapsack, fit_monotonic_value_calibrator, greedy_actions, train_margin_temperature


def test_analytic_score_and_train_temperature_are_finite() -> None:
    beliefs = np.asarray([[0.1, 0.2, 0.8, 0.9], [0.2, 0.2, 0.7, 0.8]])
    temperature = train_margin_temperature(beliefs, 2)
    quality = np.asarray([6, .8, .2, 0, 0, 0, 10, .1, .9, 0])
    score = analytic_raw_score(beliefs[0], np.asarray([0, 0, 1, 1]), quality, 2, 2, temperature)
    assert temperature > 0 and np.isfinite(score) and score >= 0


def test_monotonic_calibrator_and_discrete_solvers_allow_silence() -> None:
    calibrator = fit_monotonic_value_calibrator(np.arange(8), np.asarray([-1, -1, 0, 0, 1, .5, 2, 2]), 4)
    assert np.all(np.diff(calibrator.value_knots) >= -1e-12)
    candidates = [CandidateAction(0, 2, 1.0, 5.0), CandidateAction(0, 3, 2.0, 9.0), CandidateAction(1, 2, -1.0, 2.0)]
    assert greedy_actions(candidates, 6, 1)[0].target_granularity == 2
    assert exact_multiple_choice_knapsack(candidates, 10, 1)[0].target_granularity == 3
    assert exact_multiple_choice_knapsack(candidates, 4, 1) == ()


def test_constant_scores_calibrate_to_global_mean_without_fake_order() -> None:
    calibrator = fit_monotonic_value_calibrator(np.zeros(4), np.asarray([-1.0, 0.0, 1.0, 2.0]), 4)
    assert np.allclose(calibrator.value_knots, 0.5)
