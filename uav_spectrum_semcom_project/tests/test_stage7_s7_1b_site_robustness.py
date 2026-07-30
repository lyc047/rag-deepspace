import numpy as np

from scripts.analyze_stage7_s7_1b_site_robustness import (
    _interval,
    _site_bootstrap,
)


def test_interval_returns_ordered_quantiles() -> None:
    assert np.allclose(_interval(np.arange(101), 0.8), [10.0, 90.0])


def test_site_bootstrap_is_deterministic_and_site_level() -> None:
    rows = [
        {
            "roc_auc": 0.6,
            "average_precision": 0.3,
            "positive_prevalence": 0.1,
        },
        {
            "roc_auc": 0.8,
            "average_precision": 0.5,
            "positive_prevalence": 0.2,
        },
    ]
    first = _site_bootstrap(
        rows, replicates=100, confidence=0.95, seed=7
    )
    second = _site_bootstrap(
        rows, replicates=100, confidence=0.95, seed=7
    )
    assert first == second
    assert first["unit"] == "evaluable_validation_site"
    assert first["macro_auc_interval"][0] >= 0.6
    assert first["macro_auc_interval"][1] <= 0.8
