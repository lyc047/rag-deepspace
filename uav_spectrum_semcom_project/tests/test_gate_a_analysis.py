import numpy as np
import pytest

from spectrum_semcom.gate_a_analysis import hierarchical_paired_bootstrap


def test_hierarchical_bootstrap_detects_consistent_paired_improvement() -> None:
    baseline = np.tile(np.linspace(0.1, 0.5, 30), (5, 1))
    proposed = baseline - 0.05
    result = hierarchical_paired_bootstrap(proposed, baseline, repetitions=500, seed=1)
    assert result["difference"] == pytest.approx(-0.05)
    assert result["ci95_high"] < 0


def test_hierarchical_bootstrap_supports_tail_statistic() -> None:
    baseline = np.tile(np.asarray([0.0, 0.0, 1.0, 2.0]), (3, 1))
    proposed = baseline * 0.5
    result = hierarchical_paired_bootstrap(proposed, baseline, statistic="cvar", cvar_alpha=0.5, repetitions=500)
    assert result["difference"] < 0
