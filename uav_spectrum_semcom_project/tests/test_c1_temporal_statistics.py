import numpy as np
import pytest

from spectrum_semcom.c1_temporal_statistics import empirical_cvar_numpy, paired_cluster_bootstrap


def test_empirical_cvar_numpy_uses_worst_fraction() -> None:
    assert empirical_cvar_numpy(np.array([0.0, 1.0, 2.0, 9.0]), 0.5) == pytest.approx(5.5)


def test_paired_cluster_bootstrap_is_reproducible_and_paired() -> None:
    baseline = np.array([3.0, 2.0, 4.0, 3.0, 5.0, 4.0])
    proposed = baseline - 1.0
    clusters = ["a", "a", "b", "b", "c", "c"]
    left = paired_cluster_bootstrap(proposed, baseline, np.ones(6) * 100, np.ones(6) * 100, clusters, repetitions=500, seed=7, cvar_alpha=0.5)
    right = paired_cluster_bootstrap(proposed, baseline, np.ones(6) * 100, np.ones(6) * 100, clusters, repetitions=500, seed=7, cvar_alpha=0.5)
    assert left == right
    assert left["mean_regret_difference"] == pytest.approx(-1.0)
    assert left["mean_regret_upper_bound"] < 0
    assert left["actual_bit_difference"] == 0


def test_cluster_bootstrap_rejects_single_cluster() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_cluster_bootstrap(np.zeros(2), np.ones(2), np.ones(2), np.ones(2), ["a", "a"], repetitions=100, seed=1)
