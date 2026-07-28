import numpy as np
import pytest

from spectrum_semcom.ranking_metrics import average_ranks, spearman_correlation, top_k_recall


def test_average_ranks_handles_ties() -> None:
    assert np.allclose(average_ranks(np.asarray([1, 1, 3])), [.5, .5, 2])


def test_spearman_and_top_k_recall() -> None:
    assert spearman_correlation(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    prediction = np.asarray([[3, 2, 1], [1, 2, 3]])
    target = np.asarray([[3, 2, 1], [2, 3, 1]])
    assert top_k_recall(prediction, target, 1) == 0.5
    assert top_k_recall(prediction, target, 2) == 1.0
