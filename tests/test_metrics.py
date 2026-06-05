# tests/test_metrics.py
import numpy as np
import sys
sys.path.insert(0, 'src')
from metrics import mse, mae, pearson_corr, recall_at_k


def test_mse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_error():
    assert mse(np.array([1.0]), np.array([3.0])) == 4.0


def test_mae():
    assert mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 1.0


def test_pearson_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(pearson_corr(y, y) - 1.0) < 1e-6


def test_recall_at_k_hit():
    results = [(1, 0.1), (5, 0.3), (3, 0.5)]
    assert recall_at_k(results, ground_truth_id=1, k=3) == 1.0


def test_recall_at_k_miss():
    results = [(1, 0.1), (5, 0.3), (3, 0.5)]
    assert recall_at_k(results, ground_truth_id=2, k=3) == 0.0
