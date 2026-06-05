"""评估指标：信号重建质量 + 检索性能."""
import numpy as np
from typing import List, Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差."""
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson相关系数."""
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    return float(np.dot(yt, yp) / (np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-8))


def recall_at_k(
    retrieval_results: List[Tuple[int, float]],
    ground_truth_id: int,
    k: int = 3,
) -> float:
    """Recall@K: ground truth是否在Top-K中."""
    retrieved_ids = [r[0] for r in retrieval_results[:k]]
    return 1.0 if ground_truth_id in retrieved_ids else 0.0


def bandwidth_compression_ratio(
    n_original_bits: int,
    n_transmitted_bits: int,
) -> float:
    """带宽压缩比：原始bit数 / 传输bit数."""
    return n_original_bits / max(n_transmitted_bits, 1)
