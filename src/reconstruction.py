"""信号重建模块：模板匹配 + 残差传输."""
import numpy as np
from typing import List, Tuple
from knowledge import KnowledgeBase


def reconstruct_from_template(
    y_query: np.ndarray,
    retrieval_results: List[Tuple[int, float]],
    kb: KnowledgeBase,
) -> np.ndarray:
    """基于检索到的最佳模板重建信号.

    Phase 1 (MVP): 直接用最佳模板作为重建结果（残差=0）.
    Phase 2: 模板 + 残差.

    Args:
        y_query: 原始查询信号（用于选择最佳模板）
        retrieval_results: [(template_id, distance), ...]
        kb: 知识库

    Returns:
        重建信号
    """
    if len(retrieval_results) == 0:
        return np.zeros_like(y_query)

    best_id = retrieval_results[0][0]
    record = kb.get_record(best_id)
    if record is None:
        return np.zeros_like(y_query)

    return record.raw_data.copy()


def compute_residual(
    y_original: np.ndarray,
    y_template: np.ndarray,
) -> np.ndarray:
    """计算原始信号与模板的残差."""
    return y_original - y_template
