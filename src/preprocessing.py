"""信号预处理工具.

提供检索前的信号预处理功能，包括相位对齐等。
用于解决周期性信号因随机相位偏移导致的模板匹配失败。
"""
import numpy as np
from typing import Tuple


def phase_align_via_cross_correlation(
    query: np.ndarray,
    template: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """通过互相关将 template 对齐到 query.

    计算 query 和 template 的归一化互相关，找到使相关性最大化的
    滞后量，然后循环移位 template 使其与 query 相位对齐。

    算法：O(N log N) — numpy 底层用 FFT 实现 correlate。

    Args:
        query: 查询信号 shape=(n_samples,)
        template: 模板信号 shape=(n_samples,)

    Returns:
        (aligned_template, lag, peak_correlation)
        - aligned_template: 对齐后的模板，shape 与输入相同
        - lag: 滞后量（正=右移template，负=左移），单位：样本
        - peak_correlation: 峰值互相关系数 [-1, 1]，可用于质量判断
    """
    n = len(query)
    if len(template) != n:
        raise ValueError(
            f"query和template长度必须相同: {n} vs {len(template)}"
        )

    # 归一化去均值（避免直流分量干扰互相关）
    q = query - np.mean(query)
    t = template - np.mean(template)

    # 互相关（mode='full'：覆盖所有可能的滞后）
    corr = np.correlate(q, t, mode='full')

    # 归一化：除以各自的标准差
    q_std = np.std(q)
    t_std = np.std(t)
    if q_std < 1e-10 or t_std < 1e-10:
        # 信号几乎恒定，无法对齐
        return template.copy(), 0, 0.0

    corr_norm = corr / (n * q_std * t_std)

    # 最优滞后量
    # mode='full'：索引0对应 lag=-(n-1)，索引n-1对应 lag=0，索引2n-2对应 lag=n-1
    best_idx = np.argmax(corr_norm)
    lag = best_idx - (n - 1)  # 转换为样本滞后
    peak_corr = float(corr_norm[best_idx])

    # 循环移位对齐
    aligned = np.roll(template, lag)

    return aligned, int(lag), peak_corr


def phase_align_and_distance(
    query: np.ndarray,
    template: np.ndarray,
    distance_fn,
    min_correlation: float = 0.0,
):
    """相位对齐后计算距离.

    便利函数：先对齐 template 到 query，再调用 distance_fn(query, aligned)。

    Args:
        query: 查询信号
        template: 模板信号
        distance_fn: 距离函数 callable(q, t) -> float
        min_correlation: 最小相关系数阈值。若峰值互相关低于此值，
                         跳过对齐，直接用原始template

    Returns:
        (distance, aligned_template, lag, correlation)
    """
    aligned, lag, corr = phase_align_via_cross_correlation(query, template)

    if corr < min_correlation:
        # 相关性太弱，不对齐
        aligned = template
        lag = 0

    dist = distance_fn(query, aligned)
    return dist, aligned, lag, corr
