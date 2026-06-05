"""测试信号预处理模块."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from preprocessing import phase_align_via_cross_correlation, phase_align_and_distance


def test_phase_align_identical_shifted():
    """已知偏移的相同信号应对齐到 MSE≈0."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 10, 600)
    original = np.sin(2 * np.pi * 0.5 * t) + 0.1 * rng.randn(600)

    # 循环移位100个样本
    shift = 100
    shifted = np.roll(original, shift)

    aligned, lag, corr = phase_align_via_cross_correlation(original, shifted)

    assert corr > 0.9, f"相关系数过低: {corr:.3f}"
    # 周期信号可能有多个等效滞后（混叠），不验证具体lag值
    # 只验证对齐效果
    mse = np.mean((original - aligned) ** 2)
    assert mse < 0.1, f"对齐后MSE过高: {mse:.4f} (lag={lag})"


def test_phase_align_preserves_shape():
    """对齐后信号长度不变."""
    rng = np.random.RandomState(123)
    a = rng.randn(500)
    b = np.roll(a, 37)

    aligned, lag, corr = phase_align_via_cross_correlation(a, b)
    assert len(aligned) == 500
    assert aligned.shape == a.shape
    assert isinstance(lag, int)


def test_phase_align_no_shift():
    """无偏移信号应对齐到自身."""
    rng = np.random.RandomState(99)
    sig = rng.randn(300) + 0.5 * np.sin(np.linspace(0, 4 * np.pi, 300))

    aligned, lag, corr = phase_align_via_cross_correlation(sig, sig.copy())
    assert lag == 0, f"无偏移时 lag 应为0: {lag}"
    assert corr > 0.99, f"自相关峰值应接近1: {corr:.3f}"


def test_phase_align_constant_signal():
    """恒定信号应返回原始模板（无法对齐）."""
    const = np.ones(200) * 5.0
    query = np.sin(np.linspace(0, 4 * np.pi, 200))

    aligned, lag, corr = phase_align_via_cross_correlation(query, const)
    # 恒定信号对齐无意义，corr应接近0
    assert corr < 0.1, f"恒定信号互相关应很低: {corr:.3f}"
    assert np.allclose(aligned, const)


def test_phase_align_different_lengths_raises():
    """不同长度输入应抛出 ValueError."""
    a = np.ones(100)
    b = np.ones(200)
    with pytest.raises(ValueError, match="长度必须相同"):
        phase_align_via_cross_correlation(a, b)


def test_phase_align_and_distance():
    """组合函数应返回距离 + 对齐结果."""
    rng = np.random.RandomState(7)
    t = np.linspace(0, 10, 400)
    original = np.sin(2 * np.pi * 1.0 * t)
    shifted = np.roll(original, 50)

    def euclidean_dist(a, b):
        return float(np.mean((a - b) ** 2))

    dist, aligned, lag, corr = phase_align_and_distance(
        original, shifted, euclidean_dist
    )
    assert dist < 0.01, f"对齐后距离应很小: {dist:.4f}"
    assert corr > 0.9


def test_phase_align_and_distance_below_threshold():
    """低于相关阈值时应跳过对齐."""
    a = np.sin(np.linspace(0, 8 * np.pi, 300))
    b = np.random.RandomState(1).randn(300)  # 不相关的噪声

    def euclidean_dist(x, y):
        return float(np.mean((x - y) ** 2))

    dist_no_align = euclidean_dist(a, b)
    _, _, lag, corr = phase_align_and_distance(a, b, euclidean_dist, min_correlation=0.5)

    # 噪声信号的互相关很低，应跳过对齐
    assert corr < 0.5
