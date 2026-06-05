"""测试真实数据加载器（离线测试，不需要下载数据）."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

# 注意：以下导入依赖 sys.path hack
from real_data import (
    _classify_signal_type,
    segment_time_series,
    estimate_physics_metadata,
    split_train_test,
)


# --- 信号类型分类测试 ---

def test_classify_slow_varying():
    """缓慢漂移信号应分类为slow_varying."""
    rng = np.random.RandomState(1)
    t = np.linspace(0, 100, 600)
    sig = 25.0 + 0.1 * np.sin(2 * np.pi * 0.005 * t) + rng.normal(0, 0.05, 600)
    assert _classify_signal_type(sig) == 'slow_varying'


def test_classify_periodic():
    """强周期信号应分类为periodic."""
    t = np.linspace(0, 100, 600)
    sig = np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.random.RandomState(2).randn(600)
    assert _classify_signal_type(sig) == 'periodic'


def test_classify_transient():
    """含尖峰信号应分类为transient."""
    rng = np.random.RandomState(3)
    sig = rng.normal(0, 0.01, 600)
    sig[200:210] = 0.5  # 显著尖峰
    assert _classify_signal_type(sig) == 'transient'


def test_classify_short_signal():
    """过短信号默认归为slow_varying."""
    sig = np.random.RandomState(4).randn(30)
    assert _classify_signal_type(sig) == 'slow_varying'


# --- 分段测试 ---

def test_segment_time_series_basic():
    """基本分段功能."""
    data = np.arange(1000, dtype=float)
    segments = segment_time_series(data, window_size=200, stride=100, max_segments=10)
    assert len(segments) > 0
    assert all(len(s) == 200 for s in segments)


def test_segment_time_series_short_data():
    """短于窗口的数据应填充."""
    data = np.array([1.0, 2.0, 3.0])
    segments = segment_time_series(data, window_size=600, stride=200)
    assert len(segments) == 1
    assert len(segments[0]) == 600
    assert segments[0][0] == 1.0


def test_segment_max_limit():
    """max_segments限制应生效."""
    data = np.random.RandomState(5).randn(5000)
    segments = segment_time_series(data, window_size=200, stride=50, max_segments=10)
    assert len(segments) <= 10


# --- 物理元数据测试 ---

def test_estimate_physics_metadata_keys():
    """元数据应包含所有必需字段."""
    meta = estimate_physics_metadata("test_channel", 0)
    required = ['distance_au', 'sun_earth_probe_angle', 'snr_db',
                'doppler_shift_hz', 'scintillation_index', 'mode']
    for key in required:
        assert key in meta, f"缺少字段: {key}"


def test_estimate_physics_metadata_ranges():
    """元数据取值范围应合理."""
    meta = estimate_physics_metadata("ch_1", 5)
    assert 0.3 <= meta['distance_au'] <= 3.0, f"距离异常: {meta['distance_au']}"
    assert 0 <= meta['sun_earth_probe_angle'] <= 90, f"太阳角异常: {meta['sun_earth_probe_angle']}"
    assert -170 <= meta['snr_db'] <= -130, f"SNR异常: {meta['snr_db']}"


def test_estimate_physics_deterministic():
    """相同输入应产生相同元数据."""
    m1 = estimate_physics_metadata("A", 0)
    m2 = estimate_physics_metadata("A", 0)
    assert m1['distance_au'] == m2['distance_au']


# --- 训练/测试拆分 ---

def test_split_train_test():
    """拆分逻辑应保持样本完整性."""
    # 构造假样本
    class FakeSample:
        def __init__(self, ch, sid):
            self.channel_name = ch
            self.signal_type = 'slow_varying'
            self.sample_id = sid
            self.signal = np.ones(100)
            self.physics = {}

    samples = []
    for ch in ['A', 'B', 'C']:
        for i in range(20):
            samples.append(FakeSample(ch, f"{ch}_{i}"))

    train, test = split_train_test(samples, train_ratio=0.7, seed=42)

    # 总数不变
    assert len(train) + len(test) == 60

    # 每个通道在训练和测试中都有
    train_ch = {s.channel_name for s in train}
    test_ch = {s.channel_name for s in test}
    assert train_ch == test_ch == {'A', 'B', 'C'}

    # 训练集约70%
    assert 35 <= len(train) <= 49, f"训练集比例异常: {len(train)}/60"
