"""NASA真实遥测数据加载器.

从NASA SMAP+MSL telemanom数据集加载真实航天器遥测数据，
转换为项目的TelemetrySample格式。

数据源：Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs..." (KDD 2018)
公开S3: s3-us-west-2.amazonaws.com/telemanom/data.zip (~100MB)

局限性（诚实标注）：
- 物理元数据（距离/太阳角/SNR）为估计值，非真实任务参数
- 数据已被作者归一化到[0,1]区间，丢失了原始物理尺度
- MSL = 火星表面漫游车，非深空巡航阶段
"""
import os
import sys
import zipfile
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 项目内部导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from telemetry import TelemetrySample, generate_physical_metadata

# 常量
# 主下载源（可能需翻墙或已失效）
TELEMANOM_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
# 备选：Kaggle数据集
# https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
# 备选：GitHub Releases
# https://github.com/khundman/telemanom
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'telemanom')
WINDOW_SIZE = 600
WINDOW_STRIDE = 200
MAX_SAMPLES_PER_CHANNEL = 40


# --- 下载 ---

def download_telemanom(data_dir: str = DATA_DIR, quiet: bool = False) -> Path:
    """下载并解压telemanom数据集。若已存在则跳过下载。

    Args:
        data_dir: 本地缓存目录
        quiet: 是否静默（不打印进度）

    Returns:
        解压后的数据目录路径

    Raises:
        RuntimeError: 下载失败且本地无缓存时
    """
    data_path = Path(data_dir)
    zip_path = data_path / 'data.zip'

    # 已解压（兼容多种目录结构）
    npy_files_exist = list(data_path.rglob('*.npy'))
    if npy_files_exist:
        if not quiet:
            print(f"[real_data] 数据已缓存 ({len(npy_files_exist)} .npy文件): {data_path}")
        return data_path

    # 下载
    data_path.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        if not quiet:
            print(f"[real_data] 下载中: {TELEMANOM_URL}")
            print(f"[real_data] (~100MB, 可能需要1-2分钟)")

        try:
            urllib.request.urlretrieve(TELEMANOM_URL, str(zip_path))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if (data_path / 'train').exists():
                if not quiet:
                    print(f"[real_data] 下载失败但本地缓存可用: {e}")
                return data_path
            raise RuntimeError(
                f"无法自动下载telemanom数据集（S3可能已限制访问）。\n"
                f"请手动下载数据：\n"
                f"  1. Kaggle: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl\n"
                f"  2. 或运行: pip install kaggle && kaggle datasets download patrickfleith/nasa-anomaly-detection-dataset-smap-msl\n"
                f"  3. 解压所有.npy文件到: {data_path}\n"
                f"错误详情: {e}"
            ) from e

    # 解压
    if not quiet:
        print(f"[real_data] 解压中...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(data_path))

    if not quiet:
        print(f"[real_data] 数据就绪: {data_path}")
    return data_path


# --- 信号类型推断 ---

def _classify_signal_type(signal: np.ndarray, fs_hz: float = 1.0) -> str:
    """启发式分类：根据信号统计特性推断信号类型.

    Args:
        signal: 1D信号数组
        fs_hz: 采样率（telemanom数据归一化后，假设fs=1）

    Returns:
        信号类型: 'slow_varying' | 'periodic' | 'transient'
    """
    n = len(signal)
    if n < 50:
        return 'slow_varying'

    # 自相关检测周期性
    autocorr = np.correlate(signal - np.mean(signal),
                            signal - np.mean(signal), mode='full')
    autocorr = autocorr[n - 1:]  # 只取非负滞后
    autocorr = autocorr / (autocorr[0] + 1e-10)  # 归一化

    # 在 lag ∈ [5, n//4] 范围内找次高峰
    search_start = max(5, n // 20)
    search_end = n // 4
    if search_end > search_start:
        ac_segment = autocorr[search_start:search_end]
        peak_ac = np.max(ac_segment) if len(ac_segment) > 0 else 0

        if peak_ac > 0.4:
            return 'periodic'

    # 瞬态检测：显著尖峰（超过5σ）
    std = np.std(signal)
    if std > 1e-8:
        threshold = 5 * std
        spike_count = np.sum(np.abs(signal - np.mean(signal)) > threshold)
        if spike_count > 0 and spike_count < n * 0.05:
            return 'transient'

    return 'slow_varying'


# --- 数据解析与转换 ---

def parse_npy_channel(filepath: Path) -> np.ndarray:
    """解析单个.npy文件为1D numpy数组.

    telemanom .npy文件格式: shape (n_timesteps,) 或 (n_timesteps, 25)
    对于多变量数据，取第一个主成分（feature 0）作为1D信号。
    """
    data = np.load(str(filepath))
    if data.ndim == 2:
        # (n_timesteps, n_features) → 取第一列
        data = data[:, 0]
    elif data.ndim == 1:
        pass
    else:
        data = data.flatten()
    return data.astype(np.float64)


def segment_time_series(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    max_segments: int = MAX_SAMPLES_PER_CHANNEL,
    rng: Optional[np.random.RandomState] = None,
) -> List[np.ndarray]:
    """将长时间序列切分为固定窗口.

    Args:
        data: 长时间序列
        window_size: 每个窗口的样本数
        stride: 滑动步长
        max_segments: 最多返回多少个窗口（随机采样）
        rng: 随机状态（用于随机采样）

    Returns:
        窗口列表，每个窗口是 shape=(window_size,) 的数组
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(data)
    if n < window_size:
        # 数据太短，填充或返回整个
        padded = np.zeros(window_size)
        padded[:n] = data
        return [padded]

    # 生成所有可能的窗口起始位置
    starts = list(range(0, n - window_size, stride))
    if len(starts) > max_segments:
        # 均匀采样（避免只取开头）
        indices = np.linspace(0, len(starts) - 1, max_segments, dtype=int)
        starts = [starts[i] for i in indices]

    segments = []
    for start in starts:
        seg = data[start:start + window_size].copy()
        segments.append(seg)

    return segments


def estimate_physics_metadata(
    channel_name: str = 'unknown',
    segment_index: int = 0,
) -> Dict[str, float]:
    """生成估计的物理元数据.

    ⚠️ 重要：这些值是基于火星-地球链路的合理假设，不是真实任务参数。

    MSL漫游车在火星表面，地球-火星距离 0.5-2.5 AU，
    太阳角取决于火星在黄道上的位置。
    """
    rng = np.random.RandomState(hash(channel_name) % (2**31) + segment_index)

    # 火星与地球距离：0.5 - 2.5 AU
    distance_au = float(rng.uniform(0.5, 2.5))

    # 火星轨道上太阳角变化
    sun_angle = float(rng.uniform(0, 60))

    # SNR：X-band ~ -150 dB typical
    snr_db = float(rng.uniform(-165, -140))

    # 多普勒：火星表面漫游车相对速度低
    doppler_hz = float(rng.uniform(-5000, 5000))

    # 闪烁指数：火星远离太阳日冕，闪烁低
    scint_idx = float(rng.uniform(0, 0.15))

    return {
        'distance_au': distance_au,
        'sun_earth_probe_angle': sun_angle,
        'snr_db': snr_db,
        'doppler_shift_hz': doppler_hz,
        'scintillation_index': scint_idx,
        'mode': 'science_obs',
    }


# --- 主入口 ---

def load_msl_as_telemetry_samples(
    data_dir: str = DATA_DIR,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    max_per_channel: int = MAX_SAMPLES_PER_CHANNEL,
    seed: int = 42,
    channel_filter: Optional[List[str]] = None,
    quiet: bool = False,
) -> List[TelemetrySample]:
    """加载MSL遥测数据并转换为TelemetrySample列表.

    主入口函数：下载→解析→分类→分段→转换。

    Args:
        data_dir: 数据缓存目录
        window_size: 窗口大小（样本数）
        stride: 滑动步长
        max_per_channel: 每个通道最多取多少窗口
        seed: 随机种子
        channel_filter: 只加载指定通道（None = 全部）
        quiet: 是否静默

    Returns:
        TelemetrySample列表，可用于 KnowledgeBase.insert()

    Raises:
        RuntimeError: 数据不可用且无法下载时
    """
    rng = np.random.RandomState(seed)
    data_path = download_telemanom(data_dir, quiet=quiet)

    # 查找所有.npy文件
    npy_files = list(data_path.rglob('*.npy'))
    if not npy_files:
        raise RuntimeError(
            f"在{data_path}中未找到.npy文件。"
            f"请确认数据已正确下载。"
        )

    # 筛选真正的遥测数据（排除模型输出和误差文件）
    def _is_telemetry(fpath: Path) -> bool:
        """过滤：只保留 train/ 和 test/ 下的原始遥测，排除模型输出."""
        parts = fpath.parts
        # 排除 models/, smoothed_errors/, y_hat/ 目录
        for skip in ('models', 'smoothed_errors', 'y_hat', 'results'):
            if skip in parts:
                return False
        return True

    telemetry_files = [f for f in npy_files if _is_telemetry(f)]
    if not telemetry_files:
        # 回退：使用所有.npy文件
        telemetry_files = npy_files

    # 优先使用 train/ 目录下的文件（更多数据）
    train_files = [f for f in telemetry_files if 'train' in str(f).lower()]
    test_files = [f for f in telemetry_files if 'test' in str(f).lower()]

    if train_files:
        use_files = train_files
        source = "train"
    elif test_files:
        use_files = test_files
        source = "test"
    else:
        use_files = telemetry_files
        source = "all"

    if not quiet:
        print(f"[real_data] 使用 {source}/ 目录: {len(use_files)} 个通道")

    samples = []
    skipped = 0

    for fpath in use_files:
        channel_name = fpath.stem  # 文件名即通道名

        try:
            data = parse_npy_channel(fpath)
        except Exception as e:
            if not quiet:
                print(f"[real_data] 跳过 {fpath.name}: {e}")
            skipped += 1
            continue

        # 推断信号类型
        sig_type = _classify_signal_type(data)

        # 分段
        segments = segment_time_series(
            data, window_size=window_size, stride=stride,
            max_segments=max_per_channel, rng=rng,
        )

        for i, seg in enumerate(segments):
            # 估计物理元数据
            physics = estimate_physics_metadata(channel_name, i)

            sample = TelemetrySample(
                signal=seg.astype(np.float32),
                physics=physics,
                signal_type=sig_type,
                channel_name=channel_name,
                sample_id=f"msl_{channel_name}_{i}",
            )
            samples.append(sample)

    if not quiet:
        n_slow = sum(1 for s in samples if s.signal_type == 'slow_varying')
        n_periodic = sum(1 for s in samples if s.signal_type == 'periodic')
        n_transient = sum(1 for s in samples if s.signal_type == 'transient')
        print(f"[real_data] 加载完成: {len(samples)} 样本 "
              f"(slow={n_slow}, periodic={n_periodic}, transient={n_transient})")
        if skipped > 0:
            print(f"[real_data] 跳过 {skipped} 个文件")

    return samples


# --- 便捷函数 ---

def get_sample_counts_by_type(samples: List[TelemetrySample]) -> Dict[str, int]:
    """统计各信号类型的样本数."""
    counts = {}
    for s in samples:
        counts[s.signal_type] = counts.get(s.signal_type, 0) + 1
    return counts


def split_train_test(
    samples: List[TelemetrySample],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[TelemetrySample], List[TelemetrySample]]:
    """按通道分层拆分为训练/测试集."""
    rng = np.random.RandomState(seed)

    # 按通道分组
    by_channel: Dict[str, List[TelemetrySample]] = {}
    for s in samples:
        by_channel.setdefault(s.channel_name, []).append(s)

    train, test = [], []
    for channel, ch_samples in by_channel.items():
        rng.shuffle(ch_samples)
        n_train = max(1, int(len(ch_samples) * train_ratio))
        train.extend(ch_samples[:n_train])
        test.extend(ch_samples[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
