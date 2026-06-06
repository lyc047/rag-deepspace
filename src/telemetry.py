"""合成深空遥测信号生成器.

生成4类遥测信号：慢变平台参数、周期性动态信号、瞬态事件、科学载荷数据。
每种信号附带物理元数据标签，用于知识库的双重索引。
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional


@dataclass
class TelemetrySample:
    """单条遥测样本，包含信号数据+物理元数据."""
    signal: np.ndarray          # 遥测信号值  shape=(n_samples,)
    physics: Dict[str, float]   # 物理元数据
    signal_type: str            # 信号类型标签
    channel_name: str = 'ch0'   # 通道名
    sample_id: Optional[str] = None


def generate_physical_metadata(
    distance_au: Optional[float] = None,
    sun_angle: Optional[float] = None,
    snr_db: Optional[float] = None,
    doppler_hz: Optional[float] = None,
    scint_idx: Optional[float] = None,
    mode: Optional[str] = None,
) -> Dict[str, float]:
    """生成物理元数据字典.

    深空通信场景：火星-地球链路典型参数范围。
    若不指定则随机采样。
    """
    if distance_au is None:
        distance_au = np.random.uniform(0.5, 3.0)
    if sun_angle is None:
        sun_angle = np.random.uniform(0, 90)
    if snr_db is None:
        snr_db = np.random.uniform(-170, -130)
    if doppler_hz is None:
        doppler_hz = np.random.uniform(-50000, 50000)
    if scint_idx is None:
        # 闪烁指数与太阳角负相关
        base_si = max(0, 0.5 - sun_angle / 180)
        scint_idx = base_si + np.random.uniform(-0.05, 0.05)
        scint_idx = np.clip(scint_idx, 0, 0.5)
    if mode is None:
        mode = np.random.choice(['cruise', 'science_obs', 'safe_mode', 'orbit_correction'])

    return {
        'distance_au': float(distance_au),
        'sun_earth_probe_angle': float(sun_angle),
        'snr_db': float(snr_db),
        'doppler_shift_hz': float(doppler_hz),
        'scintillation_index': float(scint_idx),
        'mode': mode,
    }


def generate_slow_varying(
    duration_sec: float = 60.0,
    fs_hz: float = 10.0,
    base_temp: float = 25.0,
    trend: float = 0.0,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成慢变平台参数遥测（如舱体温度）.

    特点：缓慢变化的基线 + 微小随机波动 + 可能的渐变趋势。
    模拟航天器热控系统的温度读数。
    """
    rng = np.random.RandomState(seed)
    n = int(duration_sec * fs_hz)
    t = np.arange(n) / fs_hz

    # 缓慢漂移（多频率正弦叠加）
    drift = (
        0.3 * np.sin(2 * np.pi * 0.001 * t) +
        0.15 * np.sin(2 * np.pi * 0.005 * t + 1.5) +
        0.1 * np.sin(2 * np.pi * 0.01 * t + 0.7)
    )

    # 线性趋势（模拟长期温度变化）
    trend_component = trend * t

    # 随机噪声（传感器噪声+环境微小扰动）
    noise = rng.normal(0, noise_std, n)

    signal = base_temp + drift + trend_component + noise
    return signal.astype(np.float32)


def generate_periodic(
    duration_sec: float = 60.0,
    fs_hz: float = 10.0,
    base_freq: float = 0.5,
    amplitude: float = 1.0,
    harmonics: int = 3,
    noise_std: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成周期性动态信号（如陀螺仪角速度、反作用轮转速）.

    特点：基频 + 谐波 + 幅值调制 + 噪声。
    模拟航天器姿态控制系统中的周期性运动。
    """
    rng = np.random.RandomState(seed)
    n = int(duration_sec * fs_hz)
    t = np.arange(n) / fs_hz

    signal = np.zeros(n)
    for k in range(1, harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        signal += (amplitude / k) * np.sin(2 * np.pi * base_freq * k * t + phase)

    # 缓慢的幅值调制
    envelope = 1.0 + 0.1 * np.sin(2 * np.pi * 0.02 * t)
    signal *= envelope

    # 噪声
    signal += rng.normal(0, noise_std, n)

    return signal.astype(np.float32)


def generate_transient(
    duration_sec: float = 10.0,
    fs_hz: float = 100.0,
    event_time: Optional[float] = None,
    pulse_width: float = 0.5,
    pulse_amplitude: float = 5.0,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成事件驱动瞬态信号（如推进器点火电流脉冲）.

    特点：平稳背景 + 突然脉冲 + 指数衰减恢复。
    模拟一次性事件（推进器点火、模式切换、阀门动作等）。
    """
    rng = np.random.RandomState(seed)
    n = int(duration_sec * fs_hz)
    t = np.arange(n) / fs_hz

    if event_time is None:
        event_time = duration_sec * rng.uniform(0.2, 0.6)

    # 基线
    signal = rng.normal(0, noise_std, n)

    # 脉冲
    pulse_start = int(event_time * fs_hz)
    pulse_end = int((event_time + pulse_width) * fs_hz)
    pulse_end = min(pulse_end, n)

    if pulse_start < n:
        signal[pulse_start:pulse_end] += pulse_amplitude

    # 指数衰减尾部
    decay_start = pulse_end
    if decay_start < n:
        decay_len = n - decay_start
        decay = pulse_amplitude * np.exp(-np.arange(decay_len) / (fs_hz * 0.3))
        signal[decay_start:] += decay

    return signal.astype(np.float32)


def generate_science_data(
    duration_sec: float = 60.0,
    fs_hz: float = 10.0,
    base_value: float = 100.0,
    variability: float = 10.0,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成科学载荷数据（如磁强计读数、等离子体密度）.

    特点：1/f噪声 + 随机起伏 + 可能的突变。
    模拟空间环境测量数据。
    """
    rng = np.random.RandomState(seed)
    n = int(duration_sec * fs_hz)
    t = np.arange(n) / fs_hz

    # 1/f 噪声分量
    freqs = np.fft.rfftfreq(n, d=1 / fs_hz)
    freqs[0] = freqs[1]  # 避免除零
    spectrum = rng.normal(0, 1, len(freqs)) + 1j * rng.normal(0, 1, len(freqs))
    spectrum *= 1.0 / np.sqrt(freqs)
    spectrum[0] = 0
    f_noise = np.fft.irfft(spectrum, n=n).real
    f_noise *= variability / np.std(f_noise)

    # 慢变调制
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * t)

    signal = base_value + f_noise * modulation + rng.normal(0, noise_std, n)
    return signal.astype(np.float32)


def generate_telemetry_segment(
    signal_type: str = 'slow_varying',
    duration_sec: float = 60.0,
    fs_hz: float = 10.0,
    physics: Optional[Dict[str, float]] = None,
    channel_name: str = 'ch0',
    seed: Optional[int] = None,
    model_mismatch: float = 0.0,
) -> TelemetrySample:
    """生成一条完整的遥测片段（信号 + 物理元数据）.

    Args:
        model_mismatch: 模型失配强度 [0, 1].
            0 = 完美模型 (模板和查询同分布)
            0.1 = 轻微失配 (传感器漂移0.1°C, 额外噪声10%)
            0.5 = 严重失配
            KB模板用0, 查询用>0 模拟真实场景

    便捷函数，一步生成TelemetrySample。
    """
    if physics is None:
        physics = generate_physical_metadata()

    generators = {
        'slow_varying': generate_slow_varying,
        'periodic': generate_periodic,
        'transient': generate_transient,
        'science_data': generate_science_data,
    }

    gen_func = generators.get(signal_type, generate_slow_varying)
    signal = gen_func(duration_sec=duration_sec, fs_hz=fs_hz, seed=seed)

    # 模型失配: 模拟真实传感器的不完美
    if model_mismatch > 0:
        rng = np.random.RandomState(
            seed + 99999 if seed is not None else None)
        n = len(signal)
        signal_std = np.std(signal)

        # 传感器漂移 (缓慢偏移)
        drift = model_mismatch * signal_std * np.sin(
            2 * np.pi * 0.0003 * np.arange(n) / fs_hz
            + rng.uniform(0, 2 * np.pi))

        # 额外噪声
        extra_noise = rng.normal(0, model_mismatch * signal_std * 0.1, n)

        # 刻度误差 (校准偏差)
        scale = 1.0 + model_mismatch * rng.uniform(-0.02, 0.02)

        signal = signal * scale + drift + extra_noise

    import uuid
    sample_id = str(uuid.uuid4())[:8]

    return TelemetrySample(
        signal=signal.astype(np.float32),
        physics=physics,
        signal_type=signal_type,
        channel_name=channel_name,
        sample_id=sample_id,
    )
