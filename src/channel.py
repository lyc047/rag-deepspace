"""深空信道仿真器.

Phase 1 (MVP): AWGN + 自由空间损耗
Phase 2 (B级): + 多普勒频移 + 太阳闪烁
"""
import numpy as np
from typing import Optional
from scipy import signal as sp_signal


class AWGNChannel:
    """加性高斯白噪声信道."""

    def __init__(self, snr_db: float, seed: Optional[int] = None):
        self.snr_db = snr_db
        self.rng = np.random.RandomState(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        signal_power = np.mean(x ** 2)
        if signal_power < 1e-12:
            return x.astype(np.float32)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = self.rng.normal(0, np.sqrt(noise_power), x.shape)
        return (x + noise).astype(np.float32)


class FreeSpacePathLoss:
    """自由空间路径损耗模型. L = (4πd/λ)²"""

    SPEED_OF_LIGHT = 3e8
    AU_TO_M = 1.496e11

    @staticmethod
    def compute_loss_db(distance_au: float, freq_ghz: float = 8.4) -> float:
        d_m = distance_au * FreeSpacePathLoss.AU_TO_M
        wavelength = FreeSpacePathLoss.SPEED_OF_LIGHT / (freq_ghz * 1e9)
        loss_linear = (4 * np.pi * d_m / wavelength) ** 2
        return 10 * np.log10(loss_linear)

    @staticmethod
    def attenuate(x: np.ndarray, distance_au: float, freq_ghz: float = 8.4) -> np.ndarray:
        loss_db = FreeSpacePathLoss.compute_loss_db(distance_au, freq_ghz)
        loss_linear = 10 ** (-loss_db / 20)
        return (x * loss_linear).astype(np.float32)


class DopplerShift:
    """多普勒频移效应.

    模拟探测器-地球相对运动引起的频率偏移。
    对基带信号的影响：在频谱上产生平移，等效于时域相位旋转。

    简化模型：通过重采样模拟频率偏移。
    y(t) ≈ x(t) * exp(j*2π*fd*t)，对实信号通过频谱搬移近似。
    """

    def __init__(self, doppler_hz: float, fs_hz: float = 10.0):
        self.doppler_hz = doppler_hz
        self.fs_hz = fs_hz

    def forward(self, x: np.ndarray) -> np.ndarray:
        """施加多普勒频移.

        对实信号：通过正交调制/解调模拟频移。
        y[n] = x[n] * cos(2π*fd*n/fs) - H{x}[n] * sin(2π*fd*n/fs)
        其中H{x}是x的Hilbert变换。
        """
        if abs(self.doppler_hz) < 1.0:
            return x.astype(np.float32)

        n = len(x)
        t = np.arange(n) / self.fs_hz

        # 正交频移：x_real * cos + x_imag * sin
        analytic = sp_signal.hilbert(x)
        carrier_cos = np.cos(2 * np.pi * self.doppler_hz * t)
        carrier_sin = np.sin(2 * np.pi * self.doppler_hz * t)

        y = x * carrier_cos - np.imag(analytic) * carrier_sin
        return y.astype(np.float32)


class SolarScintillation:
    """太阳闪烁效应.

    太阳风等离子体引起的信号幅度和相位随机起伏。
    m4 (闪烁指数): 0=无闪烁, 0.5=强闪烁。
    """

    def __init__(self, scintillation_index: float, fs_hz: float = 10.0,
                 seed: Optional[int] = None):
        self.m4 = np.clip(scintillation_index, 0, 0.5)
        self.fs_hz = fs_hz
        self.rng = np.random.RandomState(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """施加太阳闪烁（乘性噪声）.

        闪烁幅度服从Rice分布，闪烁指数m4决定起伏强度。
        同时叠加缓慢的相位漂移。
        """
        if self.m4 < 0.01:
            return x.astype(np.float32)

        n = len(x)

        # 生成闪烁包络（低频乘性噪声）
        # 闪烁的截止频率 ~0.1Hz（深空链路典型值）
        cutoff = 0.1 / (self.fs_hz / 2)
        if cutoff >= 1.0:
            cutoff = 0.99
        b, a = sp_signal.butter(2, cutoff)

        # 生成白噪声 → 低通滤波 → 指数变换 → 对数正态分布闪烁
        white_noise = self.rng.randn(n)
        filtered = sp_signal.filtfilt(b, a, white_noise)
        filtered = filtered / np.std(filtered)

        # m4越高，起伏越大。映射m4∈[0,0.5] → σ∈[0,0.3]
        sigma = self.m4 * 0.6
        envelope = np.exp(sigma * filtered)
        envelope = envelope / np.mean(envelope)  # 保持平均功率

        return (x * envelope).astype(np.float32)


class DeepSpaceChannel:
    """深空信道（完整B级：自由空间损耗 + 多普勒 + 太阳闪烁 + AWGN）."""

    def __init__(
        self,
        distance_au: float = 2.0,
        snr_db: float = -150,
        freq_ghz: float = 8.4,
        doppler_hz: float = 0.0,
        scintillation_index: float = 0.0,
        fs_hz: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.distance_au = distance_au
        self.snr_db = snr_db
        self.freq_ghz = freq_ghz
        self.fs_hz = fs_hz
        self.rng = np.random.RandomState(seed)

        self._awgn = AWGNChannel(snr_db=snr_db, seed=seed)
        self._doppler = DopplerShift(doppler_hz, fs_hz)
        self._scint = SolarScintillation(scintillation_index, fs_hz, seed)

    def forward(self, x: np.ndarray, level: str = 'B') -> np.ndarray:
        """深空信道前向传播.

        Args:
            x: 输入信号
            level: 'A'=仅AWGN+FSPL, 'B'=完整B级(多普勒+闪烁)

        Returns:
            信道输出信号
        """
        y = FreeSpacePathLoss.attenuate(x, self.distance_au, self.freq_ghz)

        if level == 'B':
            y = self._doppler.forward(y)
            y = self._scint.forward(y)

        y = self._awgn.forward(y)
        return y
