"""深空信道仿真器.

Phase 1 (MVP): AWGN + 自由空间损耗
Phase 2: + 多普勒频移 + 太阳闪烁
"""
import numpy as np
from typing import Optional


class AWGNChannel:
    """加性高斯白噪声信道."""

    def __init__(self, snr_db: float, seed: Optional[int] = None):
        self.snr_db = snr_db
        self.rng = np.random.RandomState(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """在输入信号上叠加AWGN噪声.

        Args:
            x: 输入信号

        Returns:
            含噪信号
        """
        signal_power = np.mean(x ** 2)
        if signal_power < 1e-12:
            return x.astype(np.float32)

        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = self.rng.normal(0, np.sqrt(noise_power), x.shape)
        return (x + noise).astype(np.float32)


class FreeSpacePathLoss:
    """自由空间路径损耗模型.

    L = (4πd/λ)²
    d: 距离(m), λ: 波长(m)
    """

    SPEED_OF_LIGHT = 3e8  # m/s
    AU_TO_M = 1.496e11    # 1 AU = 1.496e11 m

    @staticmethod
    def compute_loss_db(distance_au: float, freq_ghz: float = 8.4) -> float:
        """计算路径损耗 (dB).

        Args:
            distance_au: 距离(AU)
            freq_ghz: 载波频率(GHz)，默认X波段8.4GHz

        Returns:
            路径损耗 (dB)
        """
        d_m = distance_au * FreeSpacePathLoss.AU_TO_M
        wavelength = FreeSpacePathLoss.SPEED_OF_LIGHT / (freq_ghz * 1e9)
        loss_linear = (4 * np.pi * d_m / wavelength) ** 2
        return 10 * np.log10(loss_linear)

    @staticmethod
    def attenuate(
        x: np.ndarray,
        distance_au: float,
        freq_ghz: float = 8.4,
    ) -> np.ndarray:
        """对信号施加自由空间衰减.

        Args:
            x: 输入信号
            distance_au: 距离(AU)
            freq_ghz: 频率(GHz)

        Returns:
            衰减后的信号
        """
        loss_db = FreeSpacePathLoss.compute_loss_db(distance_au, freq_ghz)
        loss_linear = 10 ** (-loss_db / 20)  # 幅值衰减
        return (x * loss_linear).astype(np.float32)


class DeepSpaceChannel:
    """深空信道（Phase 1 MVP：自由空间损耗 + AWGN）."""

    def __init__(
        self,
        distance_au: float = 2.0,
        snr_db: float = -150,
        freq_ghz: float = 8.4,
        seed: Optional[int] = None,
    ):
        self.distance_au = distance_au
        self.freq_ghz = freq_ghz
        self.rng = np.random.RandomState(seed)
        self._awgn = AWGNChannel(snr_db=snr_db, seed=seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """深空信道前向传播.

        Phase 1: 自由空间衰减 → AWGN
        Phase 2 将加入多普勒和太阳闪烁.
        """
        # Step 1: 自由空间损耗
        y = FreeSpacePathLoss.attenuate(x, self.distance_au, self.freq_ghz)
        # Step 2: AWGN
        y = self._awgn.forward(y)
        return y
