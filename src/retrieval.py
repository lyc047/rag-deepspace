"""检索策略模块.

实现了5种检索策略：
- RandomRetriever: 随机检索（基线1）
- PhysicsOnlyRetriever: 纯物理检索（基线2）
- SignalOnlyRetriever: 纯信号检索（基线3）
- CoarseRetriever: 物理粗筛
- FineRetriever: 信号精排
- HierarchicalRetriever: 分层检索 = 粗筛 + 精排（核心方案）
"""
import numpy as np
from typing import List, Tuple
from knowledge import KnowledgeBase


class CoarseRetriever:
    """物理参数粗筛：基于SQLite范围查询."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def retrieve(
        self,
        distance_au: float,
        sun_angle: float,
        k: int = 50,
        distance_window: float = 0.5,
        angle_window: float = 15.0,
        snr_range: Tuple[float, float] = (-170, -130),
    ) -> List[Tuple[int, float]]:
        """粗筛：返回候选内部ID列表."""
        candidates = self.kb.coarse_query(
            distance_au=distance_au,
            sun_angle=sun_angle,
            distance_window=distance_window,
            angle_window=angle_window,
            snr_range=snr_range,
        )
        # 如果候选太多，按距离中心最近取Top-K
        if len(candidates) > k:
            candidates.sort(
                key=lambda c: abs(c['distance_au'] - distance_au)
            )
            candidates = candidates[:k]

        return [(c['id'], 0.0) for c in candidates]


class FineRetriever:
    """信号特征精排：在候选集中按信号相似度排序.

    支持单维度度量和多维度加权融合：
    - euclidean: 时域欧氏距离
    - spectral: 频谱包络距离
    - statistical: 统计特征距离 [mean, std, skew, kurt]
    - dtw: 动态时间规整

    Args:
        kb: 知识库
        metric: 距离度量 'euclidean' | 'dtw' | 'multi'
        normalize: 是否归一化
        weights: 多维权重 dict, 仅 metric='multi' 时生效
                 默认 {'euclidean': 0.5, 'spectral': 0.3, 'statistical': 0.2}
    """

    def __init__(self, kb: KnowledgeBase, metric: str = 'euclidean',
                 normalize: bool = False,
                 weights: dict = None):
        self.kb = kb
        self.metric = metric
        self.normalize = normalize
        self.weights = weights or {'euclidean': 0.5, 'spectral': 0.3, 'statistical': 0.2}

    def _prepare(self, q: np.ndarray, t: np.ndarray):
        """准备信号（可选归一化）."""
        if self.normalize:
            q = (q - np.mean(q)) / (np.std(q) + 1e-8)
            t = (t - np.mean(t)) / (np.std(t) + 1e-8)
        return q, t

    def _euclidean_dist(self, q: np.ndarray, t: np.ndarray) -> float:
        return float(np.mean((q - t) ** 2))

    def _spectral_dist(self, q: np.ndarray, t: np.ndarray) -> float:
        """频谱包络距离：比较归一化FFT幅值."""
        fft_q = np.abs(np.fft.rfft(q))
        fft_t = np.abs(np.fft.rfft(t))
        # 归一化总能量
        fft_q = fft_q / (np.sum(fft_q) + 1e-8)
        fft_t = fft_t / (np.sum(fft_t) + 1e-8)
        return float(np.mean((fft_q - fft_t) ** 2))

    def _statistical_dist(self, q: np.ndarray, t: np.ndarray) -> float:
        """统计特征距离."""
        from scipy import stats
        feat_q = np.array([np.mean(q), np.std(q), stats.skew(q), stats.kurtosis(q)])
        feat_t = np.array([np.mean(t), np.std(t), stats.skew(t), stats.kurtosis(t)])
        return float(np.mean((feat_q - feat_t) ** 2))

    def _compute_distance(
        self, y_query: np.ndarray, y_template: np.ndarray,
    ) -> float:
        """计算两条信号的相似度距离."""
        q, t = self._prepare(y_query, y_template)

        if self.metric == 'euclidean':
            return self._euclidean_dist(q, t)

        elif self.metric == 'dtw':
            try:
                from dtaidistance import dtw
                return float(dtw.distance(q, t))
            except ImportError:
                return self._euclidean_dist(q, t)

        elif self.metric == 'multi':
            w = self.weights
            d_euc = self._euclidean_dist(q, t)
            d_spec = self._spectral_dist(q, t)
            d_stat = self._statistical_dist(q, t)
            return (w.get('euclidean', 0.5) * d_euc +
                    w.get('spectral', 0.3) * d_spec +
                    w.get('statistical', 0.2) * d_stat)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def retrieve(
        self,
        y_query: np.ndarray,
        candidates: List[Tuple[int, float]],
        k: int = 3,
    ) -> List[Tuple[int, float]]:
        """精排：对候选集中每条模板计算相似度，返回Top-K."""
        scored = []
        for internal_id, _ in candidates:
            record = self.kb.get_record(internal_id)
            if record is None:
                continue
            dist = self._compute_distance(y_query, record.raw_data)
            scored.append((internal_id, dist))

        scored.sort(key=lambda x: x[1])
        return scored[:k]


class HierarchicalRetriever:
    """分层检索器：粗筛 + 精排（核心方案）."""

    def __init__(
        self,
        kb: KnowledgeBase,
        coarse_k: int = 50,
        fine_k: int = 3,
        metric: str = 'euclidean',
        normalize: bool = False,
        weights: dict = None,
        distance_window: float = 0.5,
        angle_window: float = 15.0,
    ):
        self.kb = kb
        self.coarse = CoarseRetriever(kb)
        self.fine = FineRetriever(kb, metric=metric, normalize=normalize,
                                  weights=weights)
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self.distance_window = distance_window
        self.angle_window = angle_window

    def retrieve(
        self,
        y_query: np.ndarray,
        distance_au: float,
        sun_angle: float,
        snr_range: Tuple[float, float] = (-170, -130),
    ) -> List[Tuple[int, float]]:
        """分层检索：粗筛 → 精排 → Top-K."""
        candidates = self.coarse.retrieve(
            distance_au=distance_au,
            sun_angle=sun_angle,
            k=self.coarse_k,
            distance_window=self.distance_window,
            angle_window=self.angle_window,
            snr_range=snr_range,
        )
        if len(candidates) == 0:
            return []
        return self.fine.retrieve(y_query, candidates, k=self.fine_k)


class RandomRetriever:
    """随机检索（基线1）."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def retrieve(
        self, y_query: np.ndarray, k: int = 1, **kwargs
    ) -> List[Tuple[int, float]]:
        n = self.kb.count
        if n == 0:
            return []
        indices = np.random.choice(n, size=min(k, n), replace=False)
        return [(int(i) + 1, np.nan) for i in indices]


class PhysicsOnlyRetriever:
    """纯物理检索（基线2）：只用粗筛，不做精排."""

    def __init__(self, kb: KnowledgeBase):
        self.coarse = CoarseRetriever(kb)

    def retrieve(
        self, y_query: np.ndarray,
        distance_au: float, sun_angle: float,
        k: int = 3,
        distance_window: float = 0.5,
        angle_window: float = 15.0,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        results = self.coarse.retrieve(
            distance_au=distance_au, sun_angle=sun_angle,
            k=k, distance_window=distance_window,
            angle_window=angle_window,
        )
        return results[:k]


class SignalOnlyRetriever:
    """纯信号检索（基线3）：只用精排，不做粗筛."""

    def __init__(self, kb: KnowledgeBase, metric: str = 'euclidean',
                 normalize: bool = False, weights: dict = None):
        self.kb = kb
        self.fine = FineRetriever(kb, metric=metric, normalize=normalize,
                                  weights=weights)

    def retrieve(
        self, y_query: np.ndarray, k: int = 3, **kwargs
    ) -> List[Tuple[int, float]]:
        n = self.kb.count
        if n == 0:
            return []
        all_candidates = [(i + 1, 0.0) for i in range(n)]
        return self.fine.retrieve(y_query, all_candidates, k=k)
