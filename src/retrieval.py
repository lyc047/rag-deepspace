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

    Args:
        kb: 知识库
        metric: 距离度量 'euclidean' | 'dtw'
        normalize: 是否归一化。True=只看形态, False=保留尺度信息。
                   慢变信号（温度/电压）应设为False，周期性信号可设为True。
    """

    def __init__(self, kb: KnowledgeBase, metric: str = 'euclidean',
                 normalize: bool = False):
        self.kb = kb
        self.metric = metric
        self.normalize = normalize

    def _compute_distance(
        self, y_query: np.ndarray, y_template: np.ndarray,
    ) -> float:
        """计算两条信号的相似度距离."""
        if self.normalize:
            q = (y_query - np.mean(y_query)) / (np.std(y_query) + 1e-8)
            t = (y_template - np.mean(y_template)) / (np.std(y_template) + 1e-8)
        else:
            q = y_query
            t = y_template

        if self.metric == 'euclidean':
            return float(np.mean((q - t) ** 2))

        elif self.metric == 'dtw':
            try:
                from dtaidistance import dtw
                return float(dtw.distance(q, t))
            except ImportError:
                return float(np.mean((q - t) ** 2))

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

        scored.sort(key=lambda x: x[1])  # 距离升序
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
        distance_window: float = 0.5,
        angle_window: float = 15.0,
    ):
        self.kb = kb
        self.coarse = CoarseRetriever(kb)
        self.fine = FineRetriever(kb, metric=metric, normalize=normalize)
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
                 normalize: bool = False):
        self.kb = kb
        self.fine = FineRetriever(kb, metric=metric, normalize=normalize)

    def retrieve(
        self, y_query: np.ndarray, k: int = 3, **kwargs
    ) -> List[Tuple[int, float]]:
        n = self.kb.count
        if n == 0:
            return []
        all_candidates = [(i + 1, 0.0) for i in range(n)]
        return self.fine.retrieve(y_query, all_candidates, k=k)
