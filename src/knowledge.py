"""知识库模块：SQLite物理索引 + FAISS信号特征向量检索.

双重索引架构：
- SQLite：存储物理元数据，支持范围查询（粗筛）
- FAISS：存储信号统计特征向量，支持ANN检索（精排辅助）
"""
import sqlite3
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats


@dataclass
class TemplateRecord:
    """知识库中的一条模板记录.

    Attributes:
        raw_data: 原始信号数据 (n_samples,)
        physics: 物理元数据字典
        signal_type: 信号类型标签
        channel_name: 通道名
        sample_id: 可选的唯一标识符
    """
    raw_data: np.ndarray
    physics: Dict[str, float]
    signal_type: str = 'slow_varying'
    channel_name: str = 'ch0'
    sample_id: Optional[str] = None

    def __post_init__(self):
        self._compute_features()

    def _compute_features(self):
        """计算信号统计特征向量 [mean, std, skewness, kurtosis]."""
        y = self.raw_data
        self.stat_features = np.array([
            float(np.mean(y)),
            float(np.std(y)),
            float(sp_stats.skew(y)),
            float(sp_stats.kurtosis(y)),
        ], dtype=np.float32)


class KnowledgeBase:
    """知识库：双重索引（SQLite + FAISS）."""

    def __init__(self, db_path: str = ':memory:'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self._records: List[TemplateRecord] = []
        self._faiss_index = None  # 延迟构建

    def _init_db(self):
        """初始化SQLite表."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT,
                distance_au REAL,
                sun_earth_probe_angle REAL,
                snr_db REAL,
                doppler_shift_hz REAL,
                scintillation_index REAL,
                mode TEXT,
                signal_type TEXT,
                channel_name TEXT,
                stat_mean REAL,
                stat_std REAL,
                stat_skew REAL,
                stat_kurt REAL,
                n_samples INTEGER
            )
        ''')
        self.conn.commit()

    def insert(self, record: TemplateRecord) -> int:
        """插入一条模板记录，返回内部ID."""
        import uuid
        if record.sample_id is None:
            record.sample_id = str(uuid.uuid4())[:8]

        p = record.physics
        cursor = self.conn.execute('''
            INSERT INTO templates
            (sample_id, distance_au, sun_earth_probe_angle, snr_db,
             doppler_shift_hz, scintillation_index, mode,
             signal_type, channel_name,
             stat_mean, stat_std, stat_skew, stat_kurt, n_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.sample_id,
            p['distance_au'], p['sun_earth_probe_angle'], p['snr_db'],
            p['doppler_shift_hz'], p['scintillation_index'], p['mode'],
            record.signal_type, record.channel_name,
            float(record.stat_features[0]), float(record.stat_features[1]),
            float(record.stat_features[2]), float(record.stat_features[3]),
            len(record.raw_data),
        ))
        self.conn.commit()
        internal_id = cursor.lastrowid
        self._records.append(record)
        return internal_id

    def coarse_query(
        self,
        distance_au: float,
        sun_angle: float,
        distance_window: float = 0.5,
        angle_window: float = 15.0,
        snr_range: Tuple[float, float] = (-170, -130),
    ) -> List[Dict]:
        """物理参数粗筛：范围查询返回候选集.

        Args:
            distance_au: 当前距离(AU)
            sun_angle: 当前太阳角(°)
            distance_window: 距离窗口(±AU)
            angle_window: 角度窗口(±°)
            snr_range: SNR范围

        Returns:
            候选模板的物理参数列表（不含raw_data）
        """
        query = '''
            SELECT id, sample_id, distance_au, sun_earth_probe_angle,
                   snr_db, doppler_shift_hz, scintillation_index, mode,
                   signal_type, channel_name, stat_mean, stat_std,
                   stat_skew, stat_kurt
            FROM templates
            WHERE distance_au BETWEEN ? AND ?
              AND sun_earth_probe_angle BETWEEN ? AND ?
              AND snr_db BETWEEN ? AND ?
        '''
        params = (
            distance_au - distance_window, distance_au + distance_window,
            sun_angle - angle_window, sun_angle + angle_window,
            snr_range[0], snr_range[1],
        )
        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_raw_data(self, internal_id: int) -> Optional[np.ndarray]:
        """根据内部ID获取原始信号数据."""
        idx = internal_id - 1
        if 0 <= idx < len(self._records):
            return self._records[idx].raw_data
        return None

    def get_record(self, internal_id: int) -> Optional[TemplateRecord]:
        """根据内部ID获取完整的TemplateRecord."""
        idx = internal_id - 1
        if 0 <= idx < len(self._records):
            return self._records[idx]
        return None

    def stat_filter(
        self,
        y_query: np.ndarray,
        candidate_ids: List[int],
        stat_window: float = 3.0,
        max_keep: int = 200,
    ) -> List[int]:
        """统计特征过滤：保留统计特征接近查询的候选.

        对候选模板计算 [mean, std, skew, kurt] 欧氏距离，
        按距离排序后保留前 max_keep 个。

        Args:
            y_query: 查询信号
            candidate_ids: 候选模板的内部ID列表
            stat_window: 统计特征距离阈值（标准差倍数），留作接口
            max_keep: 最多保留多少个候选

        Returns:
            过滤后的内部ID列表
        """
        from scipy import stats as sp_stats
        q_features = np.array([
            float(np.mean(y_query)),
            float(np.std(y_query)),
            float(sp_stats.skew(y_query)),
            float(sp_stats.kurtosis(y_query)),
        ])

        scored = []
        for cid in candidate_ids:
            record = self.get_record(cid)
            if record is None:
                continue
            dist = float(np.sqrt(np.mean((q_features - record.stat_features) ** 2)))
            scored.append((cid, dist))

        if len(scored) <= max_keep:
            return [cid for cid, _ in scored]

        scored.sort(key=lambda x: x[1])
        return [cid for cid, _ in scored[:max_keep]]

    @property
    def count(self) -> int:
        cursor = self.conn.execute('SELECT COUNT(*) FROM templates')
        return cursor.fetchone()[0]

    def build_faiss_index(self):
        """构建FAISS索引（在插入所有模板后调用）.

        FAISS为可选依赖。若未安装，使用暴力搜索(NumPy)替代.
        """
        if len(self._records) == 0:
            return
        features = np.array([r.stat_features for r in self._records],
                            dtype=np.float32)
        try:
            import faiss
            d = features.shape[1]
            self._faiss_index = faiss.IndexFlatL2(d)
            self._faiss_index.add(features)
        except ImportError:
            # FAISS未安装：存储特征矩阵用于暴力搜索
            self._faiss_index = features

    def search_by_features(
        self, query_features: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """基于统计特征的ANN检索（FAISS或暴力搜索）.

        Args:
            query_features: 查询特征向量 shape=(4,)
            k: 返回Top-K

        Returns:
            (indices, distances) 距离越小越相似
        """
        if self._faiss_index is None:
            self.build_faiss_index()
        query = query_features.astype(np.float32).reshape(1, -1)

        try:
            import faiss
            if isinstance(self._faiss_index, faiss.IndexFlatL2):
                distances, indices = self._faiss_index.search(query, k)
                return indices[0], distances[0]
        except ImportError:
            pass

        # 暴力搜索回退 (FAISS未安装或特征矩阵模式)
        features = self._faiss_index  # np.ndarray (n_records, d)
        dists = np.sum((features - query) ** 2, axis=1)
        top_k = min(k, len(dists))
        indices = np.argsort(dists)[:top_k]
        return indices, dists[indices]


class KnowledgeBaseBuilder:
    """知识库构建器：批量生成模板并填充知识库."""

    def __init__(
        self,
        n_templates: int = 1000,
        signal_types: Optional[List[str]] = None,
        duration_sec: float = 60.0,
        fs_hz: float = 10.0,
    ):
        self.n_templates = n_templates
        self.signal_types = signal_types or ['slow_varying']
        self.duration_sec = duration_sec
        self.fs_hz = fs_hz
        self._physics_grid = self._generate_physics_grid(n_templates)

    def _generate_physics_grid(self, n: int) -> List[Dict[str, float]]:
        """网格采样物理参数空间."""
        from telemetry import generate_physical_metadata
        grid = []
        for i in range(n):
            distance = 0.5 + 2.5 * i / max(n - 1, 1)
            angle = 90 * i / max(n - 1, 1)
            snr = -170 + 40 * i / max(n - 1, 1)
            grid.append(generate_physical_metadata(
                distance_au=distance,
                sun_angle=angle,
                snr_db=snr,
            ))
        return grid

    def build(self, kb: KnowledgeBase, seed: int = 42):
        """构建知识库：生成模板并插入.

        Args:
            kb: 目标知识库
            seed: 随机种子
        """
        from telemetry import generate_telemetry_segment
        rng = np.random.RandomState(seed)

        for i, physics in enumerate(self._physics_grid):
            signal_type = self.signal_types[i % len(self.signal_types)]
            seg = generate_telemetry_segment(
                signal_type=signal_type,
                duration_sec=self.duration_sec,
                fs_hz=self.fs_hz,
                physics=physics,
                seed=rng.randint(0, 2**31 - 1),
            )
            record = TemplateRecord(
                raw_data=seg.signal,
                physics=seg.physics,
                signal_type=seg.signal_type,
                channel_name=seg.channel_name,
                sample_id=seg.sample_id,
            )
            kb.insert(record)

        kb.build_faiss_index()

    @property
    def physics_grid(self):
        return self._physics_grid
