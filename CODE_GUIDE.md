# 项目代码文件详解

> 总代码量: 5,655行 Python (src 2,288 + experiments 2,717 + tests 650)
> 10个核心模块 | 12个实验 | 61个测试 | 6个文档

---

## 一、核心模块 (`src/`)

### 1. `src/telemetry.py` (264行) — 遥测信号生成器

**项目入口**，负责生成所有合成遥测信号。

```
TelemetrySample (dataclass)
├── signal: np.ndarray (600,)     ← 信号数据
├── physics: dict                  ← 物理元数据(距离/角度/SNR/多普勒/闪烁/模式)
├── signal_type: str               ← 'slow_varying'|'periodic'|'transient'|'science_data'
└── sample_id: str                 ← UUID唯一标识
```

**四个生成函数:**

| 函数 | 输出 | 数学模型 | 应用场景 |
|------|------|---------|---------|
| `generate_slow_varying()` | 温度~25°C | 多频正弦叠加+线性趋势+N(0,0.05) | 舱体温度/电压 |
| `generate_periodic()` | 振幅~1.0 | 基频0.5Hz+3次谐波(随机相位)+幅值调制+N(0,0.02) | 陀螺仪/反作用轮 |
| `generate_transient()` | 脉冲+指数衰减 | 平稳背景N(0,0.01)+5.0脉冲(0.5秒)+exp(-t/0.3)衰减 | 推进器点火 |
| `generate_science_data()` | 1/f噪声 | 频域法: 随机频谱×1/sqrt(f)+慢变调制 | 磁强计/等离子体 |

**关键细节:**
- 第114行: 周期信号的谐波相位是 `rng.uniform(0, 2π)` — 每次调用不同 → 这就是"相位偏移问题"的根源
- `generate_physical_metadata()`: 生成6维物理参数(30行), `model_mismatch` 参数施加传感器漂移+刻度误差+额外噪声(第220行)

---

### 2. `src/channel.py` (168行) — 深空信道模拟器

**模拟信号从探测器到地球的物理传输过程。**

```
DeepSpaceChannel
├── FreeSpacePathLoss  →  FSPL(dB) = 20·log₁₀(4πd/λ)
├── AWGN              →  w[n] ~ N(0, σ²), σ² = P_signal/10^(SNR/10)
├── DopplerShift      →  y[n] = x[n]·cos(2π·fd·n/fs) - H{x}[n]·sin(2π·fd·n/fs)
└── SolarScintillation →  Butterworth滤波有色噪声, m₄∈[0,0.5]
```

**两级信道:**
- A级(第115行): AWGN + FSPL（基础）
- B级(第128行): 全部四效应叠加 `forward(x, level='B')`

---

### 3. `src/knowledge.py` (320行) — 知识库

**存储和索引信号模板，是RAG的Retrieval组件。**

```
KnowledgeBase (第45行)
├── SQLite 表 (第57行)        → 6维物理参数的范围查询
│   distance_au, sun_angle, snr_db, doppler_hz, scintillation, mode
├── FAISS 向量索引 (第164行)   → 4维统计特征的ANN搜索
│   [mean, std, skew, kurt]
├── 内存列表 (第52行)         → 原始信号数据(600,)快速存取
│   _records: List[TemplateRecord]
├── stat_filter() (第170行)   → 统计特征过滤（核心创新）
│   对所有候选算4-dim欧氏距离 → 保留Top-200
└── KnowledgeBaseBuilder (第214行) → 批量生成模板+填充KB
    _generate_physics_grid(): 网格采样距离/角度/SNR参数空间
```

**TemplateRecord (第14行):**
- 每条模板存储: raw_data(600,) + physics(6) + stat_features(4)
- `_compute_features()`: 插入时自动计算 [mean, std, skew, kurt]

---

### 4. `src/retrieval.py` (293行) — 检索策略

**实现7种检索方法，是RAG的Retrieval+Ranking组件。**

```
检索器层次结构:

CoarseRetriever (第16行)
├── SQLite WHERE distance_au BETWEEN ? AND ?
└── 返回: [(id, score), ...]  ← 物理粗筛

FineRetriever (第49行)
├── _euclidean_dist()       → mean((q-t)²)
├── _spectral_dist()        → mean((|FFT(q)|-|FFT(t)|)²)
├── _statistical_dist()     → mean((feat_q-feat_t)²)
├── _compute_distance(dtw)  → dtaidistance.dtw(q,t,window=n//10)
├── _compute_distance(multi)→ 0.5*euc + 0.3*spec + 0.2*stat
└── phase_align (第86行)    → 精排前对齐模板到尾询

HierarchicalRetriever (第147行)
├── CoarseRetriever + FineRetriever 的串联
├── stat_filter (第192行)           → 统计特征过滤（核心）
├── adaptive_window (exp03)          → 太阳角+SNR联合调节
└── last_coarse_count                → 效率追踪

RandomRetriever (第191行)     → 随机基线
PhysicsOnlyRetriever (第207行) → 仅物理粗筛基线
SignalOnlyRetriever (第229行)   → 全库搜索上界
```

---

### 5. `src/reconstruction.py` (195行) — 信号重建

**实现模板+残差+量化的完整传输流水线。**

```
核心函数: reconstruct_from_template() (第55行)

完整流水线:
  发射端:
    ① 查KB得 y_template_kb
    ② [相位对齐] aligned, lag, corr = phase_align(y_original, y_template_kb)
    ③ residual = y_original - aligned
    ④ [信道] residual = channel.forward(residual)
    ⑤ [量化] residual_q = uniform_quantize(residual, n_bits)
    ⑥ 传输: template_id(+CRC8) + lag(10bit) + mode(1bit) + residual_q

  接收端:
    ⑦ 查KB得 y_template_kb
    ⑧ y_template_receiver = np.roll(y_template_kb, lag)  ← 复原对齐
    ⑨ y_recon = y_template_receiver + residual_q

辅助函数:
  quantize_uniform() (第21行):  step = 2·max/(2^bits-1)
  bandwidth_bits() (第131行):  开销计算(含CRC+lag+mode)
  crc8() (第15行):            多项式 0x107
```

---

### 6. `src/preprocessing.py` (121行) — 信号预处理

```
phase_align_via_cross_correlation() (第11行)
├── 互相关: corr = correlate(q_demean, t_demean, mode='full')
├── 归一化: corr_norm = corr/(N·σ_q·σ_t)
├── 最优lag: argmax(corr_norm) - (N-1)
├── 对齐: aligned = np.roll(template, lag)
└── 返回: (aligned, lag, peak_correlation)

doppler_compensate() (第90行)
└── 逆多普勒频移: DopplerShift(-doppler_hz).forward(y)

phase_align_and_distance() (第98行)
└── 对齐+距离计算组合
```

---

### 7. `src/ccsds121.py` (168行) — CCSDS 121.0编码器

```
ccsds121_preprocess() (第14行)
├── 单位延迟预测器: delta[n] = x[n] - x[n-1]
└── 参考样本: 每J=16个样本保留原始值

rice_encode_block() (第32行)
├── 遍历k=0..14搜索最优Rice参数
├── 编码: quotient(unary) + remainder(k-bit) + sign(1-bit)
└── 返回: (总比特数, 最优k)

ccsds121_compress() (第58行)
├── 8-bit线性量化 → 预处理 → 逐块Rice编码
└── 返回: {bits_per_sample, compression_ratio, ...}

benchmark_ccsds() (第133行)
└── 快速压缩比基准测试
```

---

### 8. `src/real_data.py` (396行) — NASA真实数据加载器

```
download_telemanom() (第32行)
├── 检查本地缓存 data/telemanom/
├── 下载 ~100MB data.zip (S3/Kaggle)
└── 解压提取 .npy 文件

_classify_signal_type() (第121行)
├── 自相关检测周期性 → 'periodic'
├── 5σ尖峰检测瞬态 → 'transient'
└── 默认 → 'slow_varying'

parse_npy_channel() (第147行)
├── shape (T, 25) → 取第一列 → shape (T,)
└── 返回 1D 数组

segment_time_series() (第158行)
├── 长时序 → window_size=600 滑动窗口
└── max_per_channel=40 限制

estimate_physics_metadata() (第194行)
├── 火星-地球距离 0.5-2.5 AU (估计)
└── 标注为"非真实任务参数"

load_msl_as_telemetry_samples() (第224行) → 主入口
├── 下载→解析→分类→分段→转换
└── 返回: List[TelemetrySample]

split_train_test() (第310行)
└── 按通道分层拆分 70/30
```

---

### 9. `src/metrics.py` (38行) — 评估指标

| 函数 | 公式 | 用途 |
|------|------|------|
| `mse(y, y_hat)` | mean((y-y_hat)²) | 重建质量 |
| `mae(y, y_hat)` | mean(|y-y_hat|) | 绝对误差 |
| `pearson(y, y_hat)` | cov/(σy·σŷ) | 波形相似度 |
| `recall_at_k(oracle_id, candidates, k)` | oracle ∈ top-k? | 检索准确率 |
| `bandwidth_compression_ratio(n_bits, n_samples)` | PCM/ours | 带宽压缩 |

---

### 10. `src/visualization.py` (324行) — 论文图表

| 函数 | 输出 |
|------|------|
| `plot_snr_vs_mse()` | SNR-MSE横评曲线 (semilogy) |
| `plot_multisignal_grid()` | 多信号2×2子图 |
| `plot_efficiency_comparison()` | 候选数vs SNR |
| `plot_signal_waveform()` | 原始/重建/模板波形对比 |
| `_apply_jitter()` | 重合曲线微偏移 (1±2%) |
| `_add_overlap_annotations()` | 重合区域标注 |

样式映射 `_STYLE_MAP`: 17种方法的颜色/线型/标记预设。

---

## 二、实验脚本 (`experiments/`)

| 文件 | 行数 | 核心问题 | 关键结果 |
|------|:---:|------|---------|
| `exp01_mvp_baselines.py` | 195 | 6方法基线对比 | 自适应MSE=0.005追平上界 |
| `exp02_diagnosis.py` | 203 | 为什么窄窗口失败？ | 87%最优模板在0.5AU外 |
| `exp03_adaptive_window.py` | 258 | 自适应窗口8方法对比 | v2(角度+SNR)最优 |
| `exp04_multidimensional.py` | 130 | 多信号×多维加权+B级信道 | 多维度无显著增益 |
| `exp05_periodic_phase.py` | 202 | 相位对齐+DTW修复periodic | Phase-AE -44% |
| `exp06_real_data.py` | 239 | NASA SMAP真实数据 | 同通道残差≈0 |
| `exp07_bandwidth.py` | 288 | 质量vs比特率曲线 | 1-bit=87%带宽省 |
| `exp08_large_scale.py` | 285 | 500→10000模板+条件分层 | Stats-Only 100% Recall |
| `exp09_template_value.py` | 176 | 三信号模板重建 | slow 124k×, periodic 3×, transient 1500× |
| `exp10_phase_recon.py` | 202 | 对齐前后对比 | transient 2930× |
| `exp11_resource_compare.py` | 301 | 带宽/质量/计算量+失配鲁棒性 | η=0.3仍优2-45000× |
| `exp12_ccsds_stack.py` | 237 | CCSDS叠加压缩 | 1.02bps (7.9×) |

**共同模式:** 每个实验都遵循 KB构建 → 检索器初始化 → 扫描循环 → 聚合(trim_mean) → 画图

---

## 三、测试 (`tests/`)

| 文件 | 测试数 | 覆盖内容 |
|------|:---:|------|
| `test_telemetry.py` | 7 | 信号形状/频率/脉冲/元数据范围 |
| `test_channel.py` | 10 | AWGN/FSPL/Doppler/Scintillation/B级 |
| `test_knowledge.py` | 6 | 插入/查询/统计特征/FAISS |
| `test_retrieval.py` | 6 | 粗筛/精排/分层/随机/物理/信号检索器 |
| `test_reconstruction.py` | 6 | 重建/空结果/残差/量化/带宽/CRC8 |
| `test_preprocessing.py` | 7 | 互相关对齐/恒定信号/长度检测/阈值 |
| `test_metrics.py` | 6 | MSE/MAE/Pearson/Recall |
| `test_real_data.py` | 11 | 分类/分段/元数据/拆分 (离线) |
| **总计** | **61** | |

---

## 四、文档 (`*.md`)

| 文件 | 内容 |
|------|------|
| `THESIS.md` (914行) | 10章硕士论文格式完整文档 |
| `KNOWLEDGE_MAP.md` (350行) | 10大类知识点清单+交叉矩阵 |
| `PPT_OUTLINE.md` (280行) | 30张幻灯片大纲+配图建议 |
| `SUMMARY.md` (145行) | 项目概述+核心结论+文件清单 |
| `PLAN_CCSDS_THEORY.md` (187行) | CCSDS对接+理论分析实施计划 |
| `README.md` | 项目说明 |

---

## 五、数据流全景

```
┌─────────────────────────────────────────────────────────────┐
│                    一次实验的数据流                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  telemetry.py ──→ channel.py ──→ retrieval.py ──→ reconstruction.py
│  生成y_original    加噪声+衰落     找最佳模板         残差+重建
│       │                                 │                │
│       └──→ knowledge.py ←──────────────┘                │
│            存入KB模板            从KB检索模板              │
│                                                           │
│  metrics.py ←── y_original vs y_recon → MSE               │
│  visualization.py ←── 聚合结果 → PNG图表                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```
