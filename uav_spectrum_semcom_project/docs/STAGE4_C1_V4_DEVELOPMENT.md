# C1-v4：面向连续频谱块选择的任务充分语义

> 日期：2026-07-22  
> 当前性质：开发协议、真实缓存、两轮开发实验与统计均已完成  
> 结论边界：尚无 C1-v4 性能结果，不得写成已验证创新

## 1. 为什么不继续增加 C1-v3 尾部权重

C1-v3 在新校准数据上平均功率 regret 略有改善，但在300景时域确认中平均和 CVaR 均略差于固定4-bit基线。事后分析表明，两种方法最坏30景重合28景，且174/300景的资源决策完全相同。主要矛盾不是 CVaR 权重不足，而是训练目标与最终任务不一致：模型在视觉/occupancy 域优化块排序，确认指标却是连续 dBm 功率代价。阈值化 occupancy 丢失的功率次序无法通过继续调整损失权重恢复。

C1-v4 因此不再回答“怎样把 occupancy 概率修得更好”，而是回答：

> 在真实数字链路和固定协议开销下，发送连续资源块选择所需的任务充分统计量，能否同时降低资源 regret 和实际发送bit？

## 2. 冻结的五种表示

2.4 GHz ISM 子带仍划分为8个等宽信道，资源需求为连续4信道，因此只有5个候选连续块。

| 表示 | 数值数量×位数 | 应用层总bit | 接收端决策 | 角色 |
|---|---:|---:|---|---|
| `occupancy_fixed4` | 8×4 | 184 | 对8信道占用做4点滑动平均 | 原任务代理基线 |
| `channel_power_fixed4` | 8×4 | 184 | 对归一化信道功率做滑动平均 | 经典直接功率基线 |
| `block_score_fixed4` | 5×4 | 172 | 在5个连续块代价中取最小 | 核心任务充分表示 |
| `block_score_fixed6` | 5×6 | 182 | 同上 | 近等码率高精度对照 |
| `best_block_indicator1` | 5×1 | 157 | 取0值所在的最优块 | 决策下界/极简命令基线 |

每种方法都使用相同的152-bit应用层固定开销：69-bit版本化头部和83-bit质量兼容区。资源类型、节点、scene tag、值数量和量化位数都进入真实bitstream；没有用“向量维度”代替实际通信开销。

数据包经过既有 Stage-2 分包、CRC、Hamming(7,4)、调制、AWGN/Rayleigh/Rician、有限重传与时延逻辑。CRC成功才使用量化语义，失败统一回退到 block start 0。不同表示使用共同随机数，避免信道随机性污染配对差异。

## 3. 真实数据开发协议

从 AERPAW 三个固定站点的原始 ZIP 选择开发场景，先永久排除：

- 20个原 LW1 pilot；
- 原200景三站 Final 的全部600个站点 provenance；
- C1-v3 的60景时域校准和300景确认来源。

随后按整日而非随机逐景拆分：

| 分区 | 本地时间 | 每站场景 | 总场景 | 用途 |
|---|---|---:|---:|---|
| train | 2022-02-08 12:00 至 02-10 00:00 | 90 | 270 | 后续学习型扩展预留 |
| calibration | 2022-02-10全天 | 60 | 180 | 参数/表示校准 |
| validation | 2022-02-11全天 | 60 | 180 | 冻结表示的一次开发比较 |

每站先做15分钟贪心稀疏化，再等距选样；统计重采样单位为“站点×本地小时”。这仍是开发数据，不是未来独立确认集。

## 4. 预注册评价与选择规则

开发链路覆盖3种信道、0/3/6/9 dB Eb/N0，每组合10次重复。报告：

- clean resource regret；
- 数字链路后的平均 regret 和 CVaR0.9；
- 应用层bit、实际发送bit和帧成功率；
- 相对 `occupancy_fixed4` 的10,000次 paired cluster bootstrap。

候选只有同时满足以下条件才可进入未来独立确认：

1. 平均 regret 差的单侧95%分组 bootstrap 上界不大于0；
2. 实际发送bit差的单侧95%上界严格小于0。

若多个候选通过，依次按平均 regret、CVaR、实际bit、应用层bit选择。验证结果出来后不得增加第六种表示或修改准入规则。

## 5. 与近期工作的关系

近期目标导向语义通信研究强调应按最终控制/任务误差而非重构误差设计压缩层级，并联合速率自适应；另有工作采用离线语义效用与量化收益表降低在线分配复杂度。[Pan 等的闭环通信—感知—控制框架](https://arxiv.org/abs/2512.19177)和[Lei 等的 importance-aware 量化资源分配](https://arxiv.org/abs/2606.29052)支持“任务指标直接进入语义设计”的总体方向。协作频谱感知方面，[Yi 等的分布式语义通信与空中计算工作](https://doi.org/10.1109/TCOMM.2024.3468215)说明了频谱感知语义压缩的研究价值。

C1-v4 不复制这些论文的网络或优化器。其具体区别是：以连续块 dBm regret 为任务、显式构造5维块充分统计量、计入可执行数字包成本，并用尾部风险与时间块统计验证。

## 6. 当前完成情况

已完成：

- `c1_v4_block_semantics.py`：真实 `rf32_le` ZIP读取、4维信道特征、连续块代价、5类表示、量化、编解码及链路决策；
- `stage4_c1_v4_development_protocol.json`：数据排除、整日拆分、表示、链路和统计规则；
- `build_stage4_c1_v4_real_development_cache.py`：支持原盘符和 `SITE=PATH` 迁移覆盖的缓存构建器；
- `evaluate_stage4_c1_v4_representations.py`：真实数字链路配对评价、cluster bootstrap和候选选择；
- 全项目181项测试通过。

真实630景缓存和第一轮表示评价已经完成，`best_block_indicator1` 按预注册规则入选；`block_score_fixed4` 也满足平均 regret 不劣且实际bit更低的准入条件。由于 clean 映射误差已接近0，没有增加学习型校正器。

随后在新日期2月12日冻结并完成零比特“最近成功块”回退实验：平均 regret 降低28.48%，CVaR降低37.52%，实际bit严格不变。完整结果见 `docs/STAGE4_C1_V4_RESULTS.md`。

## 7. 复现实验执行方式

若仍挂载为 F 盘：

```powershell
python scripts/build_stage4_c1_v4_real_development_cache.py
python scripts/evaluate_stage4_c1_v4_representations.py
```

若盘符发生变化，例如 G 盘：

```powershell
python scripts/build_stage4_c1_v4_real_development_cache.py `
  --archive "LW1=G:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsLW1Feb2022_SigMF.zip" `
  --archive "CC1=G:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsCC1Feb2022_SigMF.zip" `
  --archive "CC2=G:\uav_spectrum_final_data\aerpaw_sub6_feb2022\raw\ResultsCC2Feb2022_SigMF.zip"
python scripts/evaluate_stage4_c1_v4_representations.py
```

盘符迁移时脚本会重新计算整个ZIP的 SHA-256；文件名或大小不匹配也会中止。

## 8. 下一决策

五类表示比较已经证明学习型 block-score 校正器没有必要。当前候选为“最优块指示 + 最近成功块回退”；较丰富的 `block_score_fixed4` 作为多需求/融合扩展保留。下一步是冻结陈旧时间保护规则，并在另一公开数据集或新采集批次上做一次独立确认。
