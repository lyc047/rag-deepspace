# Spectrum System Completion Report Before Visual-Semantic Expansion

## 1. 当前状态

在扩展 UAV 图像语义之前，频谱系统已经完成如下闭环：

```text
RadDet 时频观测
→ 频谱占用语义检测
→ hard / soft 语义包上报
→ 丢包 / 误码链路仿真
→ 多 UAV 频谱状态融合
→ 连续频谱资源块选择
→ clean rate / regret / net utility / map MAE 评估
```

本报告汇总了扩展图像语义之前补充完成的四项工作：

1. 公平大载荷传输基线；
2. packet loss / BER 链路条件 sweep；
3. 多 UAV 频谱地图融合评估；
4. 频谱闭环性能整理与图表输出。

## 2. 公平大载荷传输基线

原始 `spectrogram8_gate` 和 `iq12_gate` 是保守 all-or-nothing 模型，即大载荷必须完整到达才能使用。为了避免对完整频谱图/IQ 基线不公平，当前新增了 partial recovery 基线：

```text
spectrogram8_partial
spectrogram8_rep3_partial
iq12_partial
```

其核心假设是：

- 频谱图/IQ 被拆成多个 packet；
- packet 到达且无误码后可贡献部分信息；
- 若恢复比例超过最低阈值，则保留一部分检测框；
- 默认 FEC overhead = 1.25；
- 默认最低恢复比例 = 0.25。

这比原 all-or-nothing gate 更有利于大载荷传输，因此更适合作为公平对比。

## 3. 代表性资源优化结果

默认场景：

- 4 架 UAV 同时观测；
- 1000 个决策时刻；
- 频谱轴切成 4 个候选子信道；
- 业务需要连续 2 个子信道；
- packet loss = 0.2；
- BER = 1e-4；
- switch cost = 0.005。

| 方法 | Clean rate | Net utility | Regret | bits/decision-step |
|---|---:|---:|---:|---:|
| random | 0.6968 | 0.9787 | 0.0141 | 0 |
| fixed_ch0 | 0.7006 | 0.9812 | 0.0150 | 0 |
| spectrogram8_gate | 0.6884 | 0.9784 | 0.0145 | 524288 |
| spectrogram8_partial | 0.8788 | 0.9890 | 0.0040 | 655360 |
| spectrogram8_rep3_partial | 0.9310 | 0.9915 | 0.0014 | 1966080 |
| iq12_partial | 0.8746 | 0.9887 | 0.0042 | 120000000 |
| semantic_hard | 0.8858 | 0.9894 | 0.0035 | 1045.9 |
| semantic_hard_rep3 | 0.9308 | 0.9917 | 0.0012 | 3137.7 |
| semantic_soft_rep3 | 0.9308 | 0.9916 | 0.0013 | 3907.3 |
| oracle | 0.9440 | 0.9930 | 0 | 0 |

关键结论：

- `semantic_hard_rep3` 的 clean rate 为 0.9308，接近 oracle 的 0.9440；
- `spectrogram8_rep3_partial` 的 clean rate 也达到 0.9310，但需要约 1.97 Mbit/decision-step；
- `semantic_hard_rep3` 只需要约 3.14 kbit/decision-step；
- 因此二者资源选择效果接近，但语义方法比重复保护频谱图小约 626 倍；
- 相比 `iq12_partial`，语义方法开销小约 38243 倍，同时 clean rate 更高。

这说明语义方法不是只依赖保守大载荷基线获胜；即使引入更公平的 partial/FEC-like 大载荷恢复，语义方法仍有明显开销优势。

## 4. 链路条件 sweep

在 4 UAV 压力场景下，对 packet loss 和 BER 做 sweep。代表性结果如下，固定 BER = 1e-4：

| Packet loss | Random clean | Semantic hard rep3 clean | Spectrogram partial clean | I/Q partial clean | Semantic hard rep3 net utility |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6928 | 0.9394 | 0.9233 | 0.9194 | 0.9918 |
| 0.03 | 0.7094 | 0.9339 | 0.9072 | 0.9144 | 0.9916 |
| 0.05 | 0.6933 | 0.9333 | 0.9067 | 0.8994 | 0.9917 |
| 0.1 | 0.6928 | 0.9267 | 0.8989 | 0.8850 | 0.9916 |
| 0.2 | 0.6856 | 0.9217 | 0.8644 | 0.8583 | 0.9904 |
| 0.3 | 0.7094 | 0.9328 | 0.8617 | 0.8561 | 0.9916 |

结论：

- random clean rate 约 0.69–0.71，基本不能利用频谱状态；
- `semantic_hard_rep3` 在 packet loss 最高到 0.3 时仍保持约 0.93 clean rate；
- partial 频谱图/IQ 基线明显强于 gate 基线，但在高丢包下仍低于语义重复保护；
- 语义方案在多种链路扰动下保持了稳定资源选择质量。

## 5. 多 UAV 压力 sweep

在 packet loss = 0.2、BER = 1e-4 下改变同时观测 UAV 数量：

| UAVs | Random clean | Semantic hard | Semantic hard rep3 | Oracle |
|---:|---:|---:|---:|---:|
| 1 | 0.9142 | 0.9800 | 0.9974 | 0.9992 |
| 2 | 0.8370 | 0.9586 | 0.9828 | 0.9882 |
| 4 | 0.6846 | 0.8844 | 0.9284 | 0.9396 |
| 8 | 0.4306 | 0.6894 | 0.7316 | 0.7554 |

结论：

- UAV 数量越多，频谱压力越大；
- random/fixed 基线性能快速下降；
- 语义方法始终更接近 oracle；
- 8 UAV 时，semantic hard rep3 相比 random 的 clean rate 绝对提升约 0.301。

## 6. 频谱地图融合评估

为了评估资源选择之前的中间层，本阶段新增了频谱地图融合评价。指标包括：

- channel occupancy MAE；
- block occupancy MAE；
- clean-block accuracy；
- bits/decision。

在 packet loss = 0.2、BER = 1e-4 下：

| UAVs | Scheme | Block MAE | Clean-block accuracy | bits/decision |
|---:|---|---:|---:|---:|
| 1 | semantic_hard | 0.0051 | 0.9402 | 262.1 |
| 1 | semantic_hard_rep3 | 0.0053 | 0.9461 | 786.4 |
| 1 | spectrogram8_partial | 0.0050 | 0.9385 | 163840 |
| 4 | semantic_hard | 0.0179 | 0.7944 | 1047.3 |
| 4 | semantic_hard_rep3 | 0.0206 | 0.7904 | 3142.0 |
| 4 | spectrogram8_partial | 0.0172 | 0.8102 | 655360 |
| 8 | semantic_hard | 0.0334 | 0.7422 | 2099.2 |
| 8 | semantic_hard_rep3 | 0.0413 | 0.7348 | 6297.6 |
| 8 | spectrogram8_partial | 0.0313 | 0.7413 | 1310720 |

需要注意：

- 地图 MAE 上，partial 频谱图通常略优于语义框，因为它模拟了更多原始频谱信息恢复；
- 但其开销远高于语义方法；
- 资源选择结果中，`semantic_hard_rep3` 仍然表现很好，说明资源选择任务不一定需要完整重建频谱地图，只需要足够支持“选哪个资源块”。

这正是任务导向语义通信的核心思想：不追求还原全部数据，而追求完成下游任务。

## 7. 当前性能评估

当前频谱系统可以给出如下阶段性性能结论：

| 指标 | 当前结果 |
|---|---:|
| 4 UAV、PL=0.2、BER=1e-4 下 semantic_hard_rep3 clean rate | 0.9308 |
| 同条件 oracle clean rate | 0.9440 |
| 同条件 random clean rate | 0.6968 |
| semantic_hard_rep3 bits/decision-step | 3137.7 |
| spectrogram8_rep3_partial bits/decision-step | 1966080 |
| iq12_partial bits/decision-step | 120000000 |
| 8 UAV 压力下 semantic_hard_rep3 相比 random clean rate 提升 | +0.301 |
| 测试通过数量 | 19 passed |

综合判断：

> 频谱系统在“扩展图像语义之前”的四项补强已经完成。当前系统不仅证明语义上报开销低，而且在多 UAV 压力、链路扰动、公平大载荷基线和频谱地图融合评价下，均表现出较好的任务导向资源优化价值。

## 8. 仍需谨慎表述的地方

1. RadDet 不是低空 UAV 实测通信频谱数据，而是时频目标检测数据；
2. 多 UAV 场景是采样叠加构造的压力场景，不是真实多 UAV 同步采集；
3. partial/FEC-like 大载荷基线仍是抽象模型，不是具体通信协议实现；
4. 当前语义以占用框为主，尚不可靠支持精细信号类别识别；
5. 频谱坐标目前是归一化坐标，后续如果接入真实频率单位，需要进一步校准。

## 9. 已生成文件

- 主资源优化仿真：`resource_optimization_simulation.md`
- 链路条件 sweep：`link_condition_resource_sweep.md`
- 链路 clean rate 曲线：`link_condition_clean_rate.png`
- 链路 net utility 曲线：`link_condition_net_utility.png`
- 多 UAV 压力 sweep：`multi_uav_pressure_sweep.md`
- 多 UAV clean rate 曲线：`multi_uav_pressure_clean_rate.png`
- 多 UAV regret 曲线：`multi_uav_pressure_regret.png`
- 频谱地图融合评估：`spectrum_map_fusion_eval.md`
- 频谱地图 MAE 曲线：`spectrum_map_fusion_block_mae.png`
- 主脚本：`scripts/simulate_resource_optimization.py`
- 链路 sweep 脚本：`scripts/sweep_link_conditions_resource.py`
- 地图融合脚本：`scripts/evaluate_spectrum_map_fusion.py`

## 10. 下一步

频谱系统已经具备进入图像语义扩展的基础。下一阶段可以开始做：

```text
UAV 图像语义检测 / 标注语义
→ 视觉语义重要性评估
→ 频谱状态驱动视觉传输粒度选择
→ 频谱语义 + 视觉语义联合上报
→ 多模态任务性能 / 通信开销联合评估
```
