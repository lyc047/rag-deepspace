# 阶段 2 最终报告：公平数字链路与资源选择验证

## 已冻结的问题与公平规则

阶段 2 验证的不是“谁更容易把数据包送达”，而是在相同分包、96-bit
协议头、CRC-16、FEC、重传上限、调制方式和时延预算下，哪一种上报表示能
更可靠地支持连续空闲频谱块选择。所有结果均报告实际发送 bit（含保护、填充
与重传）以及下游 clean rate 和 occupancy regret。

## 主证据

完整冻结 `test` 目录含 20,001 帧，其中 20,000 帧按预注册规则构成 5,000
个四观测压力决策。AWGN 6 dB 下，hard semantic 为 clean rate 0.9289、
regret 0.000666、2.70 kbit/decision；原始冻结 1-bit PSD 为 0.6994、
0.015757、2.32 kbit/decision。实际规则 LDPC/BP 在 2 dB 下仍给出语义
0.9161 clean rate 与 0.001431 regret，而 PSD 为 0.6998 与 0.015699；
二者发送量仅相差约 1.7%。

为避免弱PSD对照，新增了验证集选择的高分辨率 PSD 候选（2/4/8 个
sub-bin/候选信道、1/2-bit 量化、mean/max/p75 与 CFAR 摘要）。验证集按
最低 regret、最高 clean rate、最低源码 bit 的顺序冻结为：8 sub-bin/信道、
本地 CFAR、1-bit 上报。它在完整测试集的 AWGN 3 dB 下达到 clean rate
0.7048、regret 0.015052、3.60 kbit/decision，仍低于 hard semantic 的
0.8405、0.006062、4.34 kbit/decision。

## 非理想链路与工程成本

完整冻结集的鲁棒性结果如下；Rayleigh/Rician 采用理想接收均衡，burst 为
显式 Gilbert--Elliott 包尝试级模型，25 ms 是共同预算。

| 条件 | hard semantic clean / regret | 强 PSD clean / regret |
|---|---:|---:|
| AWGN 3 dB, 250 ms | 0.8405 / 0.006062 | 0.7048 / 0.015052 |
| Rayleigh 6 dB, 250 ms | 0.8919 / 0.002876 | 0.7060 / 0.014932 |
| Rician 6 dB, 250 ms | 0.9139 / 0.001557 | 0.7065 / 0.014961 |
| Burst 6 dB, 25 ms | 0.9257 / 0.000849 | 0.7055 / 0.014994 |

规则 LDPC（n=1200, k=602, rate≈0.502, BP 最大20次）的实际 CPU
wall-clock 编码加译码代价为：2 dB 约605 ms/codeword、3 dB 约504
ms/codeword（20 个试验的开发机测量）。这不是机载实时性结论，但明确显示
Python/pyldpc 原型不能直接等同于无人机实时实现；需采用编译型译码器、并行
硬件或更小码长后才能进入部署讨论。

## 可成立结论

在当前冻结 detector、合成 RadDet、四观测压力任务和受控数字链路下，任务
语义上报相对于粗PSD及验证集选择的高分辨率CFAR-PSD，均提供更好的
bit--clean-rate / bit--regret 折中。该结论在 AWGN、块衰落、显式突发误码和
实际 LDPC/BP 参考链路中均未被否定，因而可作为 H1/H4 的阶段性证据。

## 严格边界与下一阶段

- PSD 是归一化谱图强度代理，不是校准 dBm 接收功率；
- RadDet 是合成雷达数据，并非真实无人机采集；
- 四帧压力聚合不构成同场景空间协同，不能证明 H3；
- LDPC 不是 3GPP NR 基图、速率匹配或 HARQ；burst 也仍是模型，不是外场链路；
- 阶段三应使用时间连续、同场景多节点的真实/半实物观测，研究质量感知融合与预测。

## 复现入口

```powershell
python scripts/run_stage2_full_test.py --device auto
python scripts/validate_stage2_ldpc.py
python scripts/run_stage2_completion.py --device cpu
python scripts/profile_stage2_ldpc.py --trials 20
```

结果分别保存于 `results/stage2/full_test/`、`ldpc_validation/` 与
`completion/`。
