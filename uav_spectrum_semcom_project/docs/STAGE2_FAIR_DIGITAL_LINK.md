# 硕士论文阶段 2：公平数字上报链路与任务级验证

> 状态：第一版已完成并通过波形级一致性校验  
> 上游约束：`docs/THESIS_STAGE0_SCOPE.md`、`docs/STAGE1_REPRODUCIBILITY.md`  
> 正式配置：`configs/stage2_digital_link.json`

## 1. 研究目的

阶段 2 解决旧链路中“语义包使用重复保护，而谱图/IQ 裸传且要求整帧零错误”的不公平问题。所有源表示现在共享同一套：

```text
应用载荷
→ 统一分包与链路头
→ CRC-16/CCITT
→ Hamming(7,4)
→ QPSK
→ AWGN/衰落信道
→ CRC 检错
→ 有限 ARQ
→ 统一决策时延预算
```

比较对象包括 hard semantic、量化 soft semantic、PNG 谱图、8-bit 谱图和理论 I/Q 容量参考。总发送 bit 按下式审计：

$$
B_{\mathrm{tx}}=B_{\mathrm{payload}}+B_{\mathrm{header}}+B_{\mathrm{CRC}}+B_{\mathrm{FEC}}+B_{\mathrm{padding}}+B_{\mathrm{retransmission}}.
$$

## 2. 双层链路设计

### 2.1 解析包级主实验

大规模实验按包计算调制 BER、FEC 后包成功概率、CRC 失败、重传、实际发送 bit 和链路时延。该层可处理 24 Mbit I/Q 容量参考，不需要对每个 IQ bit 逐一生成波形。

Hamming(7,4) 的包成功概率按每个码字至多发生一个错误计算：

$$
P_{cw}=(1-p)^7+7p(1-p)^6,
\qquad
P_{pkt}=P_{cw}^{\lceil L/4\rceil}.
$$

### 2.2 波形级校验

小规模校验实际生成随机比特，执行 CRC、FEC、BPSK/QPSK/16QAM、AWGN/Rayleigh/Rician 块衰落、硬解调和译码。当前正式 QPSK/AWGN 校验中，解析包成功率与 2,000 次波形 Monte Carlo 的绝对误差为：

| Eb/N0 | 解析成功率 | 波形成功率 | 绝对误差 |
|---:|---:|---:|---:|
| 0 dB | 0.0018 | 0.0015 | 0.0003 |
| 3 dB | 0.5412 | 0.5385 | 0.0027 |
| 6 dB | 0.9929 | 0.9930 | 0.0001 |
| 9 dB | 1.0000 | 1.0000 | 0.0000 |

这表明解析层足以用于后续大载荷扫描。Eb/N0 定义为每个已发送编码 bit 的能量比，FEC 通过增加发送 bit 和能量换取可靠性，不引入“免费编码增益”。

## 3. 冻结公平协议

- 应用载荷：1,024 bit/packet；
- 链路头：96 bit/packet；
- CRC：16 bit；
- FEC：Hamming(7,4)；
- 调制：QPSK；
- 符号率：1 Mbaud；
- 最大重传：1 次；
- 决策时延预算：0.25 s；
- Eb/N0：0、3、6、9、12 dB；
- 主链路重复次数：30；
- 统计量：均值、样本标准差、95% 置信区间。

Hamming(7,4) 是透明、可审计的研究基线，不代表 5G 级信道编码。论文最终系统应进一步增加 LDPC 或 Polar 对照。

## 4. 链路级主要结论

在 6 dB 时，hard/soft semantic 均能在约 1.4 ms 内完整送达，实际发送量约为 658/770 bit；PNG 谱图约需 233 ms 和 226 kbit，完整送达率约为 0.8；8-bit 谱图因协议/FEC/时延开销不能在 0.25 s 内完整发送；24 Mbit I/Q 只能发送约 0.5% 的应用载荷。

该结果只能证明时延和码率可行性，不能单独证明任务准确率。部分谱图或 I/Q bit 到达不能自动等价为可用感知信息。

## 5. 任务级适配

任务级实验将链路连接到冻结的 RadDet occupancy detector 和资源选择器：

- hard/soft semantic：只有相关应用头和语义单元所在分包通过 CRC 时才恢复该语义；
- PNG：任意分片缺失时不能解码，不输出检测结果；
- 8-bit 谱图：按行优先分片，收到的字节原样恢复，缺失字节用接收字节均值进行显式 concealment，再运行冻结检测器；
- I/Q：本地 RadDet 没有原始 I/Q，因此不伪造任务级检测结果，只保留容量参考。

前 200 个单帧样本过于稀疏，资源 regret 几乎为零。正式阶段 2 诊断因此使用每 4 个无关观测构成一个资源压力决策。它只用于 H1/H4 链路压力验证，不用于证明 H3 多 UAV 空间协同。

代表性结果：

| Eb/N0 | 方法 | F1 | Clean rate | Regret | bit/decision | 并行链路时延 |
|---:|---|---:|---:|---:|---:|---:|
| 3 dB | hard semantic | 0.4266 | 0.8440 | 0.004155 | 4.38 kbit | 2.86 ms |
| 3 dB | soft semantic | 0.3461 | 0.8120 | 0.006137 | 5.26 kbit | 3.15 ms |
| 3 dB | PNG 谱图 | 0.0000 | 0.6600 | 0.014281 | 970.14 kbit | 249.37 ms |
| 6 dB | hard semantic | 0.5495 | 0.9400 | 0.000031 | 2.70 kbit | 1.51 ms |
| 6 dB | 8-bit 分片谱图 | 0.4786 | 0.9400 | 0.000026 | 970.14 kbit | 249.37 ms |

这些数值支持一个阶段性结论：在当前冻结 detector、0.25 s 预算和受控 AWGN 条件下，语义上报能以显著较小的实际数字链路开销维持资源选择性能。它尚不是最终 H1/H4 结论，因为测试规模、信道模型和谱图分片恢复方法仍需扩展。

## 6. 学术边界与未完成项

1. 当前 H1/H4 结果是受控压力验证，不是关联多 UAV 协同证据；
2. 需要在完整冻结测试集上运行，而非只使用 200 帧；
3. 需要加入 Rician、Rayleigh、突发错误和反馈时延；
4. 需要对 8-bit 谱图比较严格整帧、均值填充、独立 tile 和擦除感知训练；
5. 需要加入量化能量/PSD soft fusion 基线；
6. 需要增加 LDPC/Polar 或标准 BLER 曲线，避免 Hamming 基线过于理想化；
7. IQ 任务结果必须等待真实 IQ 数据，不能用理论码长替代；
8. 阶段 3 的关联多节点数据建立后，才能正式验证 H3。

## 7. 复现命令

运行链路和波形校验：

```powershell
python scripts/run_stage2_digital_link.py
```

运行任务级适配：

```powershell
python scripts/evaluate_stage2_task_link.py
```

运行测试：

```powershell
python -m pytest
```

结果分别位于：

```text
results/stage2/digital_link/
results/stage2/task_link/
```

所有正式结果保存配置校验值、数据 frame ID 校验值、模型权重校验值、环境版本、逐次试验 CSV、统计汇总和机器可读 JSON。

## 8. 阶段 2 增强：衰落鲁棒性与 Pareto 曲线

增强实验入口：

```powershell
python scripts/sweep_stage2_robustness_pareto.py
```

该实验冻结 detector、分包、CRC、Hamming(7,4)、QPSK、ARQ 和资源任务，只改变上报信道或所有方案共享的时延预算。

### 8.1 上报信道鲁棒性

在 100 个冻结测试帧、每 4 观测一个压力决策和 10 次重复下，对 AWGN、Rayleigh 和 Rician 块衰落进行扫描。代表结果如下：

| 信道 | Eb/N0 | 方法 | Clean rate | Regret | bit/decision |
|---|---:|---|---:|---:|---:|
| AWGN | 6 dB | hard semantic | 0.9600 | 0.000051 | 2.75 kbit |
| AWGN | 6 dB | PNG 谱图 | 0.9120 | 0.002641 | 920.75 kbit |
| Rayleigh | 6 dB | hard semantic | 0.9240 | 0.000854 | 3.84 kbit |
| Rayleigh | 6 dB | PNG 谱图 | 0.6400 | 0.013573 | 970.14 kbit |
| Rician | 6 dB | hard semantic | 0.9400 | 0.000559 | 3.36 kbit |
| Rician | 6 dB | PNG 谱图 | 0.6400 | 0.013573 | 970.14 kbit |

结果表明短语义包在块衰落下也会退化，但有限 ARQ 能以较小额外 bit 保持较高任务性能；大 PNG 载荷即使收到较高比例的分包，也可能因没有任何一帧完整通过而无法运行 detector。这里的结论依赖“PNG 必须完整解码”这一明确 codec 属性，不可外推到独立 tile 或擦除码谱图。

### 8.2 统一时延预算 Pareto 扫描

在 AWGN、6 dB 下，将所有方案的共同预算设为 5、25、100 和 250 ms。hard semantic 在 5 ms 内已达到 clean rate 0.96，约需 2.74 kbit/decision；PNG 和行优先分片谱图在 100 ms 以内不能恢复有效检测结果，250 ms 时才接近冻结 detector 性能，发送量约为 0.92--0.97 Mbit/decision。

Pareto 图位于：

```text
results/stage2/enhanced/stage2_rate_task_pareto.png
```

完整逐次试验、置信区间和机器记录位于 `results/stage2/enhanced/`。该增强实验仍是 100 帧诊断结果；Rayleigh/Rician 假设接收端具有理想信道估计，尚未包含突发错误、估计误差和关联多节点观测。

## 9. 阶段 2 传统基线：量化 PSD 与独立谱图 tile

新增传统基线入口：

```powershell
python scripts/evaluate_stage2_classical_baselines.py
```

量化 PSD 将每个观测压缩为 4 个候选子信道的平均灰度能量，经过同一数字链路后，在压力决策内对成功收到的节点能量取平均。该方法直接选择最低能量连续资源块，不产生目标框，因此不报告 F1。独立 tile 基线把谱图划分为 16×16 uint8 块，每个 tile 携带 32-bit 位置头并独立经过 CRC、FEC 和 ARQ；只有完整 tile 才恢复，之后重新运行冻结 detector。

100 帧诊断结果表明：

- AWGN 6 dB 时，hard semantic 约 2.74 kbit/decision，clean rate 为 0.96；
- 4-bit PSD 约 2.45 kbit/decision，但 clean rate 仅为 0.68；
- 8-bit PSD 约 2.55 kbit/decision，clean rate 为 0.60；
- 独立 tile 约 804 kbit/decision，恢复约 71.4% tile，clean rate 为 0.96，但 detector F1 只有约 0.289。

该结果说明：低开销本身不能证明任务语义有效；只上传粗粒度平均能量虽然与 hard semantic 的 bit 数接近，但无法保留足够的时频占用结构。tile 基线消除了整幅谱图 gate 的不公平性，同时表明局部恢复可以支持资源选择，但通信开销仍远高于目标语义。

### 9.1 PSD 量化位数的防测试集调参

量化位数使用独立验证集选择：

```powershell
python scripts/select_stage2_psd_quantizer.py
```

预先声明候选集合为 1、2、3、4、6、8 bit/value，按“最小 regret → 最大 clean rate → 最低 bit”选择。验证集选择出 1 bit/value：验证 clean rate 为 0.80、regret 为 0.011455。冻结后在测试集只评估一次，clean rate 降至 0.64、regret 为 0.013573。

验证—测试差距说明简单灰度能量摘要存在明显分布敏感性，不能通过提高量化精度解决。8-bit 比 4-bit 更差并非链路错误，而是粗量化在当前代理数据上产生了类似正则化的排序变化。论文中应将其作为经典 soft-information 基线的失效边界，而不是选择测试集上表现最好的位宽。

完整结果位于：

```text
results/stage2/classical_baselines/
results/stage2/psd_quantizer/
```

## 10. 完整冻结测试目录主实验

完整集入口：

```powershell
python scripts/run_stage2_full_test.py --device auto
```

本地冻结目录名为 `test`，实际包含 20,001 帧。完整实验使用全部帧，其中 20,000 帧形成 5,000 个四观测压力决策，最后 1 帧按预先规则丢弃。PSD 位宽固定为独立验证集选择的 1 bit/value；测试集不再选择 detector 阈值、量化位宽、FEC、ARQ、包长、信道条件或时延预算。

完整集结果：

| Eb/N0 | 方法 | Clean rate | Regret | bit/decision | 并行链路时延 |
|---:|---|---:|---:|---:|---:|
| 3 dB | frozen 1-bit PSD | 0.7018 | 0.015494 | 3.62 kbit | 2.63 ms |
| 3 dB | hard semantic | 0.8402 | 0.006046 | 4.34 kbit | 2.82 ms |
| 6 dB | frozen 1-bit PSD | 0.6994 | 0.015757 | 2.32 kbit | 1.39 ms |
| 6 dB | hard semantic | 0.9289 | 0.000666 | 2.70 kbit | 1.51 ms |
| 9 dB | frozen 1-bit PSD | 0.6994 | 0.015757 | 2.30 kbit | 1.34 ms |
| 9 dB | hard semantic | 0.9290 | 0.000661 | 2.66 kbit | 1.44 ms |

在 6 dB 时，hard semantic 相比冻结 PSD 仅增加约 16.4% 发送 bit，却将 clean rate 绝对提高 0.2295，并将平均 occupancy regret 降低约 95.8%。在 3 dB 时，hard semantic 仍将 clean rate 提高 0.1384，regret 下降约 61.0%。因此阶段 2 现在可以在完整冻结本地测试目录上支持 H1/H4 的受控结论：任务语义相对粗粒度经典 PSD 摘要具有更好的任务性能—数字链路开销折中。

该结论仍受以下边界约束：本地非标准 split 数量的上游来源尚未核验；压力决策由无关帧构成，不证明 H3；RadDet 是合成雷达数据；PSD 不是校准 dBm；Hamming(7,4) 不是标准 5G 编码。

首次完整集运行会生成冻结特征缓存：

```text
results/stage2/full_test/frozen_feature_cache.json
```

后续相同 detector、split 和帧顺序的链路重复实验直接读取缓存。完整主表和复现记录位于 `results/stage2/full_test/`。

## 11. 实际 LDPC 波形与完整任务验证

安装研究依赖并运行：

```powershell
python -m pip install -e ".[research]"
python scripts/validate_stage2_ldpc.py
```

本实验使用 `pyldpc 0.7.9` 生成固定规则 LDPC：$n=1200$、$k=602$、码率约 0.5017、$d_v=3$、$d_c=6$。每个信息块包含 96-bit 应用头和 CRC-16，可承载 490 个应用 bit。LDPC 与 Hamming 对照都使用单位能量 BPSK/AWGN，Eb/N0 定义为每个实际发送编码 bit 的能量比；低码率通过增加发送 bit 和能量换取可靠性，没有人为平移 SNR。

实际波形译码结果：

| Eb/N0 | LDPC 包成功率 | Hamming 包成功率 |
|---:|---:|---:|
| 1 dB | 0.18 | 0.00 |
| 2 dB | 0.76 | 0.02 |
| 3 dB | 1.00 | 0.16 |
| 4 dB | 1.00 | 0.62 |
| 5 dB | 1.00 | 0.84 |
| 6 dB | 1.00 | 1.00 |

每点使用 50 次真实 BP/CRC 试验、最多 20 次译码迭代，因此采用 Wilson 95% 区间；该曲线用于验证趋势和接入任务，不应解释为高精度标准 BLER 曲线。

将实测 LDPC 包成功率接入 20,001 帧冻结任务缓存后：

| Eb/N0 | 方法 | Clean rate | Regret | bit/decision | 并行时延 |
|---:|---|---:|---:|---:|---:|
| 2 dB | frozen PSD + LDPC | 0.6998 | 0.015699 | 5.95 kbit | 3.75 ms |
| 2 dB | hard semantic + LDPC | 0.9161 | 0.001431 | 6.05 kbit | 3.89 ms |
| 3 dB | frozen PSD + LDPC | 0.6994 | 0.015757 | 4.80 kbit | 2.25 ms |
| 3 dB | hard semantic + LDPC | 0.9290 | 0.000661 | 4.87 kbit | 2.37 ms |

2 dB 时两种表示的发送量只相差约 1.7%，hard semantic 将 clean rate 提高 0.2163，并将 regret 降低约 90.9%。这说明此前语义优势不是 Hamming(7,4) 特有现象；在更强的实际 LDPC/BP 链路下，任务表示本身仍决定资源选择质量。

学术边界：该码是实际 LDPC，但不是 3GPP NR 基图、速率匹配或 HARQ 实现；完整任务使用实测包成功曲线进行可扩展 Monte Carlo，而不是对每个任务包重复 BP；AWGN 假设理想同步与信道知识，尚未测量译码能耗和时延。

结果位于：

```text
results/stage2/ldpc_validation/
```

## 12. 阶段 2 收尾：强 PSD 与完整集非理想链路

为避免将 4 个候选信道的平均灰度、1-bit 量化 PSD 误作唯一传统基线，新增了
验证集选择的高分辨率 PSD 候选：每候选信道 2/4/8 个 sub-bin、1/2-bit
量化，以及 mean/max/p75 与本地 CFAR 摘要。候选仅在验证集按最低 regret、
最高 clean rate、最低源码 bit 选择；完整测试集不再调参。最终冻结为 8
sub-bin/信道的本地 CFAR、1-bit 上报。它仍是归一化谱图强度代理，而非校准
dBm PSD。

在完整 20,001 帧冻结测试集上，该强 PSD 与 hard semantic 在相同链路规则下
运行于 AWGN、Rayleigh、Rician 和 Gilbert--Elliott 突发误码模型。代表结果：
AWGN 3 dB 下语义/强 PSD 的 clean rate 为 0.8405/0.7048，regret 为
0.006062/0.015052；突发模型 6 dB、共同 25 ms 下为 0.9257/0.7055 与
0.000849/0.014994。完整表、验证候选和逐次试验位于
`results/stage2/completion/`。

LDPC 的 Python/pyldpc 原型另测得约 605 ms/codeword（2 dB）和 504
ms/codeword（3 dB）的 CPU wall-clock 编码加 BP 译码时间（20 个试验）。该
数值用于说明原型复杂度边界，并非机载实时性声明。阶段 2 最终可引用的统一
结论、边界和复现命令见 `docs/STAGE2_FINAL_REPORT.md`。
