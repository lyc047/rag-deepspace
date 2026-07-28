# 阶段 4 Batch 9：实际解码路径端到端信道 sweep

日期：2026-07-16  
状态：已完成；冻结开发系统的多信道证据已生成  
范围：validation development evidence；final holdout 未建立、未访问

## 1. 协议

固定比较 `all_G1`、`G1 + 阶段3先验单节点G2`、`all_G2`。150 个 scene，AWGN/Rayleigh/Rician，global Eb/N0 为 0/3/6/9 dB，每 scene 内嵌套 5 次链路重复。节点具有固定相对 Eb/N0 偏置。

每条消息真实执行应用层量化与编码、分包、协议头、CRC、Hamming FEC、调制填充、截断 ARQ 和语义解码。融合只使用成功解码的每节点最新报告；完全无报告时使用 0.5 occupancy。bit 为实际发送 bit，时延采用保守串行上行求和。regret、clean、Brier 和 CVaR 均在真实解码后计算。

## 2. 主要结果

在中高 Eb/N0 下，单节点 G2 通常以显著少于全 G2 的 bit 获得相近均值任务性能。例如：

- AWGN 9 dB：单节点 G2 regret `0.025368`、bit `1863.3`；全 G2 `0.026258 / 2075.5`；
- Rayleigh 9 dB：`0.024930 / 2312.4` 对 `0.026221 / 2638.3`；
- Rician 9 dB：`0.025274 / 2030.1` 对 `0.026244 / 2319.3`。

低 Eb/N0 时报告交付显著下降，任务收益收缩。例如 AWGN 0 dB 的单节点 G2 报告交付率只有 0.0347，regret `0.027819`，与 all-G1 接近。

all-G1 的任务指标几乎不随交付率变化：四节点 1-bit 报告本来完全相同，而完全无报告的 0.5 fallback 在当前任务上常产生相同资源排序。这是一个重要反例：payload/report delivery ratio 不能替代解码后频谱选择性能。

## 3. 配对统计

先在每个 scene 内平均 5 次链路重复，再进行 10000 次 scene 配对 bootstrap。

- 单节点 G2 相对全 G2：12 个信道点的实际 bit 差 95% CI 全部低于 0，节省约 212—666 bit/decision；
- regret 差的 95% CI 在全部 12 个点都跨 0，均值有时更好、有时更差；
- 单节点 G2 相对 all-G1：regret 均值普遍降低，但所有 CI 仍跨 0，同时增加约 502—1024 bit。

因此当前可写结论是：“在 development split 上，选择性 G2 对全 G2 形成稳定的实际 bit 节省，未观察到稳定的 regret 差异。”不得把 CI 跨 0 改写为已经证明任务非劣。正式非劣结论需要预注册非劣界并在独立 final holdout 一次验证。

## 4. 结论

端到端实现和公平成本口径通过，但该 sweep 不改变 Gate B/C 决策。最终开发候选仍采用简单可解释调度/融合；唯一已有显著任务对齐算法证据的是 C1 resource-aware loss。所有信道点、任务指标和 bootstrap 已保存于 `results/stage4/end_to_end_sweep_v1/`。
