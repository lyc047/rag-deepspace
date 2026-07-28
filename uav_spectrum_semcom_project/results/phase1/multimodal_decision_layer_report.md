# Multimodal Decision Layer Report

## 1. 目标

本阶段正式实现“频谱状态驱动 UAV 图像语义传输”的决策层。

系统输入包括：

```text
频谱侧：
clean rate、packet loss、BER、频谱资源选择方案

图像侧：
视觉目标语义 bits、JPEG bits、ROI bits、视觉优先级、ROI 面积比例
```

系统输出是图像传输动作：

| 动作 | 含义 |
|---|---|
| summary_only | 只传状态摘要 |
| semantic_only | 只传视觉语义框 |
| semantic_plus_roi | 传视觉语义 + ROI 图像块 |
| lowres_plus_semantic | 传视觉语义 + 低分辨率图像 |
| jpeg_full | 传完整 JPEG |

当前实现的是规则决策层，不是强化学习。后续 DQN 可以基于同一状态、动作和效用函数继续扩展。

## 2. 已实现策略

当前对比策略包括：

| 策略 | 含义 |
|---|---|
| summary_only | 永远只传摘要 |
| semantic_only | 永远只传视觉语义 |
| roi_always | 永远传语义 + ROI |
| lowres_always | 永远传语义 + 低分辨率图 |
| jpeg_always | 永远传完整 JPEG |
| spectrum_rule | 根据频谱状态、链路质量和视觉优先级自适应选择 |
| oracle_policy | 根据效用函数选择最优动作，作为上界 |

## 3. 效用函数

当前效用函数形式为：

```text
utility =
expected_task_value
- bit_cost_per_mbit × transmitted_mbits
```

其中 `expected_task_value` 同时考虑：

- 视觉语义是否成功送达；
- 图像细节是否成功送达；
- 是否是高优先级图像帧；
- 当前频谱 clean rate；
- packet loss 和 BER 对大载荷传输的影响。

因此，决策层不是单纯追求传得越多越好，而是在任务收益和通信开销之间折中。

## 4. 代表性结果

在 `semantic_hard_rep3` 频谱语义上报下，规则策略会随 packet loss 自动改变行为。

| Packet loss | Policy | bits/frame | Utility | Priority detail | Main action |
|---:|---|---:|---:|---:|---|
| 0.00 | semantic_only | 5305.6 | 1.7804 | 0.0000 | semantic_only |
| 0.00 | spectrum_rule | 48748.5 | 3.7905 | 0.9028 | lowres/ROI |
| 0.05 | semantic_only | 5305.6 | 1.6785 | 0.0000 | semantic_only |
| 0.05 | spectrum_rule | 44761.9 | 2.3317 | 0.5339 | lowres/ROI |
| 0.10 | semantic_only | 5305.6 | 1.5830 | 0.0000 | semantic_only |
| 0.10 | spectrum_rule | 15998.3 | 1.7118 | 0.1741 | semantic + partial detail |
| 0.20 | semantic_only | 5305.6 | 1.4102 | 0.0000 | semantic_only |
| 0.20 | spectrum_rule | 5445.9 | 1.4182 | 0.0074 | mostly semantic_only |

这个结果说明：

```text
链路较好时，规则策略会主动传更多图像细节；
链路变差时，规则策略自动退回低开销语义传输。
```

这正是频谱感知多模态语义通信需要体现的核心行为。

## 5. 当前结论

当前项目已经从：

```text
频谱模块 + 图像模块
```

推进到：

```text
频谱状态驱动图像语义传输决策
```

这使系统真正具备多模态语义通信闭环。

## 6. 下一步

下一步可以在当前决策层基础上继续做：

1. DQN 决策层；
2. 不同奖励权重消融；
3. 真实视觉检测器输出替代 ground-truth 视觉语义；
4. ROI 图像块实际裁剪与压缩；
5. 多 UAV 多图像任务队列仿真。

相关文件：

- `scripts/simulate_multimodal_decision_policy.py`
- `results/phase1/multimodal_decision_policy.md`
- `results/phase1/multimodal_decision_policy.csv`
- `results/phase1/multimodal_decision_policy_tradeoff.png`
