# Phase 1 Method Comparison

## 1. 当前方法对比

| 方法 | 设置 | Test Precision | Test Recall | Test F1 | 语义 payload |
|---|---|---:|---:|---:|---:|
| Tuned traditional STFT baseline | `freq_median / sigma=5.0 / min_cells=32` | 0.5263 | 0.7692 | 0.6250 | 88.266 kbps |
| Frame-level Tiny STFT-CNN | `device=cuda / epochs=12 / patches=64` | 1.0000 | 0.5385 | 0.7000 | 52.646 kbps |

## 2. 传输瓶颈基准

| 数据表示 | 等效速率 |
|---|---:|
| 原始 32+32 bit I/Q | 491.520 Mbps |
| 重量化 12+12 bit I/Q | 184.320 Mbps |
| 8-bit STFT | 约 245.535 Mbps，取决于 STFT 参数 |

## 3. 阶段性判断

1. frame-level CNN 在 GPU 环境下已经超过 tuned traditional baseline，但当前数据量很小，不能作为最终泛化结论。
2. 传统 baseline 仍然很重要，它提供了可解释、低成本、强对照的下界。
3. 两类语义候选框 payload 都是 kbps 级，显著低于原始 I/Q 和 STFT 上传。
4. 下一阶段需要增加更多真实/半真实数据，并比较硬判决、软概率、候选框和 feature token 的 payload。

## 4. 下一步工程任务

- 将一个长 SigMF 录制切成多个 frame-level 样本，建立 train/val/test manifest。
- 对每个 frame 生成 STFT、mask、标注和 payload 统计。
- 训练 CNN 时按 frame 划分，而不是在同一整帧内随机采 patch 后又整帧测试。
- 同时保留传统 baseline，作为后续语义 token 和多 UAV 融合实验的强对照。
