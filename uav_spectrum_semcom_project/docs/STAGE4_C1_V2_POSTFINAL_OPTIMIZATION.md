# C1-v2：Final后探索性优化记录

> 日期：2026-07-21  
> 性质：Final后探索性开发，不改变已完成的AERPAW Final结论  
> 数据边界：只读取原`train/calibration/validation`缓存，未加载任何AERPAW Final源文件、代理label或scene指标

## 1. Final暴露的问题

C1在AERPAW Final中的平均regret点估计改善约1.46%，CVaR显著改善，但平均regret区间略跨0，而且实际发送bit增加约1.40%。代码审计发现旧`detection_plus_resource`没有显式rate约束；可变精度头在不同输入分布下会在1-bit与8-bit间跳变，因此开发集上的近似同bit不能保证外部数据上的同bit。

C1-v2据此采用三个原则：

1. 两种损失共享同一个固定精度策略，从结构上保证应用层长度一致；
2. 对检测占用输入施加稀疏、稠密、温度和噪声校准扰动，提高映射变化下的稳定性；
3. checkpoint只由calibration选择，validation只用于一次性方法比较。

## 2. 代码改动

- `FixedPrecisionOccupancyHead`：删除易受域偏移影响的rate logits，保留轻量残差占用修正；
- `calibration_shift_views`：生成不读取真值或Final数据的可复现校准扰动视图；
- `contiguous_block_ranking_loss`：直接要求真实最优连续资源块在预测块代价中排在其他候选之前；
- `train_stage4_c1_v2_exploratory.py`：五种子、共享初始化、calibration选checkpoint、scene级bootstrap和Final路径拒绝；
- 新增固定精度、扰动复现性和块排序梯度测试。

## 3. 三次完整披露的开发尝试

| 版本 | 唯一主要变化 | 平均regret差及95% CI | CVaR差及95% CI | bit差 | 结论 |
|---|---|---:|---:|---:|---|
| v2-fixed2 | 固定2-bit、鲁棒视图、soft regret | +0.000027 [−0.000075, +0.000155] | 0.000000 [约0, 约0] | 0 | 失败；量化过粗 |
| v2-fixed4-ranking | 固定4-bit、增加块排序损失 | −0.001502 [−0.003659, +0.000425] | −0.006156 [−0.021435, +0.007278] | 0 | 有利趋势，区间未通过 |
| v2-paired-finetune | 从每个种子的鲁棒检测基线做小学习率配对资源微调 | **−0.001515 [−0.003304, −0.000013]** | −0.008988 [−0.020767, +0.000576] | **0** | 平均与bit通过，CVaR仍未通过 |

第三版在validation上的平均regret由0.024092降至0.022577，约降低6.29%；CVaR点估计约降低6.68%；Brier由0.007488降至0.007392。两方法固定为4-bit G2，名义发送bit均为518，因此不存在旧C1由精度头引起的码率漂移。

## 4. 当前判断

代码优化已经解决两个核心工程问题：

- **bit公平性**：由“开发集上统计近似相同”提升为“编码结构上严格相同”；
- **平均资源收益稳定性**：配对微调版本的scene级bootstrap区间已经完全低于0。

但CVaR区间上界仍为+0.000576，故第三版也不满足预设的平均、尾部和bit三项同时通过规则。继续根据同一validation反复调整tail权重会形成验证集泄漏，本轮在此停止，不进行第四次自动试参。

## 5. 下一步严谨方案

1. 将第三版作为下一轮候选，不覆盖原C1或AERPAW Final结果；
2. 新增独立calibration数据，用它选择tail权重、ranking margin与固定精度；现有validation不再参与调参；
3. 在新的独立多站点数据上只运行一次C1-v2确认性实验；
4. 两种方法使用相同应用长度和相同链路随机数，实现scene内严格配对的实际发送bit比较；
5. 新Final仍同时要求平均regret、CVaR和bit三项门槛，不能只报告点估计。

因此，C1-v2目前是“明显优于首版的开发候选”，不是已经被新Final证明有效的算法。
