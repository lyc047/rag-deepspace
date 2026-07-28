# 阶段 4 Batch 3 报告：Gate A控制训练与C1决策

日期：2026-07-16  
状态：完成  
对应假设：H2/C1  
final holdout：未建立、未访问

## 1. 结论

在相同冻结detector、相同缓存、相同初始权重、相同epoch数据顺序和5个训练seed下，`detection+resource`相对`detection-only`在近似相同nominal数字链路bit下显著降低validation平均occupancy regret与CVaR0.9。因此C1资源选择感知损失进入后续冻结候选。

单纯rate方法虽降低bit但regret恶化；当前full joint没有稳定改善且校准变差。二者作为负结果保留，不继续在同一validation上搜索联合权重。码率控制转入C2显式节点—语义粒度调度。

## 2. 公平控制

- 300 train、50 calibration、150 validation scenes；
- 四方法共享相同轻量残差head和1/2/4/8 bit量化器；
- 每个seed共享完全相同的初始化状态与epoch permutation；
- 每方法160轮、相同优化器和学习率；
- checkpoint仅按validation预算违约、离散regret、CVaR、Brier依次选择；
- bit包含应用载荷、协议头、CRC、对齐、Hamming FEC和调制填充的nominal单次发送成本；
- 未使用test/final truth。

## 3. 五seed聚合结果

| Method | Regret | CVaR0.9 | Brier | nominal bit |
|---|---:|---:|---:|---:|
| detection-only | 0.023624 | 0.130095 | 0.009204 | 505.7 |
| detection+rate | 0.024987 | 0.135002 | 0.009608 | 485.8 |
| detection+resource | **0.021895** | **0.123676** | 0.009223 | 506.4 |
| full joint | 0.023298 | 0.131178 | 0.014034 | 491.3 |

## 4. 层次配对bootstrap

统计层级为5个训练seed×150个相同validation scenes，共10000次bootstrap。

| 方法对比detection-only | Regret差 [95% CI] | CVaR差 [95% CI] | nominal bit差 [95% CI] |
|---|---:|---:|---:|
| detection+rate | 0.001363 [−0.000566,0.003513] | 0.004907 [−0.004821,0.015191] | −19.86 [−28.75,−8.10] |
| detection+resource | **−0.001729 [−0.003082,−0.000534]** | **−0.006419 [−0.013153,−0.001264]** | 0.78 [−1.83,3.53] |
| full joint | −0.000326 [−0.002251,0.001505] | 0.001442 [−0.007452,0.010088] | −14.35 [−18.84,−10.01] |

## 5. 决策与边界

Gate A在validation阶段通过`resource-only`分支。该结果支持继续保留C1，但不是final H2结论。后续不得继续用同一validation反复调整C1权重；C1 checkpoint规则冻结，最终只允许在新final holdout建立后运行一次。

当前nominal bit不含随机重传实现值；C2和最终H4评价还必须加入链路随机性、ACK、反馈和实际重传bit。

全项目108项测试通过。原始训练轨迹、20个checkpoint、逐seed指标与bootstrap结果位于`results/stage4/gate_a_training_v1/`。

## 6. 下一批

进入C2：仅在train truth上生成候选节点—粒度加入当前融合信念前后的反事实regret下降，并除以公平数字链路成本形成价值/bit监督标签。validation只评价价值排序和调度性能，不参与标签生成。
