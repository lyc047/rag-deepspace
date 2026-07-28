# 阶段 4 开发证据综合与论文写作边界

> 历史状态：本文件只汇总Final前开发证据，不得再作为最终声明来源。正式结论见`docs/STAGE4_THESIS_CHAPTER_FINAL.md`。

日期：2026-07-16  
状态：开发算法、消融和端到端链路已冻结；独立 final holdout 尚未建立

## 1. 证据结论表

| 模块 | 开发结论 | 是否进入最终候选 | 论文允许表述 |
|---|---|---|---|
| C1 resource-aware loss | 相对 detection-only，regret 与 CVaR 配对 CI 均低于0，bit CI含0 | 是 | validation 上的候选创新，待 final 验证 |
| C1 rate/full joint | rate 降 bit 但损害 regret；full joint 不稳定且 Brier 恶化 | 否 | 完整负结果/消融 |
| C2 反事实标签 | oracle 空间大，train-only 标签与真实链路成本完整 | 保留基础设施 | 可写方法与可观测性分析，不写性能已成立 |
| C2 价值网络 | 普通 G1 与付费草图下均未稳定超过简单先验；跨 split 不稳 | 否 | 不挑最佳 seed，作为负结果 |
| C2 解析/背包 | 普通 G1 因节点同质退化；草图恢复差异但系统固定开销抵消 | 否 | 可解释求解器与信息—开销发现 |
| 2-bit G1 | regret收益不显著，成本增加，Brier与升级空间恶化 | 否 | 语义粒度消融负结果 |
| 稀疏残差草图 | 恢复候选差异，局部策略改善，但系统级与跨split门槛失败 | 否 | codec/可观测性实验，不宣称通信效率贡献 |
| C3 可靠少数直接保护 | protection rate为0，实际退化为审计质量均值 | 否 | 不得称作少数保护成功 |
| C3 确认式G3 | 修正特定故障，但误触发、bit、false occupancy和CVaR失败 | 否 | 受控机制负结果，不声称H3 |
| 端到端选择性G2 | 12个信道点相对全G2显著省bit；regret差CI均跨0 | 经典候选 | development通信—任务权衡，非正式非劣证明 |

## 2. 当前硕士论文主线

论文核心应收敛为：

1. 公平数字链路与任务评价框架：所有表示统一计入头部、CRC、FEC、填充、重传、反馈、解码后 regret/clean/CVaR；
2. C1 资源选择 regret 感知语义生成：这是当前唯一通过 validation 统计门的算法候选；
3. C2 条件价值的可观测性—开销边界：反事实标签、G1 同质化、2-bit/Brier 失败、稀疏草图成本和选择性上报构成完整研究发现；
4. C3 安全回退边界：可靠少数与补传在受控故障中的收益、误报和成本冲突；
5. 多信道实际解码路径验证。

不得把网络数量或所有尝试都包装成创新。最终贡献强度取决于独立 final holdout 是否复现 C1，以及是否能在预注册非劣界下确认选择性 G2 的 bit 优势。

## 3. final 前冻结项

- C1 checkpoint 与五 seed 汇总规则；
- 普通 1-bit G1、G2 4-bit 和阶段2数字链路；
- 选择性 G2 经典调度，不使用 C2 网络或草图；
- 固定经典融合，不使用 C3 保护；
- scene 统计单位、链路重复嵌套、10000 次配对/层次 bootstrap；
- 主指标 regret、CVaR、clean、实际 bit、时延，Brier/miss 为辅助；
- final 访问一次，访问后禁止调参。

## 4. 当前不能完成的外部条件

项目现有 RadDet scene 已被开发流程使用，LoRaIQ 只有两个独立事件，均不能诚实构成新的 final holdout；真实独立多接收机 scene 也不足。因此当前不宣称 final H2 或 H3。后续需要新增独立 scene catalog，按既定 scene 级规则物化 final holdout，才能进行论文最终统计检验。
