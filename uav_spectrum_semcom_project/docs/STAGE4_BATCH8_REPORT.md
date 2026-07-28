# 阶段 4 Batch 8：C3 可靠少数保护与确认式补传

日期：2026-07-16  
状态：已完成；直接保护与确认式 G3 补传均未通过 Gate C，回退经典融合  
证据范围：受控故障鲁棒性，不是实际 H3 空间协作证据

## 1. 冻结场景与指标

在 150 个 validation scene 上固定七类场景：正常、三节点假清洁、三节点频移、三节点高置信错误、三节点陈旧、关键节点掉线、良性单点误报。所有变换不读取当前 truth；truth 只用于 regret、clean、miss、false occupancy 和 Brier 评价。

比较 mean、median、SNR weighted、阶段3 robust quality、审计质量均值、可靠少数直接保护和确认式 G3 补传。结果只用于 C3 受控机制判断，不宣称多接收机空间增益。

## 2. 直接可靠少数保护

直接策略要求候选少数的平均通道分歧、审计质量比和占用侧 excess 同时过门槛。最终所有场景 protection rate 均为 0，说明平均通道分歧会稀释稀疏冲突。`reliable_minority` 与 `audited_quality_mean` 完全相同，不能把审计质量加权的收益写成少数保护贡献。

pooled 结果中：

- `audited_quality_mean`：regret 0.025537，CVaR 0.134315，false occupancy 0.152139；
- `stage3_robust_quality`：0.026877 / 0.137259 / 0.194246；
- 最佳经典 CVaR 是 median 的 0.131170，SNR weighted 为 0.132340。

审计质量均值改善了平均 regret 和 false occupancy，但没有超过最佳经典尾部风险，因此不满足 C3 主门槛。

## 3. 确认式 G3 回退

按 Gate C 回退，仅改变冲突处置：最大逐通道 spread 超过 0.2 时，向最高审计质量节点发送 32-bit 反馈并请求 G3；确认后只在两个最大冲突通道双向向该节点融合，反馈和 G3 均通过阶段2链路计费。

pooled 结果：regret `0.025304`、CVaR `0.135580`、false occupancy `0.145803`、触发率 `46.48%`、增量成本 `390.66 bit/decision`。它在 shifted majority 上相对阶段3 robust regret 差 `−0.003220`，95% CI `[-0.007121, -0.000026]`，并在 high-confidence wrong majority 将 false occupancy 从 `0.8294` 降到 `0.4423`。

但安全性与成本失败：

- 良性单点误报触发率 100%，增量约 840 bit，false occupancy 显著增加 0.001229；
- 假清洁多数虽显著降低 miss 0.01389，却增加 false occupancy 0.04526；
- pooled CVaR 仍劣于 median/SNR weighted；
- 正常场景也有 10.67% 误触发。

## 4. 决策 D4-008

不再在当前 validation 上搜索 spread 阈值、保护强度或通道数。C3 的直接保护未实际激活，确认式补传虽能处理特定故障，但误触发、额外 bit 和尾部风险未满足预注册成功条件。最终系统回退到固定经典融合；C3 作为带完整成本和安全反例的负结果保留。

后续阶段4不再增加算法分支。冻结开发候选为：C1 `detection+resource` 前端；普通数字语义与阶段2公平链路；C2/C3 仅作为消融/负结果；融合采用 mean/median/SNR weighted 等明确基线。下一批运行端到端信道、预算、SNR、节点失效和 age sweep，并生成总消融/统计材料。

全项目回归：`127 passed`。
