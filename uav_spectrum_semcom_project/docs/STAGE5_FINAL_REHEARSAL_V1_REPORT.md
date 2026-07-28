# 阶段五独立 Final 前模拟演练报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-07-24
- Verification Status：VERIFIED
- Version Label：stage5_final_rehearsal_v1
- Data Role：已见开发验证数据，仅用于流程演练
- Final Status：不是独立 Final，不产生确认性论文结论

## 1. 演练目标与边界

本次工作只验证“冻结配置 → 执行主效应实验 → 执行 ACK 鲁棒性实验 → 保存日志与结果 → 复现性核验 → Final 隔离审计”的完整流程。演练不重新搜索参数，不读取阶段四 Final 测量或指标，不消耗新的独立数据，也不把结果解释为阶段五最终有效性证据。

演练使用两条已冻结管线：

1. 主效应管线：多查询事件语义包在 cyclic 与 Markov 查询序列上的性能；
2. 鲁棒性管线：在 8 组 ACK 丢失与重复 ACK 注入条件下，状态不确定性立即恢复策略相对 naive ACK 的表现。

## 2. 冻结输入与执行命令

### 2.1 主效应

- 配置：`configs/stage5_query_bundle_development_v1.json`
- 配置 SHA-256：`c6ed7efc481af0a57c093672b469253c29e954e65747c08ac6cdded99ff64cba`
- 执行命令：`python scripts/run_stage5_query_bundle_development.py --protocol configs/stage5_query_bundle_development_v1.json --out-dir results/stage5/final_rehearsal_v1/primary`

### 2.2 ACK 鲁棒性

- 配置：`configs/stage5_ack_state_recovery_development_v1.json`
- 配置 SHA-256：`28f94c847e0bafe2185f6bde7904ec90de839b7848c668b0d7c6d99cdb0f8c42`
- 执行命令：`python scripts/run_stage5_ack_state_recovery_development.py --protocol configs/stage5_ack_state_recovery_development_v1.json --out-dir results/stage5/final_rehearsal_v1/robustness`

两条命令均正常退出，标准错误日志均为空。

## 3. 主效应演练结果

候选方法均为冻结的 `query_event_bundle_fixed`。实际 bit 包含既定信道编码与分包开销，而不是只比较名义源编码长度。

| 查询序列 | 平均实际 bit | 平均 regret（dB） | CVaR₀.₉（dB） | 发送率 | 相对逐场景绝对索引 bit 降幅 | 平均 regret 差上界（dB） | CVaR 差上界（dB） | 描述性门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cyclic | 203.920 | 0.15952 | 0.57609 | 48.72% | 48.29% | 0.00410 | −0.05046 | 通过 |
| Markov | 199.293 | 0.15372 | 0.54124 | 47.64% | 49.50% | 0.00656 | −0.04039 | 通过 |

相对阶段五早期的“按单次查询事件发送”方法，多查询语义包在 cyclic 与 Markov 序列上又分别降低 29.64% 和 16.12% 的实际 bit。两种序列均通过既定描述性门槛，故主效应流程演练为 2/2 通过。

## 4. ACK 鲁棒性演练结果

下表比较 `uncertainty_recovery` 与 `naive_ack`。平均 regret 差上界与 CVaR 差上界为候选减基线的单侧 95% 上界；负值有利于候选。联合门槛同时约束平均 regret、CVaR 和实际 bit 增幅。

| ACK 丢失 | 重复 ACK | 候选平均 bit | 候选平均 regret（dB） | 候选 CVaR₀.₉（dB） | 状态分歧率 | bit 增幅 | 平均差上界（dB） | CVaR 差上界（dB） | 联合门槛 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0% | 0% | 210.894 | 0.14816 | 0.64534 | 0.00% | 4.02% | −0.00003 | 0.00922 | 未通过 |
| 0% | 10% | 211.504 | 0.15058 | 0.65452 | 0.00% | 1.87% | −0.00358 | 0.00806 | 未通过 |
| 5% | 0% | 215.185 | 0.14545 | 0.62160 | 2.43% | 4.45% | −0.00390 | −0.00566 | 通过 |
| 5% | 10% | 212.294 | 0.14457 | 0.62324 | 2.17% | 1.70% | −0.00279 | 0.02080 | 未通过 |
| 10% | 0% | 217.907 | 0.14537 | 0.60867 | 4.66% | 5.29% | −0.00352 | −0.00449 | 通过 |
| 10% | 10% | 217.778 | 0.14926 | 0.62499 | 4.48% | 2.12% | −0.00283 | 0.00733 | 未通过 |
| 20% | 0% | 228.874 | 0.14705 | 0.61168 | 9.73% | 5.61% | −0.00658 | −0.00253 | 通过 |
| 20% | 10% | 230.093 | 0.14779 | 0.59277 | 10.06% | 2.68% | −0.00886 | −0.00942 | 通过 |

结果严格复现 4/8 条件通过。该结果表明 8-bit 纪元保护与立即恢复机制能够处理部分中高 ACK 故障，但在低故障和“丢失 + 重复”混合条件下存在过度恢复或尾部风险，当前 ACK 扩展不能作为全面通过的默认协议。

## 5. 复现性与 Final 隔离审计

### 5.1 逐字节复现

| 结果 | 本次演练 SHA-256 | 原冻结结果 SHA-256 | 是否一致 |
|---|---|---|---|
| 主效应 | `30e9a57b88e9ee30891a6499664f21b910d806ed1925c29bf3be5d3d4cd21751` | `30e9a57b88e9ee30891a6499664f21b910d806ed1925c29bf3be5d3d4cd21751` | 是 |
| ACK 鲁棒性 | `b6c43453fad767b7263bc30a1f2de561d73cb6c531d85a0cce0ea6a86d0a5356` | `b6c43453fad767b7263bc30a1f2de561d73cb6c531d85a0cce0ea6a86d0a5356` | 是 |

两条结果文件均与原冻结结果逐字节一致，说明当前执行环境能够确定性复现阶段五核心结果。

### 5.2 治理字段

两条结果均明确记录：

- `stage4_final_measurements_loaded = false`
- `stage4_final_metrics_loaded = false`
- `fixed_policy_retuned = false`
- `confirmatory_final = false`

ACK 结果还记录 `ack_faults_are_real_measurements = false`，即故障来自受控注入而非真实无人机回传链路测量。

### 5.3 历史 Final 访问状态

| 状态文件 | 状态 | 访问次数 | SHA-256 |
|---|---|---:|---|
| `stage4_c1_temporal_access_state.json` | access_consumed | 1 | `ed01421504de10bcbe671cc15d335a036fafd715628f3a7f500e837b407e4a54` |
| `stage4_c1_v4_final_access_state.json` | access_consumed | 1 | `550caaac3c04806c0a7ccdc7dc4d6daaf2f747c27479a608b0b3fcc8d956ccf8` |
| `stage4_final_access_state.json` | access_consumed | 1 | `df312e9ed88b3a56f3094e5a468cf634b5cccfa542e2c3d36e6aae8de4d9946c` |

本次演练未调用历史 Final 入口，三个访问计数均未增加。

## 6. 谬误与偏差扫描

| 检查项 | 状态 | 判断 |
|---|---|---|
| 1. Simpson 悖论 | NOTE | cyclic、Markov 与 8 个 ACK 条件分别报告，不用汇总均值掩盖分组差异 |
| 2. 生态谬误 | CAUTION | 仿真 ACK 故障不能直接代表真实无人机回传链路 |
| 3. Berkson 悖论 | CAUTION | 主轨迹来自已见固定站点开发数据，存在选择范围限制 |
| 4. Collider 偏差 | NOTE | 未通过事后选择中间变量构造优势子集 |
| 5. 基准率忽视 | NOTE | 同时报告 ACK 故障率、发送率、bit、平均 regret 与 CVaR |
| 6. 均值回归 | NOTE | 未按极端单次结果选择条件，使用冻结条件与聚类 bootstrap |
| 7. 幸存者偏差 | NOTE | 失败包、ACK 丢失和重复 ACK 均保留在核算中 |
| 8. Look-elsewhere 效应 | CAUTION | 如实报告 4/8，不把部分条件通过解释为整体成功 |
| 9. 分岔路径 | NOTE | 配置、随机种子、门槛和方法均在运行前冻结 |
| 10. 相关不等于因果 | CAUTION | 只解释受控模型内的机制差异，不外推部署因果收益 |
| 11. 反向因果 | NOTE | 不适用于受控仿真比较 |

覆盖率：11/11。总体证据等级为“开发证据已验证，确认性证据待独立 Final”。

## 7. 正式 Final 就绪清单

已就绪：

- 主效应和鲁棒性执行脚本可正常运行；
- 候选方法、基线、门槛、随机种子与统计流程均已冻结；
- 实际 bit、平均 regret、CVaR、发送率和非劣上界均能自动输出；
- 日志、结果哈希与治理字段可以自动留痕；
- 历史 Final 单次访问保护有效；
- 在当前环境中，冻结结果可以逐字节复现。

尚未就绪：

- 缺少未被开发过程查看和调参使用的新独立数据；
- 尚未为阶段五新建数据清单、用途注册表和独立访问状态文件；
- 阶段五正式 Final 尚未冻结代码提交或等价代码快照；
- ACK 故障目前为仿真注入，缺少真实回传链路的 ACK 丢失、时延和重复分布；
- ACK 鲁棒性候选仅通过 4/8 条件，正式 Final 应把它作为扩展/边界实验，而不是已全面成熟的主结论。

## 8. 演练结论

本次模拟演练成功完成，证明阶段五核心实验管线具备可执行性、确定性复现能力和 Final 数据隔离能力。可冻结进入正式 Final 的主候选是“多查询事件语义包”；ACK 立即恢复协议应保留为具有部分正结果和明确失效边界的鲁棒性扩展。

现有数据足以演练流程，但不足以产生新的独立确认性结论。下一步不是继续在已见数据上调参，而是准备新独立数据注册表、独立访问状态和代码快照，然后只执行一次正式 Final。

## 9. 产物位置

- 主效应结果：`results/stage5/final_rehearsal_v1/primary/query_bundle_result.json`
- ACK 鲁棒性结果：`results/stage5/final_rehearsal_v1/robustness/ack_state_recovery_result.json`
- 主效应日志：`runs/stage5_final_rehearsal_v1/primary.stdout.log`
- ACK 鲁棒性日志：`runs/stage5_final_rehearsal_v1/robustness.stdout.log`
- 标准错误日志：`runs/stage5_final_rehearsal_v1/primary.stderr.log`、`runs/stage5_final_rehearsal_v1/robustness.stderr.log`

