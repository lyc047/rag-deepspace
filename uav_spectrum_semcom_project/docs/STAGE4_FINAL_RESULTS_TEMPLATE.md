# 阶段4 Final结果填写模板（访问前冻结）

> 当前状态：`LOCKED_PENDING_FINAL_DATA`  
> 禁止：在final访问后新增方法、改变margin/指标、删除不利scene或重跑随机种子。  
> 数据门槛：至少200个新增独立scene；当前0个；access count=0。

## 1. 完整性与访问记录

| 字段 | 冻结值/待填值 |
|---|---|
| final registry SHA-256 | `[[PENDING_REGISTRY_SHA256]]` |
| executable snapshot SHA-256 | `[[PENDING_FINAL_EXECUTABLE_SHA256]]` |
| scene count | `[[PENDING_N_GE_200]]` |
| real multi-receiver scene count | `[[PENDING_REAL_MULTI_COUNT]]` |
| access receipt timestamp | `[[PENDING_ACCESS_TIMESTAMP]]` |
| access count | `[[MUST_EQUAL_1_AFTER_RUN]]` |
| integrity failures | `[[PENDING; retain every failure log]]` |

## 2. 主检验族一：C1 resource loss

比较方向固定为 `detection_plus_resource - detection_only`。

| 指标 | 点估计 | 95% CI | 原始p值 | Holm调整p值 | 门槛 | 结论 |
|---|---:|---:|---:|---:|---|---|
| mean occupancy regret差 | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | CI上界<0 | `[[PASS/FAIL]]` |
| CVaR0.9 regret差 | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | CI上界<0 | `[[PASS/FAIL]]` |
| actual bit差 | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | CI包含0 | `[[PASS/FAIL]]` |

C1总体结论：`[[PASS only if all preregistered requirements pass; otherwise FAIL]]`。

## 3. 主检验族二：选择性G2

比较方向固定为 `G1 + stage3-prior selective G2 - all G2`，非劣界固定为0.0026813。

| 指标 | 点估计 | 95% CI | 原始p值 | Holm调整p值 | 门槛 | 结论 |
|---|---:|---:|---:|---:|---|---|
| mean occupancy regret差 | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | CI上界<0.0026813 | `[[PASS/FAIL]]` |
| actual transmitted bit差 | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | `[[PENDING]]` | CI上界<0 | `[[PASS/FAIL]]` |

选择性G2总体结论：`[[PASS only if both requirements pass; otherwise FAIL]]`。

## 4. 次指标

按预注册顺序报告clean resource rate、Brier、missed occupancy、端到端时延和report delivery ratio。次指标不能覆盖主检验失败。

## 5. 固定结论分支

- 两族均通过：可声明H2得到final支持，并分别报告任务对齐与通信节流贡献；
- 仅C1通过：只保留资源损失贡献，选择性G2降级为开发期节流观察；
- 仅选择性G2通过：只保留数字上报节流/非劣贡献，C1报告外部泛化失败；
- 两族均失败：如实报告development—final落差，论文贡献降级为方法、协议和负结果边界；
- H3：无论上述结果如何，没有足量真实多接收数据时始终写“未验证”。

## 6. 失败日志

`[[PENDING_APPEND_ONLY_LOG; never delete or overwrite]]`
