# 阶段 4 Batch 10：开发验收闭环与 Final 防误用工具

日期：2026-07-16  
数据范围：calibration / validation 聚合诊断；未创建、未读取 final holdout  
最终状态：development acceptance 通过；final acceptance 未通过（0/200 独立 scene）

## 1. 本批目标

本批不再改算法、阈值或预算，而是逐条核对总计划交付物，补齐质量诊断和价值排序的遗漏证据，并把 final 数据注册、泄漏检查和一次性访问规则落实为代码。验收状态严格区分 `passed`、`waived`、`blocked` 和 `failed`。

## 2. 补充证据

### 2.1 质量诊断

在 calibration 的 50 scene / 200 node rows 上拟合单调置信度校准器，在 validation 的 150 scene / 600 node rows 上只输出聚合量：

- Brier：原始 `0.179472`，校准后 `0.000421`；
- ECE：原始 `0.422625`，校准后 `0.002756`；
- 对削顶、带外泄漏、噪声不稳、陈旧、低上报可靠度、校准故障和“高置信但失准”做受控注入，并输出 pairwise detection 与 score AUC。

这里的可靠度目标是 `1 - occupancy MAE`。数值说明原始 `prediction_confidence` 与该任务可靠度不在同一尺度，不能直接解释为正确概率。受控异常只检验质量乘积的方向响应，不证明真实故障检出率。

### 2.2 价值网络诊断

对冻结五 seed 网络在 validation 内存中枚举候选，仅保存聚合指标：

- value/expected-bit MAE：`2.5264e-05`，目标标准差 `5.3171e-05`；
- pooled Spearman：`0.00938`，scene 平均 Spearman：`0.00274`；
- top-1 / top-2 / top-3 recall：`0.1200 / 0.2133 / 0.3000`；
- 所选动作相对全 G1 平均 regret 下降 `0.003097`，但 oracle 可下降 `0.022672`，且所选动作平均增加约 `580.87` expected bits。

排序结果接近无信息，完整支持而非推翻 Gate B 的负决定。validation 候选标签和逐 scene 行未写盘。

## 3. 验收与 final 工具

新增：

- `stage4_acceptance.py` 与 `audit_stage4_acceptance.py`：逐条生成机器可读验收报告；
- `final_holdout.py` 与 `register_stage4_final_catalog.py`：强制至少200个独立 scene，检查 scene/group/event/source hash/provenance 唯一性以及开发集重叠；
- `consume_stage4_final_access.py`：显式确认后才允许访问计数从0原子变为1，二次调用必定拒绝；
- `freeze_stage4_final_snapshot.py`：冻结可执行源码、配置和 checkpoint 哈希；
- `configs/stage4_final_catalog_template.json`：仅为模板，不能直接运行；
- `configs/stage4_final_access_state.json`：当前 `access_count=0`。

验收审计当前无 failed 项；development 全部通过。`F01` 新独立数据和 `F02` 一次性 final 运行保持 blocked，不能用 validation 重命名解除。

## 4. 验证

- 全量测试：137项通过；
- 协议与 protocol-only registry 校验：0错误；
- final access count：0；
- 可执行冻结快照：`e2a08e46af266bee3c2f87b08930699da48b16c9a9bad46334cf70729c23b7fc`；
- 质量与价值结果均标记 `final_holdout_accessed=false`。

## 5. 结论与下一动作

阶段4开发工作现已达到可审计冻结状态。C1是唯一保留到 final 的算法候选；选择性 G2 是通信节流候选；C2/C3按预注册门槛作为完整负结果。下一动作只能是获得真实、未消费、可审计的独立 scene catalog并通过注册器；数据到位前不再改算法，也不宣称 final H2 或 H3。
