# 阶段 4 Final Holdout 一次性运行手册

状态：单次Final已经完成；C1未通过，选择性G2通过；当前访问次数1，不得重跑

## 1. 运行前硬门槛

只有同时满足以下条件才允许物化和访问 final：

1. 至少 200 个新增独立 RF scene，不能来自已用于阶段1—4开发的 RadDet/LoRaIQ事件；
2. scene ID、采集事件、时间段和接收节点关系可审计；
3. 在任何节点/信道增强前按 scene 切分；
4. 与 train/calibration/validation 的 scene ID、源文件哈希和采集事件无重叠；
5. 标签完整性检查通过，但模型输出在一次性运行前不可查看；
6. 代码、配置、checkpoint、非劣界、统计脚本和依赖快照全部哈希冻结；
7. `final_holdout.access_count` 仍为 0。

任一项不满足则不得以 test/final 名义运行，也不得用现有 validation 重命名代替。

## 2. 两个主检验族

### C1 resource-aware loss

比较五个预注册种子的 `detection+resource` 与 `detection-only`，scene 内先汇总 seed/链路重复，再做层次配对 bootstrap。要求：

- regret 差 CI 上界 < 0；
- CVaR 差 CI 上界 < 0；
- bit 差 CI 包含 0。

### 选择性 G2 数字上报

比较 `G1 + 阶段3先验单节点G2` 与 `all_G2`。regret 非劣界固定为 `0.0026813`，等于冻结 validation 随机—oracle gap `0.026813` 的 10%；不是读取 final 后决定。要求：

- regret 差 CI 上界 < 0.0026813；
- 实际发送 bit 差 CI 上界 < 0。

两个主检验族使用 familywise alpha 0.05 和 Holm 校正。clean、Brier、miss、时延和 delivery ratio 为次指标，不得替代失败的主检验。

### AERPAW三站点M1分支

在运行任何非pilot模型前，先按`configs/aerpaw_three_site_prefinal_protocol_v1.json`检查CC1/CC2/LW1时间对齐。若30分钟稀疏化后仍有至少200个三站点scene，继续运行上述两个family，但选择性G2只能声明三固定站点迁移性。若不足200个，禁止异步拼接，catalog改为LW1单站点C1；此时只运行预注册`aerpaw_c1_only_final_statistics_v1`，保持相同C1 CI/IUT门槛。单一可用family的Holm调整p值等于原始family p值。M1分支选择只由时间戳、完整性和scene数量决定，不得使用方法输出。

## 3. 一次性运行顺序

1. 生成不可变 final registry、scene/source 哈希和环境快照；
2. 自动运行跨 split 泄漏、形状、标签范围和缺失文件测试；
3. 将 access count 从 0 原子更新为 1，并记录时间、操作者、commit/工作树哈希；
4. 一次执行冻结 C1 与端到端数字链路脚本；
5. 输出逐 scene 机器可读结果、聚合表、bootstrap分布摘要和完整日志；
6. 无论结果成功或失败，禁止重新训练、换 seed、改阈值、改预算、改 margin 或删 scene；
7. 只有预注册的文件损坏/标签缺失等完整性失败可标记无效，且必须保留原始日志。

## 4. 结果写作规则

- 两个主检验分别报告点估计、95% CI、效应量和 Holm 调整后的结论；
- final 失败即如实报告，不用 validation 结果覆盖；
- C2 网络、草图和 C3 保护已在开发阶段淘汰，不在 final 中重新复活；
- 没有真实关联多接收机 scene 时，继续禁止宣称 H3；
- IQ 容量参考继续标为理论参考，不写成实测 codec 压缩结果。

## 5. 当前等待项

CC1、CC2、LW1均已完成官方大小与SHA-256复核。三站点对齐得到13,620个scene，pilot排除和30分钟稀疏化后保留227个，M1通过；其中200个已按预注册规则完成单次Final。C1 family未通过，选择性G2 family通过；不能再进行算法、阈值、checkpoint、场景或统计规则修改后重跑确认性实验。

## 6. 已固化的执行入口

以下顺序不可交换：

1. **已完成**：200个三站点scene的catalog生成、泄漏检查、文件哈希审计与registry注册；
2. **Final前已完成**：168项全量测试及`audit_stage4_acceptance.py`，当时确认只有F02尚未执行；
3. **已完成**：生成124文件可执行快照，SHA-256为`43e6162d2d267246977da8f3ec35fa881fc2db77de8aeab8e705137afbae379f`；
4. **已完成**：运行`python scripts/run_aerpaw_three_site_final_inference.py --preflight`，结果为`ready_for_single_access`；
5. **待执行且会消耗唯一访问**：在正式推理即将开始时运行：

   ```powershell
   python scripts/consume_stage4_final_access.py --actor local-final-20260721 --code-snapshot-sha256 43e6162d2d267246977da8f3ec35fa881fc2db77de8aeab8e705137afbae379f --confirm CONSUME_STAGE4_FINAL_ACCESS_ONCE
   ```

6. 访问消费成功后立即运行`python scripts/run_aerpaw_three_site_final_inference.py`。入口按`configs/aerpaw_final_inference_protocol_v1.json`对五seed、三类信道、四个Eb/N0、六种站点偏置排列和五次链路重复做scene内聚合；
7. 推理成功后立即运行`python scripts/run_stage4_final_statistics.py --input results/stage4/final_holdout_v1/final_scene_metrics.json`；统计入口固定使用10,000次scene bootstrap和单侧配对随机化；
8. 即使程序中断，也不得把计数改回0、修改冻结文件或重跑；仅预注册的文件完整性失败可以判无效，并且必须保留完整日志。

当前机器状态：`configs/stage4_final_access_state.json`中`access_count=1`；200个scene的正式指标和统计结果已经生成。后续只允许结果复核、论文制图和明确标记的探索性分析，不得修改冻结方法后重新声称独立Final。完整结果见`docs/STAGE4_FINAL_RESULTS_REPORT.md`。
