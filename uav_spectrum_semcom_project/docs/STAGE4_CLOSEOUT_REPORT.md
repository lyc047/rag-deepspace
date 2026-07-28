# 阶段4收尾报告

> 收尾日期：2026-07-23  
> 完成度：约95%；核心算法、正负结果和单次Final证据链已闭环。  
> 剩余5%：整篇硕士论文排版、与其他章节衔接、答辩材料和可选的跨数据集增强验证。

## 已完成的研究闭环

1. C1原资源损失方法完成开发、原Final失败和原因审计；C1-v3完成独立时域负确认。
2. C1-v4完成任务充分块语义设计、五表示比较、实际数字链路bit核算和开发选型。
3. 零比特最近成功状态回退完成开发验证，并在Final前加入60分钟陈旧保护。
4. 使用未复用的CC1/CC2 2月14—15日280景完成一次性Final；H1和H2均通过。
5. 原Final中的选择性G2正结论保留；C2和C3以完整负结果及机制边界保留。
6. Final协议、清单、执行快照、访问回执、缓存、结果和论文资产均有SHA-256及不可覆盖目录。
7. 在不读取Final的前提下完成经典硬/软表示、ARQ与状态保持补充基线；补充结果仅用于机制解释。
8. 完成N=8—64、三种连续需求比例的规模扩展；当前独热指示在N≥32的6个任务中达到至少20%实际bit节省和regret非劣。

## 最终贡献结构

- 核心贡献一：面向连续频谱块选择的任务充分数字语义。
- 核心贡献二：带最大陈旧时间的零比特接收端状态回退。
- 补充系统贡献：真实数字包成本下的选择性G2上报节流。
- 研究方法贡献：任务指标、实际bit、CVaR、分组统计和单次访问治理组成的可审计闭环。
- 负结果贡献：C2预发送价值不可辨识及C3误触发/固定开销边界。

## 论文材料索引

- 正式章节：`docs/STAGE4_THESIS_CHAPTER_FINAL.md`
- Final详细结果：`docs/STAGE4_C1_V4_FINAL_RESULTS.md`
- 声明边界：`docs/STAGE4_CLAIM_MATRIX.md`
- 原Final历史报告：`docs/STAGE4_FINAL_RESULTS_REPORT.md`
- C1-v3负确认：`docs/STAGE4_C1_V3_TEMPORAL_CONFIRMATION.md`
- C1-v4开发报告：`docs/STAGE4_C1_V4_RESULTS.md`
- C1-v4补充基线：`docs/STAGE4_C1_V4_SUPPLEMENTARY_BASELINES.md`
- C1-v4规模扩展：`docs/STAGE4_C1_V4_SCALABILITY_EXPERIMENT.md`
- 自动图表：`results/stage4/c1_v4_final_paper_assets_v1/`
- 补充基线图表：`results/stage4/c1_v4_supplementary_baseline_assets_v1/`
- 规模扩展图表：`results/stage4/c1_v4_scalability_assets_v1/`
- 机器收尾审计：`results/stage4/c1_v4_closeout_v1/`

## 最终边界

本阶段可以宣称同一AERPAW活动内、固定站点跨日期和冻结仿真数字链路上的方法有效性。不能宣称跨数据集普遍性、无人机移动协作、MIMO物理层、原始I/Q检测精度、实测回传链路或目标硬件实时性。后续新数据只能用于增强外部效度，不能重新调参或覆盖已消费Final。
