# 阶段4 Batch 12：论文材料自动化与声明边界

日期：2026-07-16  
算法状态：冻结，不修改模型、阈值、预算、统计门槛  
数据状态：final catalog不存在，access count=0

## 1. 目标

总计划要求不仅完成算法和实验，还要形成硕士论文可用材料。本批将冻结JSON直接转换为图、CSV和Markdown表格，撰写方法—实验—讨论章节草稿，并建立声明矩阵与final结果占位模板。所有材料都经过final访问标志、文件哈希、本地链接和必要占位符审计。

## 2. 自动生成资产

`scripts/generate_stage4_paper_assets.py`生成：

- Gate A三方法相对detection-only的regret、CVaR和bit森林图；
- 选择性G2相对all-G2在三信道、四Eb/N0下的regret与实际bit差图；
- 原始/校准任务可靠度图；
- C2 top-k召回和learned—oracle regret空间图；
- Gate A与端到端比较CSV；
- C1、C2、复杂度及覆盖状态主表。

图表不隐藏不利信息：选择性G2图同时画出0和冻结非劣界0.0026813；C2图保留低top-k召回及oracle差距。manifest记录全部源JSON和生成文件SHA-256，且扫描嵌套`final_holdout_accessed`与`labels_or_model_outputs_accessed`标志。

## 3. 论文文本

新增三份材料：

1. `STAGE4_THESIS_CHAPTER_DRAFT.md`：系统模型、occupancy regret、多粒度语义、C1方法、C2/C3负结果、数字链路、复杂度、统计协议、有效性威胁和小结；
2. `STAGE4_CLAIM_MATRIX.md`：逐项规定C1、选择性G2、质量校准、C2、C3、H3、MIMO、IQ和复杂度的允许/禁止表述；
3. `STAGE4_FINAL_RESULTS_TEMPLATE.md`：在访问前冻结C1与选择性G2比较方向、CI、Holm、非劣界和成功/失败分支。

章节草稿保留`[[FINAL_PENDING_*]]`标记；审计器要求这些标记、Gate B/C失败和H3不主张表述必须存在，防止写作时提前填入有利结论。

## 4. 审计与验证

- 论文材料审计：5项通过，0项失败；
- 本地图片/文档链接：全部存在；
- manifest源文件和生成文件哈希：全部匹配；
- final访问标志：未发现true；
- final access count：0；
- 全项目：145项测试通过。

## 5. 决策

论文材料开发验收完成。后续写作必须从自动表格和章节草稿引用数值，不手工从控制台复制；final运行前不得删除占位符或把validation写成最终检验。下一主线仍是获得至少200个新增独立scene。若外部数据长期不可得，论文只能明确写为development evidence并保留H2/H3未完成状态。
