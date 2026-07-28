# 阶段 4 总计划：资源后悔感知的变长频谱语义与多节点可靠调度

> 文档性质：阶段 4 唯一执行基准（Living Plan）  
> 初始版本：v1.0  
> 建立日期：2026-07-16  
> 适用范围：硕士论文核心创新 C1、C2，以及与阶段 3 的 C3 联合优化  
> 当前状态：阶段4核心研究与单次Final已完成，进入论文整合与可选外部效度增强  

## 0. 使用与维护规则

本文件是阶段 4 的单一事实源。后续开始任何算法修改、数据处理或正式实验前，必须先检查本文件，并在“动态状态”“决策记录”或“变更记录”中登记。旧的阶段 0—3 文档继续作为已完成工作的证据，不用本文件覆盖历史结论。

维护原则：

1. 每次只围绕一个可证伪假设修改一个主要机制；
2. 训练集用于拟合，calibration 集用于校准，validation 集用于选型，final holdout 只在最终冻结后运行一次；
3. 阶段 3 已查看的 holdout v1—v3 不得作为阶段 4 最终证据；
4. 任何预算、阈值、损失权重、非劣界或成功规则必须在读取 final holdout 前登记；
5. 负结果必须保留，不得通过更换测试集、随机种子或评价规则隐藏；
6. 计划允许调整，但每次调整必须记录“依据、影响、授权数据范围和回退方案”；
7. 不能映射到 H2/H3/H4 的工作可以作为诊断或扩展，但不得包装成核心创新证据；
8. 后续执行本项目时，先更新本文件的状态，再修改代码；实验完成后立即回填结论和下一决策。

---

## 1. 当前基础与问题诊断

### 1.1 已完成基础

- 阶段 0：论文范围、RQ1—RQ3、H1—H4、主指标和证据边界已冻结；
- 阶段 1：数据注册、配置、随机种子、结果元数据和复现入口已建立；
- 阶段 2：公平数字链路、分包、CRC、FEC、QPSK、重传和实际 bit 审计已完成，形成 H1/H4 阶段性证据；
- 阶段 3：完成经典融合、质量加权、DeepSets 初测、鲁棒性压力测试、价值/bit 节点选择、ACK 候补、风险回退、同源 I/Q 回放和 LoRaIQ 四接收机 pilot；
- 当前测试基线：189项单元测试通过（新增规模协议、编码长度和结果边界测试；Final防误用与收尾审计继续通过）。

### 1.2 阶段 3 已确认的事实

1. 质量加权和 DeepSets 尚未显著优于最强经典平均融合，H3 尚未成立；
2. 固定少节点价值/bit 调度在正常条件下可明显节省 bit，但关键节点失效或信息陈旧时可能恶化；
3. ACK 候补与三节点回退可在当前半合成场景中减少约 30% 上行 bit，并维持统计上无显著差异的 regret；
4. 真实 LoRaIQ pilot 表明 SNR 高不等于占用估计可靠，前端饱和和全频带泄漏必须进入质量诊断；
5. 多数一致性可能压制唯一正确的可靠少数节点；
6. 当前真实数据只有两个独立事件，资源选择任务饱和，不能形成 H3 正式统计结论；
7. 阶段 4 不应继续堆叠融合网络，而应联合优化语义生成、节点选择、语义粒度和任务补传。

### 1.3 阶段 4 的研究定位

阶段 4 暂定方法名：

> **RAVES：Regret-Aware Variable-rate Evidence Semantic Scheduling**  
> 资源选择后悔感知的变长证据语义与可靠调度

主要验证 H2：

> 在相同实际发送 bit 预算下，直接优化资源选择 regret 的语义生成和调度方法，是否优于 detection-only、置信度 top-k、固定语义包和阶段 3 先验 value/bit 调度？

联合验证 H2+H3：

> 在多节点报告存在冗余、失效、陈旧、错误多数和可靠少数时，联合选择节点、语义粒度与补传方式，能否改善任务性能—通信成本折中？

---

## 2. 阶段 4 的最小充分创新

硕士论文采用“2+1”结构，不以网络规模衡量创新。

### C1：资源 regret 感知的频谱语义生成

将频谱前端从只优化检测 F1/mAP，扩展为联合优化检测、资源选择 regret、漏占用风险、实际码率和校准误差。

### C2：反事实边际任务价值/bit 的节点—语义粒度联合调度

学习或计算一份报告在当前融合状态下可减少多少任务损失，并在实际 bit、时延和链路可靠性约束下决定：

- 哪个节点发送；
- 发送什么粒度；
- 发送多少语义；
- 是否需要补充节点或升级证据。

### C3 增强：可靠少数保护与任务触发补传

解决错误多数压制唯一可靠节点、高 SNR 节点前端失真、关键节点掉线和信息陈旧问题。C3 是鲁棒性增强，不得喧宾夺主。

### 明确不扩展的方向

- 不新增图像语义编码研究；
- 不为了标签引入 MIMO、AirComp、轨迹优化或复杂强化学习；
- 不将网络换成 Transformer/GNN 本身视为创新；
- 不以理想标签压缩倍数或检测 F1 单独证明任务导向通信有效；
- 不在阶段 4 主线中研究完整 I/Q 重建。

---

## 3. 优化问题

阶段 4 建议求解：

$$
\begin{aligned}
\min_{\theta,\phi,\psi}\quad
&\mathbb E[\mathcal R_{\mathrm{occ}}]
+\beta\,\mathrm{CVaR}_{\alpha}(\mathcal R_{\mathrm{occ}})
+\gamma\,\mathcal L_{\mathrm{miss}}\\
\text{s.t.}\quad
&\mathbb E[B_{\mathrm{tx}}]\le B_{\max},\\
&\Pr(D_{\mathrm{e2e}}>D_{\max})\le\epsilon_d.
\end{aligned}
$$

其中：

- $\theta$：本地语义生成器；
- $\phi$：节点—语义粒度调度器；
- $\psi$：融合与资源决策器；
- $\mathcal R_{\mathrm{occ}}$：真实所选资源与 oracle 资源之间的 occupancy regret；
- $\mathrm{CVaR}_{\alpha}$：最差一部分场景的尾部 regret；
- $\mathcal L_{\mathrm{miss}}$：漏报占用资源的非对称惩罚；
- $B_{\mathrm{tx}}$：预览、协议头、CRC、FEC、填充、重传和反馈的实际总 bit；
- $D_{\mathrm{e2e}}$：检测、编码、排队、传输、反馈和融合总时延。

---

## 4. 总体算法结构

```text
同一频谱场景的多节点观测
→ 本地频谱检测与质量诊断
→ G0—G3 多粒度语义候选
→ 全节点低开销语义预览
→ 反事实边际任务价值预测
→ 节点—粒度—语义联合选择
→ 阶段 2 公平数字链路
→ 阶段 3 ACK 候补与风险回退
→ 可靠少数保护融合
→ 连续资源块选择
→ regret / clean rate / bit / latency / CVaR
```

---

## 5. 模块 A：多粒度数字语义生成

### 5.1 粒度定义

| 粒度 | 内容 | 用途 |
|---|---|---|
| G0 | 静默 | 低价值、严重预算不足 |
| G1 | 1—2 bit/子信道的粗占用预览 | 全节点低开销调度输入 |
| G2 | 量化占用概率、置信度和质量元数据 | 常规融合主载荷 |
| G3 | 时频框、类别、软概率和不确定性 | 精细资源选择或冲突复核 |
| G4（可选） | 局部谱图或短 I/Q 证据 | 高风险异常补传，不作为必做项 |

### 5.2 质量特征

每节点至少计算：

- sensing SNR；
- 预测置信度与熵；
- ADC/样本削顶比例；
- 全频带泄漏比例；
- 噪声底稳定性；
- 峰值—背景功率比；
- 信息年龄；
- 上报成功概率；
- 历史校准误差或单节点任务可靠度。

注意：SNR 只能作为一个输入，禁止再使用“SNR 越高必然越可靠”的硬单调假设。

### 5.3 数字化

- 占用概率支持 1/2/4/8 bit 量化；
- 训练采用 STE 或 soft-to-hard annealing；
- 推理必须使用真实离散量化；
- 所有粒度都通过阶段 2 的相同分包、FEC、重传和 bit 审计；
- 预览和调度反馈也必须计入成本。

---

## 6. 模块 B：反事实边际任务价值估计

### 6.1 价值定义

对节点 $n$ 的粒度 $g$：

$$
\Delta V_{n,g}
=
\mathcal L_{\mathrm{task}}(q)
-
\mathcal L_{\mathrm{task}}(q\oplus m_{n,g}),
$$

其中 $q$ 是当前融合信念，$m_{n,g}$ 是候选消息。它表示“接收这份报告预计减少多少任务损失”。

训练集可利用真值生成反事实标签；部署时由价值网络预测：

$$
\widehat{\Delta V}_{n,g}
=f_{\phi}(h_n,q,\mathrm{CSI}_n,\mathrm{age}_n,g),
$$

其中 $h_n$ 是已审计 bit 的低开销预览。测试和部署阶段不得读取真值。

### 6.2 条件价值/成本评分

$$
S_{n,g}
=
\frac{
\widehat{\Delta V}_{n,g}
-\eta D_{n,g}
+\kappa C_{n,g}^{\mathrm{risk}}
}{
\mathbb E[B_{n,g}]
+\lambda_d\mathbb E[D_{n,g}]
}.
$$

- $D_{n,g}$：与已选择报告的冗余度；
- $C_{n,g}^{\mathrm{risk}}$：对高风险频段或可靠少数的覆盖价值；
- $\mathbb E[B_{n,g}]$：考虑 FEC 和重传后的预期 bit；
- $\mathbb E[D_{n,g}]$：预期链路时延。

阶段 4 的价值必须是当前场景和当前已选集合条件下的边际价值，不再只使用阶段 3 的验证集平均节点先验。

---

## 7. 模块 C：节点—语义粒度联合选择

### 7.1 离散约束

$$
\sum_g m_{n,g}\le 1,\qquad m_{n,g}\in\{0,1\},
$$

$$
\sum_{n,g}m_{n,g}\mathbb E[B_{n,g}]\le B_{\max}.
$$

### 7.2 两级实现

1. **可解释基线**：条件边际价值/成本贪心或 0-1 背包；
2. **学习方法**：Gumbel-Softmax 或 Hard-Concrete 门控，训练后转为离散决策。

### 7.3 预算约束

优先采用原始—对偶更新：

$$
\mathcal L_{\mathrm{Lag}}
=\mathcal L_{\mathrm{task}}
+\lambda_B(\mathbb E[B]-B_{\max}),
$$

$$
\lambda_B\leftarrow
[\lambda_B+\eta_{\lambda}(\mathbb E[B]-B_{\max})]_+.
$$

不得为每个测试预算手工寻找一组只对测试集有效的权重。

---

## 8. 模块 D：可靠少数保护融合

### 8.1 融合结构

基础加权：

$$
p_k^{\mathrm{mean}}
=\frac{\sum_n w_{n,k}p_{n,k}}{\sum_n w_{n,k}}.
$$

保守少数证据：

$$
p_k^{\mathrm{safe}}
=\max_n(r_{n,k}p_{n,k}),
$$

最终融合：

$$
p_k^{\mathrm{fuse}}
=(1-\rho_k)p_k^{\mathrm{mean}}
+\rho_kp_k^{\mathrm{safe}}.
$$

$\rho_k$ 根据报告分歧、可靠少数、融合不确定性和前端异常诊断产生。

### 8.2 设计边界

- 禁止直接使用全局 max 作为完整方法，以免把所有频段判占用；
- 高置信度不等于高可靠度，必须经过质量诊断与独立校准；
- 当冲突无法判断时，优先触发补传而不是强行给出高权重；
- 必须同时报告漏占用和误报造成的频谱利用率损失。

---

## 9. 模块 E：任务触发 ACK 候补与粒度升级

### 9.1 首轮

所有节点发送 G1 预览；调度器选择首轮节点和粒度。

### 9.2 触发条件

- 报告未成功译码；
- 成功报告数不足；
- 融合不确定性超过阈值；
- 最优与次优资源块预测占用差距过小；
- 高质量少数与多数冲突；
- 预计 task regret 或尾部风险过高；
- 信息年龄超过阈值。

### 9.3 补传顺序

1. 请求下一高边际价值节点；
2. 将已有节点从 G1 升至 G2；
3. 对冲突节点从 G2 升至 G3；
4. 可选请求局部谱图证据；
5. 风险极高时回退至全节点上报。

该机制必须与 CRC 失败重传、固定重复和阶段 3 ACK 候补分别比较。

---

## 10. 联合损失

建议完整损失：

$$
\begin{aligned}
\mathcal L_{\mathrm{joint}}
=&\lambda_{\mathrm{det}}\mathcal L_{\mathrm{det}}
+\lambda_{\mathrm{res}}\mathcal L_{\mathrm{soft-regret}}\\
&+\lambda_{\mathrm{miss}}\mathcal L_{\mathrm{miss}}
+\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{Brier}}\\
&+\lambda_{\mathrm{tail}}\mathrm{CVaR}_{\alpha}(\mathcal R_{\mathrm{occ}})
+\lambda_{\mathrm{red}}\mathcal L_{\mathrm{redundancy}}\\
&+\lambda_B(\mathbb E[B_{\mathrm{tx}}]-B_{\max}).
\end{aligned}
$$

可微资源选择采用：

$$
\pi(a)=\frac{\exp(-\tau\hat O_a)}{\sum_j\exp(-\tau\hat O_j)},
$$

$$
\mathcal L_{\mathrm{soft-regret}}
=\sum_a\pi(a)O_a^{\mathrm{true}}-\min_aO_a^{\mathrm{true}}.
$$

推理阶段仍使用离散资源块选择，不使用 softmin 结果冒充实际决策。

---

## 11. 训练流程

### T4.1：冻结前端并打通多粒度链路

- 冻结当前 detector；
- 实现 G0—G3 编解码；
- 验证经过阶段 2 链路后可恢复；
- 增加 packet 和 bit 单元测试。

### T4.2：训练质量诊断器

- 使用 calibration split 拟合置信度与实际任务可靠度；
- 输出 Brier、ECE、可靠性图和异常检测指标；
- 检查削顶、泄漏和 SNR 的贡献。

完成状态：已在 calibration 拟合单调置信度校准器，并在 validation 输出 Brier、ECE、可靠性表及七类受控异常的检测敏感性；证据见 `results/stage4/quality_diagnostics_v1/`。受控注入只验证诊断量响应，不替代真实异常发生率或 H3 证据。

### T4.3：生成反事实价值标签

- 仅在 train split 枚举节点、粒度、已选集合和信道条件；
- 保存加入报告前后的任务损失差；
- 审计标签生成没有读取 validation/test 真值。

### T4.4：预训练价值网络

报告：

- 价值回归 MAE；
- 候选排序 Spearman 相关；
- top-k 高价值报告召回率；
- 选择后的真实 regret 下降。

完成状态：已补齐 validation 聚合 MAE、Spearman、top-k 与实际选择 regret；逐 scene 候选标签未保存。近零排序相关和低 top-k 召回进一步支持 Gate B 的负决定，证据见 `results/stage4/value_diagnostics_v1/`。

### T4.5：训练调度与融合

- detector 保持冻结；
- 比较贪心/背包与学习式门控；
- 使用 train 优化、validation 选择。

完成状态：贪心、精确多选择背包、五 seed 学习门控和稀疏残差变体均已实现并完成 Gate B；系统级门槛未通过，按 D4-007 冻结为负结果。

### T4.6：小学习率联合微调

- 只解冻 detector 后部或 occupancy head；
- 加入 resource-aware loss；
- 保留 detector 完全冻结版本作为消融。

处置状态：按 D4-003 正式豁免，不再在同一 validation 上继续搜索联合权重。已完成的 full-joint 对照未通过，最终候选保留 detector 冻结的 `detection+resource`，避免事后调参。

---

## 12. 阶段 4 数据协议

### 12.1 新注册表

必须建立：

```text
configs/stage4_dataset_registry_v1.json
configs/stage4_protocol.json
```

建议按 scene 划分：

- train：60%；
- calibration：10%；
- validation：15%；
- final holdout：15%。

同一原始 scene、连续记录或辐射源实例不得跨集合。

### 12.2 主算法数据

- 使用稀疏和密集频谱场景；
- 资源块数覆盖 4/8/16；
- 同一 scene 产生关联多节点观测；
- 各节点具有不同 SNR、增益、多径、频偏、age 和前端异常；
- 场景占用真值保持一致；
- 随机无关帧叠加只能作为压力测试，不能证明空间协同。

### 12.3 真实数据

现有 LoRaIQ 两事件仅作为接口和失效机制 pilot。正式 H3 最低建议：

- 至少 30 个独立 RF scene，推荐 50—100 个；
- 至少 4 个同步接收节点；
- 不同中心频率、带宽、LoS/NLoS 和占用组合；
- 包含可靠少数、多个可靠节点、前端饱和和节点失效；
- 同一 scene 的链路随机重复不得视作独立真实样本。

若真实 LoRa 始终占用中心半带，必须采用多中心频点回放、多信号源或 SDR 多子带发射，避免资源任务饱和。

---

## 13. 基线体系

### 13.1 表示基线

- hard semantic；
- 强 PSD 量化；
- compressed spectrogram；
- 固定 G1/G2/G3；
- 固定 top-k；
- confidence top-k；
- uncertainty top-k；
- RAVES 动态粒度。

### 13.2 节点选择基线

- 全节点；
- 固定最佳节点；
- 动态 SNR 最佳节点；
- 随机与轮询；
- 阶段 3 先验 value/bit；
- 反事实边际 value/bit；
- oracle 节点选择上界。

### 13.3 融合基线

- OR、majority、mean、median；
- SNR weighted；
- 阶段 3 quality weighted；
- DeepSets；
- 可靠少数保护；
- oracle 融合上界。

### 13.4 重传基线

- 无重传；
- CRC 失败重传；
- 固定重复；
- 阶段 3 ACK 候补；
- RAVES 任务风险触发补传。

公平要求：所有方法使用相同原始场景、相同链路随机数、相同 FEC、相同重传上限，以及相同 bit 或时延预算。

---

## 14. 消融实验

完整方法至少包含以下消融：

1. 去除 resource-aware loss；
2. 去除 rate 约束；
3. 用 confidence 替代反事实价值；
4. 去除冗余惩罚；
5. 去除 CVaR；
6. 去除可靠少数保护；
7. 去除异常质量诊断；
8. 去除 ACK 候补；
9. 固定语义粒度；
10. detector 冻结；
11. detector 联合微调；
12. 完整 RAVES。

一次消融只改变一个主要因素。

---

## 15. 实验矩阵

| 变量 | 计划取值 |
|---|---|
| 节点数 | 1、2、4、8 |
| 资源块数 | 4、8、16 |
| 连续带宽需求 | 1、2、4 个资源块 |
| 总预算 | 0.5、1、2、4、8 kbit/decision |
| 感知 SNR | -12、-6、0、6、12 dB及异质组合 |
| 上报 Eb/N0 | 0、3、6、9、12 dB |
| 节点失败率 | 0、0.25、0.5 |
| age | 0、1、2、4 个决策周期 |
| 异常 | 高置信错误、削顶、全带泄漏、频移 |
| 可靠少数 | 1 对 3、2 对 2 |
| FEC | 阶段 2 Hamming 与 LDPC 参考链路 |
| 重传 | 0、1、2 次 |

主指标：occupancy regret、clean rate、missed-occupancy、实际 bit、goodput、端到端时延、CVaR。  
次指标：F1/mAP、Brier、ECE、模型参数、MACs、推理时间和参考能耗。

开发覆盖状态：基线、消融和矩阵逐项审计见 `results/stage4/matrix_coverage_v1/`。19项已完成、7项因 Gate B/C 或 D4-003 正式豁免、2项依赖新增 final/H3 数据、0项无解释失败。豁免只表示不再对失败分支调参，不表示获得正结果。C1/C2复杂度见 `results/stage4/complexity_profile_v1/`；CPU时延不是无人机板端时延。

---

## 16. 统计协议与成功门槛

### 16.1 统计单位

- 真实实验以独立 RF scene 为统计单位；
- 链路随机重复嵌套在 scene 内；
- 使用配对比较和层次 bootstrap；
- 至少 5 个训练随机种子；
- 报告均值、标准差、95% CI 和效应量；
- 不允许只选择最优种子。

### 16.2 H2 成功条件

final holdout 上满足以下至少一项：

1. 相同实际 bit 下，完整方法 regret 差的 95% CI 低于 0；
2. 相同 regret 下，实际发送 bit 显著降低；
3. bit—regret Pareto 前沿优于最强基线；
4. 平均 regret 非劣，且 CVaR 显著改善。

非劣界在 validation 上根据任务尺度预注册，建议以“随机—oracle regret 差距”的固定比例定义，禁止读取 final holdout 后修改。

### 16.3 H3 成功条件

只有在足量真实或可信关联多节点 scene 上，完整方法显著优于最佳单节点、SNR 加权、全节点平均和阶段 3 自适应调度，才能宣称 H3 成立。

若数据量不足，只能报告通信节流、失效恢复和真实接口可行性，不声称空间协同增益。

---

## 17. 三个算法决策门

### Gate A：resource-aware loss

比较 detection-only、detection+rate、detection+resource 和完整联合损失。

- 若相同 bit 下 regret 改善：保留为 C1；
- 若无改善：先检查任务饱和、资源块数量、softmin 温度、标签和占用计算；
- 排除实验问题后仍无效：冻结 detector，将论文重点转向 C2 调度，保留负结果。

### Gate B：反事实 value/bit

比较 SNR、置信度、阶段 3 先验 value/bit、反事实 value/bit 和 oracle。

- 若排序和实际任务性能改善：进入动态粒度联合训练；
- 若学习价值不稳定：退回反事实标签 + 离散贪心，不强行使用神经网络；
- 若所有方法接近：检查资源任务是否饱和，而不是继续加模型。

### Gate C：可靠少数保护

在 1 对 3、高置信错误、饱和节点、关键节点失效和陈旧场景评估。

- 若尾部风险改善且误报可控：作为 C3；
- 若误报明显增加：改为冲突触发补传，不直接改变融合结果；
- 若仍无收益：保留 mean/median，论文不宣称学习融合创新。

---

## 18. 14 周执行表

| 周次 | 工作 | 交付物 | 状态 |
|---:|---|---|---|
| 1 | 冻结问题、文献矩阵和阶段 4 协议 | 本文件、protocol草案 | completed |
| 2 | 新建 scene 级 train/cal/val/holdout | dataset registry、泄漏测试 | completed；C1-v4 Final为280景、74簇、旧provenance重叠0 |
| 3 | 实现 G0—G3 与实际 bit 审计 | packet模块和单测 | completed |
| 4 | 实现削顶、泄漏、噪声底等质量特征 | 质量诊断报告 | completed；calibration/validation诊断已归档 |
| 5 | 生成反事实价值标签 | utility数据集与审计 | completed |
| 6 | 训练价值预测器 | MAE、排序、top-k报告 | completed；当前网络未通过Gate B |
| 7 | 实现贪心/背包联合调度 | 可解释算法基线 | completed；Gate B负结果已冻结 |
| 8 | 实现可微粒度选择与对偶预算 | 学习调度器 | waived by D4-003/D4-007；学习门控已验证失败，不继续调参 |
| 9 | 实现可靠少数保护 | 1对3失效测试 | completed；Gate C未通过，回退经典融合 |
| 10 | 接入阶段2链路和阶段3候补 | 端到端闭环 | completed |
| 11 | 完成训练、校准和validation选型 | 冻结模型、阈值、预算 | completed；development候选已冻结 |
| 12 | 运行预算、SNR、节点、age sweep | 主结果表 | channel/SNR completed；其余由C3场景覆盖 |
| 13 | 运行消融、CVaR和统计分析 | 消融、CI与论文图表 | completed；development synthesis、自动论文资产和章节草稿已生成 |
| 14 | 一次性运行 final holdout | 正式阶段4报告 | completed；原Final与C1-v4跨日期Final访问均已消费一次 |

真实多接收机数据扩充从第 1 周开始并行，不等待算法完成。

---

## 19. Go/No-Go 与回退方案

### 联合训练不稳定

回退为：冻结 detector → 监督价值网络 → 离散贪心调度 → 固定融合器。该路线仍可形成硕士论文算法。

### 价值网络不优于 confidence top-k

检查反事实标签、场景饱和、当前融合信念输入和数字链路成本；仍无改善则使用解析反事实贪心作为主方法。

### 可靠少数保护导致误报

限制保守融合比例，增加前端异常诊断和证据补传；禁止退化为全局 max。

### 真实数据任务饱和

改变中心频率、占用组合、资源块数和连续带宽需求；不能通过重复同一中心半带事件伪造样本量。

### 学习融合仍不优于平均

接受负结果，使用 mean/median 作为融合后端，将创新收敛到 C1+C2。

### final holdout 不支持主假设

不得继续用 final holdout 调参。报告失效条件，必要时把结论降级为非劣、通信节流或方法边界。

---

## 20. 注意事项清单

### 科研规范

- [x] 每个正式实验映射到 H2/H3/H4；
- [x] 修改前登记假设和目标指标；
- [x] final 未参与阈值和权重选择；
- [x] 开发集同一 scene 未跨 split；final 注册器另行强制泄漏检查；
- [x] 所有配对方法使用相同 scene/链路随机条件；
- [x] 实际 bit 包含预览、反馈、FEC、填充和重传；
- [x] 负结果与失效场景完整保存；
- [x] 不将链路重复当作独立 RF scene；
- [x] 不用检测 F1 单独证明资源任务有效；
- [x] 不把网络复杂度本身写成创新。

### 工程规范

- [x] 新模块有单元测试；
- [x] 配置、代码版本和数据校验值写入结果元数据；
- [x] 训练、校准、验证、final 入口分离；
- [x] 核心结果可由脚本生成 JSON/Markdown；
- [x] 保存模型、阈值、量化器和协议版本；
- [x] 不覆盖历史结果目录；
- [x] Final正式结果目录只由冻结配置生成；C1-v4执行后67文件快照变化数为0。

### 论文表述

- [x] 半合成关联场景不称真实飞行；
- [x] LoRaIQ pilot 不用于宣称 H3；
- [x] 非劣不写成显著优越；
- [x] oracle 只作为上界；
- [x] 回退机制的反馈 bit 和等待时延均报告；
- [x] 明确算法在哪些场景失效。

---

## 21. 动态状态区

> 每次继续工作时首先更新本节。

### 当前阶段

- 状态：`stage4_research_complete_thesis_integration`
- 当前周：`W20_c1_v4_final_and_closeout`
- 当前数据注册表：C1-v4 Final `280 scenes / 74 site-hour clusters / overlap 0`
- 当前正式算法版本：`c1_v4_best_block_indicator1_age_guard_60min`
- Final状态：`access_consumed / access_count=1 / overall_final_success=true`

### 已完成批次

- Batch 1（2026-07-16）：冻结阶段 4 协议和数据访问规则；实现 G0—G3 应用层bit流、量化、编解码、G3局部证据及阶段2公平链路审计；实现削顶、带外泄漏、噪声底稳定性、峰背比与预测熵特征；全项目83项测试通过。详细记录见 `docs/STAGE4_BATCH1_REPORT.md`。
- Batch 2（2026-07-16）：实现离散/可微occupancy regret、miss-risk、CVaR、Brier、归一化bit约束及投影对偶更新；冻结Gate A四种损失接口；任务难度v1失败后保留负结果，v2在阈值不变时选出8源、8资源块、连续需求4的H2开发任务；全项目95项测试通过。详见 `docs/STAGE4_BATCH2_REPORT.md`。
- Batch 3（2026-07-16）：冻结开发scene与detector缓存，完成四方法×五seed控制训练和5×150层次配对bootstrap。`detection+resource`在近似同bit下显著降低平均regret与CVaR，validation阶段保留C1；`+rate`和当前full joint未通过，完整保留负结果。详见 `docs/STAGE4_BATCH3_REPORT.md`。
- Batch 4（2026-07-16）：完成四节点受控缓存、18000个train-only反事实标签、因果特征审计和五seed价值网络。一步oracle空间充足，但当前1-bit G1价值网络未稳定超过SNR/阶段3先验；修正全节点G1预览总bit后，Gate B暂不保留神经网络主分支。全项目117项测试通过。详见 `docs/STAGE4_BATCH4_REPORT.md`。
- Batch 5（2026-07-16）：完成G1 1-bit/2-bit单变量配对诊断。2-bit多用75.15 bit，regret改善CI跨0、clean不变、Brier显著恶化并压缩升级oracle空间，因此淘汰q2、保留q1，不重复五seed训练。全项目119项测试通过。详见 `docs/STAGE4_BATCH5_REPORT.md`。
- Batch 6（2026-07-16）：实现任务歧义—分歧—可靠性解析评分、train-only PAV校准、允许G0的贪心和精确多选择背包。三split四节点G1完全同质，原始分数恒0，解析方法退化为阶段3先验；保留负结果并定位到预览可观测性，而非求解器。详见 `docs/STAGE4_BATCH6_REPORT.md`。
- Batch 7（2026-07-16）：实现113-bit稀疏残差调度草图、付费紧凑质量、全节点与选择性节点缓存。草图恢复可辨识性，解析G0调度在局部对照中改善，但系统级固定开销和calibration/validation一致性未通过；五seed网络再次失败，C2停止扩展。全项目124项测试通过。详见 `docs/STAGE4_BATCH7_REPORT.md`。
- Batch 8（2026-07-16）：完成七类受控故障、审计质量、可靠少数直接保护和确认式G3补传。直接保护未激活；确认补传改善特定故障但误触发、额外bit和pooled CVaR未过Gate C，回退经典融合。全项目127项测试通过。详见 `docs/STAGE4_BATCH8_REPORT.md`。
- Batch 9（2026-07-16）：完成150 scene×3信道×4 Eb/N0×5嵌套重复的真实解码路径sweep和scene配对bootstrap。选择性G2在全部信道点显著少于全G2 bit，但regret差CI均跨0；生成开发证据综合。详见 `docs/STAGE4_BATCH9_REPORT.md` 与 `docs/STAGE4_DEVELOPMENT_SYNTHESIS.md`。
- Batch 10（2026-07-16）：补齐质量校准/异常敏感性与价值网络 MAE、Spearman、top-k 聚合诊断；实现机器可读验收审计、独立 final catalog 泄漏检查、单次访问状态机和冻结快照。development 验收全部通过，final 明确阻塞于 0/200 新独立 scene；全项目137项测试通过。详见 `docs/STAGE4_BATCH10_REPORT.md`。
- Batch 11（2026-07-16）：完成冻结C1/C2参数量、线性MAC、checkpoint体积及单线程CPU时延审计，并逐项整理基线、消融与实验矩阵覆盖。开发项19项通过、7项按决策门豁免、0项失败；2项final/H3矩阵仍依赖外部数据；全项目141项测试通过。详见 `docs/STAGE4_BATCH11_REPORT.md`。
- Batch 12（2026-07-16）：从冻结JSON自动生成4张论文图、2份CSV和主表，完成阶段4方法—实验—讨论章节草稿、声明矩阵及访问前锁定的final结果模板；新增链接、哈希、占位符和final访问声明审计。论文材料审计5/5通过，全项目145项测试通过。详见 `docs/STAGE4_BATCH12_REPORT.md`。
- Batch 13（2026-07-16）：冻结final scene级指标schema与统计引擎，实现10,000次配对scene bootstrap、CVaR、单侧配对随机化检验、非劣门槛、intersection-union族p值和两族Holm校正；合成成功/失败/坏输入三分支全部通过，全项目148项测试通过。详见 `docs/STAGE4_BATCH13_REPORT.md`。
- Batch 14（2026-07-16）：逐条核验7篇2024—2026年相关工作，建立研究边界矩阵、BibTeX和机器可读注册表；纠正一处arXiv版本题名漂移，并将文献完整性与“非MIMO/C2—C3不得外推”规则纳入自动验收。材料审计6/6通过，全项目149项测试通过。详见 `docs/STAGE4_BATCH14_REPORT.md`。

### 当前最近任务

1. 不再修改或重跑C1-v4 Final算法、60分钟阈值、样本、随机种子和主指标；
2. 正式写作以`docs/STAGE4_THESIS_CHAPTER_FINAL.md`和`docs/STAGE4_CLAIM_MATRIX.md`为权威边界；
3. 将自动生成的三张Final图和两份CSV纳入论文与答辩材料；
4. 新数据集仅作为可选外部效度增强，必须保持当前算法冻结，不能覆盖已消费Final；
5. 继续明确不宣称无人机移动、MIMO、原始I/Q、实测回传和目标硬件实时性。

### 当前阻塞项

- 核心阶段4研究没有剩余阻塞项。
- 跨数据集、真实无人机移动、实测上报链路和目标硬件实验属于后续增强，不是本阶段完成条件。

### 最新诊断

- Gate A任务难度审计v1已完成：对train/validation上的1、2、4源压力场景，以及4/8/16资源块和1/2/4连续需求共54组组合进行truth-only审计；无组合满足预注册的非饱和门槛。
- 该负结果保存在 `results/stage4/task_difficulty_audit/`，不覆盖、不删除。
- 依据Gate A的“任务饱和”失败分支，v2只扩展每scene源数至6、8，没有放宽random-oracle gap、歧义率或clean-rate阈值。
- v2已选中8源、8资源块、连续需求4：validation oracle clean rate 0.8200，mean random-oracle gap 0.026813，mean oracle margin 0.013327，ambiguous rate 0.2133。

---

## 22. 决策记录模板

每次算法方向变化追加一条，不删除历史记录。

```text
决策编号：D4-XXX
日期：YYYY-MM-DD
对应假设：H2/H3/H4
发现的问题：
使用的数据范围：train/calibration/validation（不得为final holdout）
拟修改模块：
唯一主要变化：
预期影响指标：
成功规则：
失败回退：
实验结果：
最终决定：保留/修改/放弃
是否更新论文贡献：是/否
```

---

## 23. 周更新模板

```text
周次：WXX
本周目标：
完成内容：
新增代码/配置：
运行实验：
主要结果及95%CI：
负结果与异常：
是否触碰final holdout：否（默认）
对应决策门：A/B/C/无
下周唯一优先事项：
计划变更：无/见D4-XXX
```

---

## 24. 变更记录

| 版本 | 日期 | 变化 | 原因 | final holdout是否已访问 |
|---|---|---|---|---|
| v1.0 | 2026-07-16 | 建立阶段4总计划、算法结构、实验协议、决策门和维护规则 | 承接阶段3负结果并启动C1+C2联合优化 | 否，尚未建立 |
| v1.1 | 2026-07-16 | 启动第一执行批次，登记协议、数据注册、多粒度语义与质量特征任务 | 用户授权按闭环持续推进；先完成后续算法依赖项 | 否，尚未建立 |
| v1.2 | 2026-07-16 | 完成 Batch 1；测试基线更新至83项；转入 Gate A 资源损失实现 | 协议、codec和质量特征已通过全量回归 | 否，尚未建立 |
| v1.3 | 2026-07-16 | 记录Gate A任务审计v1负结果；新增D4-002并扩展场景密度 | 54组组合均未通过冻结非饱和门槛 | 否，尚未建立 |
| v1.4 | 2026-07-16 | 完成Batch 2资源损失、任务审计v2与95项回归；转入Gate A控制训练 | v2在不改门槛下找到非饱和开发任务 | 否，尚未建立 |
| v1.5 | 2026-07-16 | 冻结Batch 3开发scene清单；测试更新至97项 | 消除运行时抽样并确保开发split可复现 | 否，尚未建立 |
| v1.6 | 2026-07-16 | 生成冻结detector缓存并实现统一变位宽head；测试更新至104项 | 为Gate A四方法控制训练提供相同输入与模型 | 否，尚未建立 |
| v1.7 | 2026-07-16 | 完成Gate A控制训练与层次bootstrap；C1保留，转入C2反事实价值标签 | resource loss在同bit下改善regret和CVaR；full joint未通过 | 否，尚未建立 |
| v1.8 | 2026-07-16 | 完成Batch 4反事实标签与五seed价值网络；修正G1预览总bit；网络未通过Gate B | 平均regret/bit未稳定超过简单基线，下一批只检查2-bit预览可辨识性 | 否，尚未建立 |
| v1.9 | 2026-07-16 | 完成1-bit/2-bit G1配对诊断；淘汰2-bit并转入解析离散调度 | q2成本增加、regret无显著改善、Brier和升级空间显著恶化 | 否，尚未建立 |
| v2.0 | 2026-07-16 | 完成解析评分、PAV、贪心与精确背包；定位G1节点同质化 | 原始分数恒0，方法退化为阶段3先验；下一批测试付费稀疏残差草图 | 否，尚未建立 |
| v2.1 | 2026-07-16 | 完成全节点/选择性残差草图与五seed复验；C2停止扩展，转入C3 | 可辨识性恢复但系统固定开销和跨split稳定性未通过 | 否，尚未建立 |
| v2.2 | 2026-07-16 | 完成C3直接保护与确认式G3补传；Gate C未通过 | 特定故障有收益，但误触发、额外bit和pooled CVaR不满足门槛 | 否，尚未建立 |
| v2.3 | 2026-07-16 | 完成端到端三信道解码sweep、scene bootstrap和开发证据综合 | 开发系统已冻结，剩余关键条件是新增独立final scene | 否，尚未建立 |
| v2.4 | 2026-07-16 | 预注册final两主检验、非劣界、Holm校正、最小scene与一次性运行手册 | 防止获得final后选择有利门槛；等待外部新增独立scene | 否，尚未建立 |
| v2.5 | 2026-07-16 | 完成开发验收审计、质量/价值补充诊断、final注册与单次访问工具；测试更新至137项 | 将“完成、负结果、豁免、外部阻塞”机器可读地区分，防止误触final | 否，访问计数0 |
| v2.6 | 2026-07-16 | 补齐复杂度/CPU时延与基线—消融—实验矩阵覆盖审计 | 总计划§13—15的未归档项必须在final前闭环；失败分支以门控豁免记录而非伪造完整方法 | 否，访问计数0 |
| v2.7 | 2026-07-16 | 生成可追溯论文图表、章节草稿、声明矩阵和final结果锁定模板；测试更新至145项 | 将实验JSON到论文结论的抄录和过度声明风险纳入自动验收 | 否，访问计数0 |
| v2.8 | 2026-07-16 | 冻结final scene指标schema、配对bootstrap、随机化p值、IUT family p值与Holm判决器；测试更新至148项 | 防止看到final后改变聚合层级、p值算法或多重校正规则 | 否，访问计数0 |
| v2.9 | 2026-07-16 | 核验近期相关工作，建立定位矩阵、BibTeX、机器注册表及自动审计；测试更新至149项 | 防止题名漂移、错误MIMO归类和以外部文献替代本项目证据 | 否，访问计数0 |
| v3.0 | 2026-07-21 | 完成AERPAW `rf32_le`功率扫频适配、永久排除pilot治理、端到端dry-run和完整链路CPU剖析 | 官方数据是功率扫频而非复数I/Q，必须先解决输入语义和单站点证据边界，不能因文件数充足就自动注册final | 否，访问计数0 |
| v3.1 | 2026-07-21 | 冻结2.4 GHz资源代理、三站点时间对齐/30分钟独立性规则、ZIP精简抽取、三节点链路排列和M1失败回退 | CC1/CC2下载后既要避免磁盘不足，也要在任何模型输出前解决三站点是否真正同步以及四节点到三节点的声明变化 | 否，访问计数0 |
| v4.0 | 2026-07-22 | 完成C1-v4任务充分块语义、60分钟时效保护零比特回退、280景双站点单次Final、自动论文资产与收尾审计 | H1与H2固定顺序检验全部通过；阶段4核心证据链闭环并转入论文整合 | 是，C1-v4访问计数1 |

---

## 26. 已执行决策记录

### D4-001：阶段 4 注册表先冻结协议，不虚构 final holdout

- 日期：2026-07-16
- 对应假设：H2、H3、H4
- 发现的问题：RadDet 各目录已经在阶段1—3使用；LoRaIQ只有两个独立事件，均不满足新final holdout条件。
- 使用的数据范围：仅检查既有注册表、目录结构与历史使用状态；未读取新final holdout标签或模型输出。
- 唯一主要变化：建立 `protocol_only` 注册表，允许开发拆分规则冻结但禁止伪造scene ID。
- 预期影响指标：数据泄漏为零；final holdout访问次数保持0。
- 成功规则：协议校验、跨split泄漏检测和final访问规则均通过自动测试。
- 实验结果：4项协议测试通过；全量83项测试通过。
- 最终决定：保留。
- 是否更新论文贡献：否；属于科研规范与复现基础。

### D4-002：Gate A任务难度v1失败后增加场景密度

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：54组truth-only任务中没有组合通过全部非饱和筛选；资源选择任务仍偏稀疏或存在大量oracle并列。
- 使用的数据范围：RadDet train/val开发范围；未使用final holdout。
- 拟修改模块：任务压力生成，不修改算法损失。
- 唯一主要变化：sources/scene候选从1、2、4扩展为1、2、4、6、8。
- 预期影响指标：提高mean random-oracle gap并降低oracle clean饱和率。
- 成功规则：保持gap至少0.02、ambiguous rate不高于0.8、oracle clean rate位于0.2—0.98，全部阈值不变。
- 失败回退：若仍无候选，停止无关帧叠加路线，转向显式可控占用场景生成器。
- 实验结果：v2共90组train/validation组合；验证选中8源、8资源块、连续需求4，gap 0.026813、歧义率0.2133、oracle clean 0.8200。
- 最终决定：保留v1负结果，冻结v2选择并进入Gate A训练。
- 是否更新论文贡献：否；仅H2开发任务诊断。

### D4-003：Gate A保留resource loss，拆分不稳定rate/full joint

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：单纯rate项稳定降低bit但regret倾向恶化；当前full joint对regret无稳定增益且Brier明显变差。
- 使用的数据范围：train拟合、validation选型与层次bootstrap；final holdout未建立、未访问。
- 拟修改模块：C1冻结为detection+resource；码率优化转入C2节点—粒度调度。
- 唯一主要变化：不再继续联合调C1中的rate/miss/CVaR/Brier权重，先冻结已通过的resource项。
- 预期影响指标：保持C1任务对齐证据，避免同一validation上反复搜权重。
- 成功规则：resource相对detection-only的regret和CVaR配对95% CI均低于0，bit CI包含0。
- 实验结果：regret差−0.001729，95% CI [−0.003082,−0.000534]；CVaR差−0.006419，CI [−0.013153,−0.001264]；bit差0.78，CI [−1.83,3.53]。
- 最终决定：保留C1 resource-aware loss；rate/full joint记为当前负结果；进入C2。
- 是否更新论文贡献：是，但仅为validation阶段候选贡献，final H2仍未验证。

### D4-004：当前1-bit G1价值网络不进入主分支

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：一步oracle可将regret从0.028172降至0.005500，但可部署网络相对SNR-G3和阶段3先验的配对regret CI均跨0，且calibration性能不稳定。
- 使用的数据范围：train真值生成监督；calibration仅聚合诊断；validation仅聚合方法评价；未保存calibration/validation标签，final holdout未建立、未访问。
- 拟修改模块：G1预览可辨识性与后续离散调度，不修改C1、反事实价值定义或任务标签。
- 唯一主要变化：下一批将G1量化由1 bit/子信道提高到2 bit/子信道，并完整重算预览成本。
- 预期影响指标：提高候选value/bit排序可预测性，同时检查新增全节点预览bit是否值得。
- 成功规则：相对SNR与阶段3先验至少形成regret或bit-regret Pareto改善，且calibration/validation方向一致；不在validation上搜索多个网络结构。
- 失败回退：停止神经价值预测，采用反事实监督审计加离散贪心/背包主算法。
- 实验结果：当前学习方法validation regret 0.025075、总bit 2215.33、CVaR 0.114811；相对阶段3先验regret差−0.000294且CI跨0，同时多用62.84 bit。
- 最终决定：当前网络不通过Gate B；保留负结果，启动单变量2-bit G1诊断。
- 是否更新论文贡献：暂否；C2仍处于validation算法选择阶段。

### D4-005：淘汰2-bit G1预览

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：1-bit G1可能限制价值可辨识性，但增加量化位数也可能提高固定预览成本并压缩升级空间。
- 使用的数据范围：calibration/validation聚合配对诊断；未保存逐scene标签；final holdout未建立、未访问。
- 唯一主要变化：G1概率量化1 bit改为2 bit；其余组件完全固定。
- 成功规则：regret或bit-regret Pareto改善，且calibration/validation方向一致；Brier和升级空间不得明显恶化。
- 实验结果：q2−q1 validation regret −0.001576，CI跨0；成本+75.146 bit；Brier +0.008995且CI全高于0；oracle reduction下降0.012376且CI全低于0。
- 最终决定：淘汰q2并不重训其价值网络；恢复q1，进入解析贪心/背包。
- 是否更新论文贡献：否；作为语义预览设计消融与负结果。

### D4-006：解析调度因G1同质化退化

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：train/calibration/validation中四节点G1硬预览100%完全相同，任务相关节点分歧恒0。
- 使用的数据范围：train拟合PAV，calibration/validation聚合预算曲线；final holdout未建立、未访问。
- 唯一主要变化：加入冻结解析分数与精确多选择背包，没有修改预览或前端。
- 实验结果：2400个train候选原始分数唯一值数为1；解析方法在四预算点均与阶段3先验完全相同；600 bit预算相对SNR少70.01 bit但regret差CI跨0。
- 最终决定：保留求解器与负结果，不调公式；下一批新增不参与融合的付费top-k软残差草图。
- 是否更新论文贡献：可作为C2可观测性动机，最终贡献仍需草图或后续调度通过validation/final门槛。

### D4-007：C2草图恢复可辨识性但系统门槛未通过

- 日期：2026-07-16
- 对应假设：H2
- 发现的问题：普通G1无节点差异；付费草图能恢复差异，但可能被固定预览开销抵消。
- 使用的数据范围：train标签/校准，calibration/validation聚合评价；final holdout未建立、未访问。
- 实验结果：全节点草图成本2012.81 bit；选择性草图1775.34 bit。选择性600-bit解析贪心validation为regret 0.024436、总bit 2294.81，但calibration劣于阶段3先验，且相对普通G1-SNR系统多约72 bit。
- 网络复验：五seed集成validation regret 0.025594、总bit 2644.78，未通过且不挑单seed。
- 最终决定：停止C2结构/阈值搜索；保留完整正负结果和求解器，转入C3。
- 是否更新论文贡献：C2暂不作为已验证性能贡献，只作为可观测性—开销实验发现。

### D4-008：C3可靠少数与确认补传不进入最终候选

- 日期：2026-07-16
- 对应假设：H3/H4受控机制
- 发现的问题：直接保护在平均分歧门槛下未激活；最大通道冲突补传能修正特定故障但误触发明显。
- 使用的数据范围：validation受控故障变换；不是独立真实多接收机数据；final holdout未建立、未访问。
- 实验结果：确认G3 pooled regret 0.025304、CVaR 0.135580、触发46.48%、额外390.66 bit；良性误报触发100%且false occupancy显著上升。
- 最终决定：停止C3阈值搜索，回退固定经典融合；保留机制、成本和负结果。
- 是否更新论文贡献：否；不得宣称可靠少数或H3成立。

### D4-009：开发验收通过，算法冻结并启用 final 单次访问状态机

- 日期：2026-07-16
- 对应假设：H2、H3、H4
- 发现的问题：总计划中的质量数据报告与价值 MAE/排序/top-k 证据尚未独立归档；final 数据到位前缺少可执行的注册与防二次访问约束。
- 使用的数据范围：calibration 拟合质量校准；validation 只输出聚合质量/价值指标；未保存 validation 候选标签；未建立或访问 final。
- 唯一主要变化：补齐遗漏诊断，并新增验收审计、catalog 泄漏检查、一次性访问状态与可执行快照；不修改算法、阈值、预算或成功门槛。
- 实验结果：质量 Brier/ECE 从 `0.179472/0.422625` 降至 `0.000421/0.002756`；价值 pooled Spearman `0.00938`、top-1 recall `0.12`，支持 Gate B 负决定；机器审计 development 全通过、final 因0/200 scene阻塞。
- 最终决定：development 冻结；C1和选择性G2进入预注册 final，C2/C3不复活；final access count保持0。
- 是否更新论文贡献：不新增性能贡献；增强严谨性、复现性和负结果可信度。

### D4-010：失败分支采用门控豁免，不伪造完整RAVES实验

- 日期：2026-07-16
- 对应假设：H2、H3、H4
- 发现的问题：原始§13—15按完整RAVES列出大规模基线、消融和网格，但Gate B/C已经否决动态调度与可靠少数；继续全网格会构成同一validation上的事后搜索。
- 使用的数据范围：只复核既有阶段2—4结果和冻结checkpoint；复杂度计时不使用标签；final未建立、未访问。
- 唯一主要变化：为每个计划项登记passed/waived/blocked/failed，并补充C1/C2参数、MAC、文件体积和CPU时延；算法无变化。
- 实验结果：19项passed、7项waived、2项final/H3 blocked、0项failed；C1 head为1740参数、1664线性MAC/scene，本机单线程离散推理中位0.4238 ms。
- 最终决定：开发矩阵闭环；论文不宣称不存在的完整RAVES，豁免项按决策门解释，板端时延仍需未来硬件实测。
- 是否更新论文贡献：否；更新实验完整性与部署可行性讨论。

### D4-011：论文图表必须由冻结JSON生成并保留final占位符

- 日期：2026-07-16
- 对应假设：H2、H3、H4及论文证据边界
- 发现的问题：手工抄录多批实验数值容易产生版本漂移；章节写作可能把validation结论提前写成final结论。
- 使用的数据范围：只读取已冻结的stage2—4 JSON/配置和访问计数；不读取final数据。
- 唯一主要变化：自动生成论文图、CSV和表格，并对source/output哈希、本地链接、final访问标志和必要占位符做审计。
- 实验结果：4图、2 CSV、1汇总表及3份论文文档形成；材料审计5项全通过，final access count为0。
- 最终决定：后续论文数值只引用自动资产；`[[FINAL_PENDING_*]]`在单次final运行前不得删除。
- 是否更新论文贡献：不改变贡献，只提高论文可追溯性和声明严谨度。

### D4-012：Holm使用配对随机化p值，CI使用scene bootstrap

- 日期：2026-07-16
- 对应假设：H2两主检验族
- 发现的问题：仅登记“bootstrap+Holm”不足以唯一确定p值算法、scene聚合和族内多指标判决。
- 使用的数据范围：纯合成数值自检；未读取RF、标签或final。
- 唯一主要变化：冻结scene级schema；CI使用10,000次配对scene bootstrap；component p值使用单侧配对随机化检验；IUT family取最大p值后对两族做Holm校正。
- 成功规则：合成成功分支两族通过、失败分支两族失败、缺配对行被拒绝，且结果固定seed可复现。
- 实验结果：三条分支全部符合预期；定向8项及全项目148项测试通过。
- 最终决定：保留为唯一final统计入口；禁止访问后用Notebook替换。
- 是否更新论文贡献：否；属于正式统计严谨性。

### D4-013：近期文献只用于定位，不替代项目门控证据

- 日期：2026-07-16
- 对应假设：H2、H3及论文创新性表述
- 发现的问题：近期工作跨越AirComp、图像分类、视觉协同、MIMO预编码和自适应图像码率；仅凭“多节点”或“语义通信”关键词容易错误归类，且arXiv版本存在题名漂移。
- 使用的数据范围：出版社、作者机构、CVF和arXiv原始页面；不读取final数据。
- 唯一主要变化：核验7篇文献元数据，建立逐篇边界矩阵、BibTeX、机器注册表与自动审计。
- 成功规则：7篇核心参考全部有原始入口和完整标识；未核验条目为0；论文明确本系统非MIMO，且C2/C3失败不被外部文献包装为正结果。
- 实验结果：文献注册7/7完成，论文材料审计6/6通过，全项目149项测试通过。
- 最终决定：相关工作章节采用“可比较问题—不可迁移证据”结构；冻结后的算法结论不因新增引用改变。
- 是否更新论文贡献：细化贡献边界，不新增未经实验支持的贡献。

### W01 执行更新

- 本周目标：冻结阶段4协议并完成G0—G3最小可运行语义链路。
- 完成内容：协议、注册表、codec、质量特征、阶段2链路审计。
- 新增代码/配置：`stage4_protocol.json`、`stage4_dataset_registry_v1.json`、`stage4_protocol.py`、`multigranular_semantics.py`、`spectrum_quality.py`及相应测试。
- 运行实验：仅接口与确定性单元测试，尚未运行模型效果实验。
- 主要结果：阶段4定向测试19项通过；全项目83项通过。
- 负结果与异常：现有数据无法构成未消费final holdout，注册表保持未物化。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate A 前置条件。
- 下周唯一优先事项：实现并验证resource-aware loss与Gate A统一比较接口。

### W02 执行更新

- 本周目标：建立Gate A数学核心并排除任务饱和。
- 完成内容：连续资源块、离散/soft regret、miss-risk、CVaR、Brier、bit约束、对偶更新、四方法接口、truth-only任务难度审计。
- 新增代码/配置：`resource_losses.py`、`resource_task_audit.py`、`audit_stage4_resource_task.py`、Gate A配置及测试。
- 运行实验：v1共54组，未找到合格任务；v2共90组，冻结选中8源/8块/需求4。
- 主要结果：v2选中任务validation gap 0.026813、margin 0.013327、ambiguous 0.2133；全项目95项测试通过。
- 负结果与异常：单源至4源任务普遍存在clean饱和或oracle并列，完整保留v1结果。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate A。
- 下周唯一优先事项：在冻结前端和统一训练控制下运行Gate A四损失设置。

### W03 执行更新

- 本周目标：完成Gate A四方法同条件训练并决定C1去留。
- 完成内容：四方法×五seed×160轮训练、离散量化、nominal数字链路bit、5×150层次配对bootstrap。
- 主要结果：resource方法平均regret 0.021895，detection-only为0.023624；配对regret与CVaR CI均低于0，bit无显著差异。
- 负结果与异常：rate方法降低bit但regret变差；full joint不稳定且Brier升至0.014034。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate A通过resource-only分支。
- 下周唯一优先事项：生成C2反事实边际价值/bit标签并完成truth访问审计。

### W04 执行更新

- 本周目标：完成C2反事实标签、因果特征审计和价值网络Gate B评价。
- 完成内容：四节点缓存、18000个train-only标签、五seed固定末轮训练、validation配对bootstrap及全节点G1预览公平bit修正。
- 主要结果：oracle空间显著，但学习网络平均regret/bit未稳定超过简单基线；CVaR改善仅作为后续风险调度线索。
- 负结果与异常：修复listwise mask NaN和漏计G1预览成本；所有无效中间结果已重跑覆盖。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate B当前神经分支未通过。
- 下周唯一优先事项：运行2-bit G1单变量可辨识性实验并决定网络或离散贪心路线。

### W05 执行更新

- 本周目标：检验提高G1位数能否改善C2可辨识性。
- 完成内容：q1/q2 train标签分布、calibration/validation预览任务性能、真实链路bit和10000次配对bootstrap。
- 主要结果：q2无显著regret收益，固定增加75.15 bit，Brier与升级oracle空间显著恶化。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate B预览诊断失败，按回退路线进入解析离散调度。
- 下周唯一优先事项：实现并冻结任务歧义—分歧—可靠性解析评分及预算求解器。

### W06 执行更新

- 本周目标：完成非神经解析评分与离散预算求解。
- 完成内容：train-only PAV、G0贪心、精确多选择背包、四预算曲线和配对bootstrap。
- 主要结果：求解器工作正常，但G1跨节点完全同质，评分恒0并退化为固定先验。
- 负结果与异常：修复恒分数伪秩相关；将链路期望bit迁回数字链路模块以解除PyTorch耦合。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate B解析分支未通过，进入最后一次预览可观测性改进。
- 下周唯一优先事项：实现并审计付费稀疏软残差调度草图。

### W07 执行更新

- 本周目标：用付费稀疏残差恢复C2候选可辨识性并检查系统级收益。
- 完成内容：113-bit codec、全节点/选择性缓存、解析PAV、四预算曲线、五seed网络复验和侧信息审计。
- 主要结果：候选差异恢复，局部解析G0策略有效；但固定开销与跨split稳定性未过门槛。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate B最终不通过，C2停止扩展。
- 下周唯一优先事项：冻结并运行C3可靠少数保护失效场景矩阵。

### W08 执行更新

- 本周目标：完成C3受控可靠少数与安全补传门控。
- 完成内容：七场景矩阵、六经典/审计方法、直接保护、最大通道冲突G3确认、真实反馈与包成本、bootstrap。
- 主要结果：特定故障得到修正，但直接保护未激活，确认补传误触发和尾部风险未过门槛。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：Gate C未通过，回退经典融合。
- 下周唯一优先事项：运行冻结系统的端到端链路和多维稳健性sweep。

### W09 执行更新

- 本周目标：完成实际解码路径多信道稳健性和开发证据冻结。
- 完成内容：150 scene、3信道、4 Eb/N0、5嵌套重复、三表示方案、实际bit/时延、scene配对bootstrap和综合结论表。
- 主要结果：选择性G2相对全G2在全部信道点显著省bit，regret差均未显著；delivery ratio不能替代任务指标。
- 是否触碰final holdout：否，尚未建立。
- 对应决策门：development系统冻结；final H2/H3仍未验证。
- 下周唯一优先事项：获取新增独立scene并物化一次性final holdout；在此之前不再调算法。

### W10 执行更新

- 本周目标：完成总计划逐项验收并固化 final 前防误用流程。
- 完成内容：质量校准/异常聚合诊断、价值 MAE/Spearman/top-k 诊断、机器验收器、final catalog 注册器、泄漏审计、单次访问状态机和冻结快照。
- 主要结果：development acceptance=true，failed=0；final acceptance=false，阻塞项仅为新增独立scene与一次性final未执行；全项目137项测试通过。
- 负结果与异常：价值排序接近随机，进一步支持Gate B停止；第一次全量测试遇到Windows双OpenMP运行时冲突，已将Spearman改为显式点积实现后全量通过，不改变实验数值定义。
- 是否触碰final holdout：否；`configs/stage4_final_access_state.json`仍为0。
- 对应决策门：开发验收闭环；等待F01/F02外部数据条件。
- 下周唯一优先事项：只接收并验证至少200个新增独立scene catalog；注册通过前不运行final、不改算法。

### W11 执行更新

- 本周目标：补齐总计划§13—15复杂度、基线、消融和实验矩阵证据。
- 完成内容：冻结C1/C2模型复杂度剖析；28项覆盖审计；将Gate B/C终止的分支显式登记为waived。
- 主要结果：C1 head 1740参数、1664线性MAC、离散推理中位0.4238 ms；覆盖审计19 passed、7 waived、2 final blocked、0 failed；全项目141项测试通过。
- 负结果与异常：不存在可诚实评价的“完整RAVES”，因为C2/C3已经失败；不以继续扫预算/SNR来制造完整方法结果。
- 是否触碰final holdout：否；访问计数0。
- 对应决策门：D4-003、Gate B、Gate C及开发验收。
- 下周唯一优先事项：等待并严格注册至少200个新增独立scene；否则保持冻结并只推进论文写作材料。

### W12 执行更新

- 本周目标：把冻结开发证据转化为可直接进入硕士论文的图表、章节和声明边界。
- 完成内容：paper asset生成器、4张图、2份CSV、汇总表、阶段4章节草稿、claim matrix、final结果模板和材料审计器。
- 主要结果：材料审计5/5通过，所有源/输出哈希与链接有效，final占位符完整，访问计数0；全项目145项测试通过。
- 负结果与异常：图表保留C2低排序召回和选择性G2未稳定通过非劣界等不利信息，不制作只展示有利结果的图。
- 是否触碰final holdout：否。
- 对应决策门：开发论文材料验收；正式H2/H3仍等待final。
- 下周唯一优先事项：新增独立scene到位后执行注册、冻结快照和一次性final；否则只允许润色论文，不再改变算法结论。

### W13 执行更新

- 本周目标：在final数据出现前冻结唯一统计输入和判决算法。
- 完成内容：scene级schema、输入/访问绑定校验、配对bootstrap、CVaR、单侧配对随机化、IUT family p值、Holm校正、正式CLI和合成分支自检。
- 主要结果：成功/失败/坏输入分支全部符合预注册预期；全项目148项测试通过。
- 负结果与异常：明确区分bootstrap CI和随机化p值，未使用置信尾概率冒充经典p值。
- 是否触碰final holdout：否，access count=0。
- 对应决策门：final统计预注册闭环。
- 下周唯一优先事项：只有新增独立scene注册成功后才消费访问并运行冻结推理/统计；否则保持外部数据阻塞。

### W14 执行更新

- 本周目标：核验近期相关工作并闭环论文定位与引用可追溯性。
- 完成内容：7篇原始来源核验、相关工作矩阵、BibTeX、机器注册表、章节定位段落、材料与总验收扩展。
- 主要结果：7/7参考完成核验，材料审计6/6通过，阶段4开发验收15 passed、1 waived、0 failed；全项目149项测试通过。
- 负结果与异常：arXiv:2501.15414最新版题名与旧计划不一致，已按v4纠正；多节点不等于MIMO，视觉/ARQ文献不作为C2/C3正结果替代。
- 是否触碰final holdout：否，access count=0。
- 对应决策门：D4-013及开发论文材料验收。
- 下周唯一优先事项：等待至少200个新增独立scene完成严格注册；在此之前保持算法、阈值、指标和声明冻结。

---

## 25. 近期参考方向

以下论文仅用于理解问题和设计对照，不复制其网络或结论：

1. P. Yi et al., *Integrated Distributed Semantic Communication and Over-the-Air Computation for Cooperative Spectrum Sensing*, IEEE TCOM 73(4):2416–2430, 2025, DOI:10.1109/TCOMM.2024.3468215.
2. X. Li et al., *Digital Semantic Device-Edge Co-Inference With Task-Oriented ARQ*, IEEE TVT 73(9):13986–13990, 2024, DOI:10.1109/TVT.2024.3390213.
3. Y. Hu et al., *Pragmatic Communication in Multi-Agent Collaborative Perception*, arXiv:2401.12694, 2024.
4. C. Cai et al., *End-to-End Learning for Task-Oriented Semantic Communications Over MIMO Channels: An Information-Theoretic Framework*, IEEE JSAC 43(4):1292–1307, 2025, DOI:10.1109/JSAC.2025.3531575.
5. B. Liu et al., *mmCooper: A Multi-agent Multi-stage Communication-efficient and Collaboration-robust Cooperative Perception Framework*, ICCV, 28396–28406, 2025.
6. W. Chen et al., *Entropy-and-Channel-Aware Adaptive-Rate Semantic Communication with MLLM-Aided Feature Compensation*, arXiv:2501.15414v4, 2026（v1提交于2025；旧题名不得继续引用）.
7. B. Li et al., *Toward Reliable Semantic Communication: Beyond Average Performance*, arXiv:2606.01284, 2026.

逐篇原始入口、适用边界和BibTeX见 `docs/STAGE4_RELATED_WORK_MATRIX.md` 与 `docs/references/stage4_verified_refs.bib`。本项目与上述工作的主要区别应始终保持为：宽带频谱资源选择任务、反事实 occupancy-regret 价值、实际数字包成本、节点—语义粒度联合调度和可靠少数保护；其中C2、C3仍是门控失败的机制研究，不能写成正性能贡献。

---

### D4-014：AERPAW固定节点数据按功率扫频而非原始I/Q接入

- 日期：2026-07-21
- 对应假设：H2及final外部效度，不复活H3。
- 发现的问题：Dryad官方说明和ZIP真实元数据均表明LW1载荷为`rf32_le`、98,868-bin的dBm功率扫频，不是复数I/Q；LW1为单接收站且deposit没有独立人工occupancy标签。
- 使用的数据范围：官方README、下载ZIP已持久化前段、合成AERPAW同尺寸扫频和既有development scene；未注册或访问final。
- 唯一主要变化：新增独立功率扫频适配器、永久排除pilot治理和下载完成后的一键校验/安全解压工具；C1/C2/C3、预算、阈值和统计规则不变。
- 成功规则：错误datatype、截断文件、缺失配对和pilot泄漏状态均被自动测试拒绝；端到端接口和完整软件链路可执行；final access count保持0。
- 实验结果：新增4项定向测试通过；合成AERPAW接口dry-run通过；development完整软件链路单线程CPU中位/p95为42.157/51.678 ms。
- 最终决定：LW1先作为永久排除pilot和有条件C1资源代理；在标签规则、域差异和单站点声明边界闭环前，不自动将剩余文件注册为final，且不能用于selective-G2四节点主检验。
- 是否更新论文贡献：不新增性能贡献；补充真实数据输入边界、外部效度风险和完整链路复杂度。

### W15 执行更新

- 本周目标：利用AERPAW下载等待时间完成输入适配、数据治理、dry-run和完整链路复杂度。
- 完成内容：`rf32_le`读取/校验、文件对发现、8频道映射、连续资源块、永久排除pilot manifest、下载后安全解压审计、合成全链路dry-run和development全链路CPU剖析。
- 主要结果：真实元数据确认87.16--6019.18 MHz与98,868 bins；完整软件链路中位/p95 42.157/51.678 ms；定向测试4项通过。
- 负结果与异常：AERPAW LW1不是原始I/Q、无独立人工occupancy标签、无四节点同scene观测，因此不能仅凭scene数量满足当前两个final主检验。
- 是否触碰final holdout：否；访问计数0。
- 对应决策门：AERPAW pilot适配门，不改变Gate A/B/C历史决定。
- 下周唯一优先事项：下载完成后先做整包SHA-256和20-scene永久排除pilot；只有兼容性审计通过才讨论候选final catalog。

### D4-015：AERPAW三固定站点采用条件性外部验证，不冒充原四节点主条件

- 日期：2026-07-21
- 对应假设：H2的真实功率资源代理外部效度，以及选择性G2的三站点迁移性；不复活H3。
- 发现的问题：CC1、CC2、LW1共三个固定站点，尚需在完整ZIP到位后证明时间重叠；AERPAW没有独立occupancy标签和上报链路测量。F盘完整LW1解压已占约55.4 GB，CC1/CC2不能继续完整解压。
- 使用的数据范围：20个永久排除LW1 pilot、官方Dryad元数据和SHA-256、合成三站点功率扫频；没有读取非pilot方法输出，没有创建或访问final。
- 唯一主要变化：将资源代理冻结为2.4 GHz ISM频段的8个等宽频道及per-sweep median/MAD门限；增加三站点10秒对齐、30分钟稀疏化、200 scene等距选择、ZIP精简抽取及三节点链路偏置六排列。C1/C2/C3 checkpoint、损失、预算和final统计器不变。
- 成功规则：三个官方ZIP哈希通过，至少200个三站点scene满足对齐和独立性门槛，pilot overlap为0，schema/频率轴/功率范围/代理映射全部通过；final access count保持0。
- 实验结果：20个pilot的2.4 GHz子带均含1391 bins，鲁棒门限中位约-132.809 dBm，总体频道占用中位约0.03169；三节点合成dry-run中两种C1均接受3×8输入，选择性G2/all-G2完成六种链路偏置排列。所有数值均为映射或接口证据，不是性能结果。
- 失败分支：若不足200个同步且稀疏化后的三站点scene，禁止异步拼接；final降级为LW1单站点C1条件性代理，选择性G2只保留development证据。
- 最终决定：等待CC1/CC2时不再调算法；只推进数据治理、接口与论文边界。三站点结果不得写成四节点、MIMO、真实无人机协作或实测上报链路。
- 是否更新论文贡献：只增加真实数据外部验证设计与可复现数据治理，不提前增加正性能贡献。

### W16 执行更新

- 本周目标：在CC1/CC2下载期间完成真实三站点final的全部非结果型前置工作。
- 完成内容：LW1 pilot代理冻结、官方三包大小/SHA-256登记、ZIP中央目录配对发现、时区转换、最近未复用对齐、30分钟稀疏化、200 scene等距选择、精简抽取、三站点schema/频率/功率质控、代理label/catalog构建工具及三节点合成链路演练。
- 主要结果：pilot映射未退化；三节点C1和选择性G2代码路径可执行；AERPAW定向16项、全项目165项测试通过，development acceptance=true、failed=0；完整CC1/CC2到位前没有生成候选catalog或正式方法输出。
- 负结果与异常：磁盘剩余空间不足以完整解压CC1/CC2，已改为只抽取入选scene；官方说明没有单独承诺三站点逐扫频同步，因此把时间重叠保留为M1硬门槛。
- 是否触碰final holdout：否；访问计数0。
- 对应决策门：D4-014、D4-015及final catalog注册前门槛。
- 下一步唯一优先事项：下载完成后运行官方哈希、三站点时间对齐和200 scene精简审计；M1结论出来前不运行模型、不生成性能结果。
### W17：C1-v3 尾部排序优化与次级时域确认

- 冻结了未复用原三站 Final 的 60 景 LW1 校准集、270 景后时段 LW1 主确认层和 30 景后时段 CC1 次确认层；300景共32个站点×小时统计块。
- 在固定4-bit、配对 warm-start 和既有 C1-v2 设置上，只增加逐场景 block-ranking CVaR；9组预注册候选完整披露，校准选择 `tail2_margin0.05`。
- 校准平均功率 regret 改善约1.81%，CVaR不变；一次性确认中平均差 +0.004977 dB、CVaR差 +0.005472 dB，两个优势门槛均失败；应用层和实际发送bit完全相同。
- 新确认访问计数为1，禁止在该300景上继续调参或重跑。C1-v3保留为严谨负结果，不新增正向论文贡献。
- 下一优先事项改为 C1-v4：以真实功率差直接监督等码率 block-score/ranking 语义，并在另一数据集或新采集批次上确认。详见 `docs/STAGE4_C1_V3_TEMPORAL_CONFIRMATION.md`。

### W18：C1-v4 任务充分连续块语义开发

- 冻结5类表示：8维 occupancy、8维信道功率、5维4-bit块代价、5维6-bit块代价和5维1-bit最优块指示；统一使用152-bit固定协议开销并进入真实CRC/FEC/ARQ链路。
- 开发集协议按整日拆分为270/180/180景，先排除全部既有 pilot、校准和Final来源；验证统计按站点×小时分组bootstrap。
- 已完成真实ZIP读取、特征/目标构造、版本化编解码、链路评估器、候选准入规则和181项全项目测试。
- F盘恢复后完成630景真实缓存。验证集上 `block_score_fixed4` 相对 occupancy 将平均 regret/CVaR/实际bit分别降低57.42%/66.33%/2.80%；`best_block_indicator1` 分别降低57.98%/66.77%/8.43%，按预注册规则入选。
- clean block-score regret已接近0，因此取消无必要的学习型校正器；C1-v4仍是开发方案，未来独立确认要求不变。详见 `docs/STAGE4_C1_V4_DEVELOPMENT.md` 和 `docs/STAGE4_C1_V4_RESULTS.md`。

### W19：C1-v4 零比特最近成功块回退

- 在查看2月12日数据前冻结同站点最近成功块回退；报文、信道轨迹和实际发送bit与固定block-0回退完全相同。
- 新日期180景、72个站点×小时块上，平均 regret 从0.173469降至0.124060，CVaR从0.669129降至0.418094；单侧95%分组上界分别为−0.029334和−0.175251，bit差为0，三项门槛全部通过。
- 三站点方向一致。该结果仍是第二轮开发证据，不是Final；下一步需冻结陈旧时间保护并寻找另一数据集/新采集批次确认。

### W20：C1-v4 同活动跨日期单次Final

- 在读取测量值前冻结60分钟最大陈旧时间、280景双站点清单、74个站点×小时统计簇、两级固定顺序假设、20,000次分组bootstrap及67文件执行快照；与全部既有pilot、旧Final、C1-v3确认和V4开发来源重叠为0。
- H1中最优块指示相对occupancy将平均regret、CVaR和实际发送bit分别降低57.26%、60.68%和8.50%；三项单侧95%上界均满足优势门槛。
- H2中60分钟时效保护回退在相同157-bit报文与完全相同链路轨迹下，将平均regret和CVaR进一步降低28.07%和31.94%，实际bit差严格为0；固定顺序确认通过。
- CC1、CC2对两级改进均方向一致，`overall_final_success=true`；新访问状态永久为`access_count=1`，禁止重跑或基于该结果修改阈值。
- 阶段4主正向证据现包括：原Final中选择性G2通过，以及本次C1-v4两级Final通过。剩余工作主要是论文图表、章节整合和声明边界审计；新数据集属于增强外部效度，不是重做本Final。
- 证据边界：同一AERPAW活动的固定站点跨日期确认，不是跨数据集、无人机移动、MIMO、原始I/Q或实测上报链路。详见`docs/STAGE4_C1_V4_FINAL_RESULTS.md`。

### W21：Final后补充经典基线

- 冻结并运行补充协议，只读取2月11日和2月12日已经看过的开发缓存；程序明确拒绝C1-v4 Final路径，结果记录未加载Final测量值及Final指标。
- 表示侧补充硬能量1-bit、occupancy 4-bit、软功率4/8-bit、静态块、均匀随机和不可实现oracle。最优块指示相对4-bit软功率仅小幅改善平均regret/CVaR，但以低8.46%的实际bit完成任务，明确了“任务充分压缩”而非单纯精度领先的贡献。
- 可靠性侧补充无ARQ、标准/额外ARQ、无限保持和60分钟保护。60分钟保护相对标准ARQ无状态降低平均及尾部regret且bit相同；相对额外ARQ尾部更好且少18.00% bit。
- 无限保持在固定站点慢变化开发数据上优于60分钟保护。该负结果完整保留，说明年龄上限是防陈旧状态的保守安全约束，不能宣称为当前数据上的全局最优参数。
- 补充结果为Final后的描述性机制解释，不参与算法选择、不改变单次Final主声明。详见`docs/STAGE4_C1_V4_SUPPLEMENTARY_BASELINES.md`。

### W22：C1-v4规模可扩展性实验

- 在不读取Final的条件下，对2月11日180个已见开发扫频重新划分N = {8，16，32，64}信道，并测试D/N = {0.25，0.50，0.75}，形成12个固定连续块任务。
- 比较4-bit逐信道软功率、当前最优块独热指示和Final后开发的紧凑二进制块编号；三者经过相同152 bit协议开销、分包、CRC、FEC、ARQ和三类信道网格。
- 当前独热指示在全部N=32和N=64任务中达到“实际bit至少降低20%且平均regret单侧95%上界非劣”，共6/12项；N=64的实际bit降幅为39.98%—46.26%。
- N=64、D=32时，软功率/独热/二进制编号实际bit为1362.81/775.29/689.78，平均regret为0.286245/0.250619/0.244000 dB。AWGN 3 dB过渡区结果支持短报文减缓信道恶化。
- 二进制编号属于新的开发性协议优化，未经过独立确认，不得并入已完成的独热Final主声明。详见`docs/STAGE4_C1_V4_SCALABILITY_EXPERIMENT.md`。
