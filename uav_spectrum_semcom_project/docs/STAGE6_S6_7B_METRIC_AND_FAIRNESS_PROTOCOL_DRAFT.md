# 阶段6 S6.7b指标字典与公平可靠性实验协议

版本：1.0  
日期：2026-07-27  
状态：指标字典与开发协议已冻结；仪表层、执行层和完整开发粗网格均已完成，等待2022三站点分组验证  
数据边界：只允许2023 pilot和2022固定站点开发数据；禁止访问2024—2025外部Final信号值

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：experiment design
- Verification Status：PROTOCOL FROZEN；INSTRUMENTATION AND EXECUTION SMOKE VERIFIED
- Upstream Evidence：S6-FC0、S6.6b组件消融、S6.7a治理审计
- Primary Question：相同clean约束下，任务语义系统是否比自包含精确动作基线节省总通信资源
- Protocol Config：`configs/stage6_matched_reliability_development_v1.json`
- Protocol SHA-256：`cb3d5fde27edeefd39b6dcaf94db67d565997b89cda7e3be3946efa7e4515636`
- Preflight Result：`results/stage6/matched_reliability_preflight_v1/preflight.json`
- Preflight SHA-256：`eac6ffc37b72d99614c300a81f03ddc0833879e1b36dde09ef439ebb56adb6d1`
- Instrumentation Dry-run：`results/stage6/matched_reliability_instrumentation_dry_run_v1/result.json`
- Dry-run SHA-256：`2bb9caf207ba7135d00de477cbaf72139c26308a2d68a7ccf8b5ad0e2aa753a6`
- Execution Smoke：`results/stage6/matched_reliability_execution_smoke_v1/result.json`
- Smoke SHA-256：`deb4c0ee933498a8ee3344eb6fa2264463ffef3f24be06c226cb18d5a3b51411`
- Coarse Grid：`results/stage6/matched_reliability_coarse_grid_v1/result.json`
- Coarse Grid SHA-256：`0da668fe5a2c044132417d936ff0db8da7951e4be2b431933ecc7a688b6dba5f`
- Coarse Analysis：`results/stage6/matched_reliability_coarse_analysis_v1/analysis.json`
- Coarse Analysis SHA-256：`dabee7610a40560f8ed98912345e74ff31ebb3a06d49c58c426d74c708585d4f`

---

## 1. 实验假设

### 1.1 主要假设

在相同冻结任务、相同频谱场景、相同链路随机轨迹和相同目标clean rate下，S6-FC0任务语义系统的总应用层bit低于自包含精确动作基线。

对冻结目标clean c*，定义：

`Savings(c*) = 100 × [Bexact(c*) − Bsemantic(c*)] / Bexact(c*)`

其中：

- Bsemantic(c*)：语义系统达到c*所需的最小公平总bit；
- Bexact(c*)：精确动作基线达到c*所需的最小公平总bit；
- Savings(c*) > 0表示语义系统节省bit。

### 1.2 次要假设

- 语义系统在共同可达clean区间内形成不劣的Pareto前沿；
- 优势在N = 8、16、32、64和多个D/N下具有一致趋势；
- 优势不是通过大量availability损失、过期动作或未计入反馈开销获得；
- 结论在ε变化时不只依赖0.2 dB单点。

---

## 2. 评测单位

### 2.1 Scene

一个scene表示一个完整频谱扫频时刻及其N维候选信道功率向量。不同N聚合来自同一原始scene，必须共享同一scene id和数据分组。

### 2.2 Session

一个session表示接收端上下文从规定初始状态开始的一段连续scene序列。

部署模式分为：

1. **预配置码本**：会话开始时双方已具有相同码本；运行期重置或失配造成的重新安装仍计费；
2. **在线安装码本**：会话开始为空上下文，初次完整安装也计费。

两种模式分别报告，不能把预配置收益与在线安装成本混为一项。

### 2.3 会话长度

若连续数据足够，固定报告：

- 短会话：20 scenes；
- 中会话：100 scenes；
- 长会话：500 scenes。

若某一时间段不足500 scenes，则长会话使用该连续段全部scene，并明确实际长度。任何session不得跨越数据断点。

数据断点：

`相邻时间间隔 > max(3 × 站点中位间隔，120秒)`

---

## 3. 任务指标字典

### 3.1 Regret

对当前频谱状态x、查询q和接收端动作a：

`R(x,q,a) = c(x,q,a) − min[a′] c(x,q,a′)`

多查询scene regret：

`Rmax(x,a) = max[q] R(x,q,a_q)`

单位为dB。

### 3.2 Availability

若接收端当前具有可执行的动作元组，则：

`available = 1`

否则：

`available = 0`

Availability：

`Availability = 可执行scene数 / 全部scene数`

被fail-closed拒绝的错误上下文计为不可用，不能从分母删除。

### 3.3 Clean

`clean = 1`当且仅当：

- `available = 1`；
- `Rmax ≤ ε`。

Clean rate：

`Clean rate = clean scene数 / 全部scene数`

因此不可用scene必然不clean。

### 3.4 Task failure probability

`Task failure probability = 1 − Clean rate`

该指标同时包含：

- 接收端不可用；
- 可用但regret超过ε。

必须额外分解两种来源，避免无法判断是通信失败还是动作质量失败。

### 3.5 Conditional regret

只在available scene上计算：

- conditional mean regret；
- conditional maximum regret；
- conditional CVaR₀.₉。

Conditional regret用于回答“接收端有动作时动作质量如何”，不能替代包含掉线的系统指标。

### 3.6 Effective regret

对不可用scene赋予冻结掉线惩罚Pout：

`Reffective = Rmax`，当available = 1；

`Reffective = Pout`，当available = 0。

第一版保持：

`Pout = 10 dB`

报告：

- effective mean regret；
- effective CVaR₀.₉；
- CVaR中掉线scene所占比例。

如果CVaR被10 dB完全封顶，必须说明尾部风险由掉线主导，不能把CVaR不变解释为动作风险不变。

### 3.7 Escape

若没有任何冻结码字满足当前多查询ε约束，发送escape和精确动作。

`Escape rate = escape scene数 / 需要任务更新的scene数`

同时报告：

- 每scene escape率；
- 每次更新escape率；
- escape增加的实际bit；
- 外部新动作元组比例。

---

## 4. 通信开销字典

### 4.1 Payload bits

仅表示任务内容本身，不含通用头部、身份字段和可靠性信息。

用途：解释表示压缩能力。

限制：不能作为系统级主要结论。

### 4.2 Forward application bits

发送端实际尝试发出的应用层bit总数，包括：

- 任务码字或精确动作帧；
- 完整码本/上下文安装；
- 紧凑更新；
- escape精确动作；
- 重复更新；
- 心跳请求；
- 主动恢复请求。

丢失的发送尝试仍计bit。

### 4.3 Feedback application bits

接收端实际尝试发出的应用层bit总数，包括：

- 累计ACK；
- 安装ACK；
- 心跳响应；
- 恢复响应；
- 延迟重复ACK。

当前S6-FC0结果字段`ack_application_bits`实际同时包含ACK和心跳响应。S6.7b新结果中必须改名为`feedback_application_bits`，并细分：

- `ack_bits`；
- `heartbeat_response_bits`；
- `other_feedback_bits`。

不得修改S6-FC0历史结果，只在新版本结果中消除命名歧义。

### 4.4 Total actual application bits

主要通信开销：

`Bactual = Bforward + Bfeedback`

按scene报告：

`Bactual/scene = Bactual / scene数`

当前S6.6b的`mean_application_bits_per_scene`对应这一口径。

### 4.5 Reserved capacity

预测保护会预留重复发送容量。当前历史实现将其记录为`reserved_capacity_bits`，但没有全部加入`total_application_bits`。

S6.7b必须新增：

- 已预留容量；
- 实际用于重复的容量；
- 未使用但被占用的容量；
- 被保护更新的真实重复bit。

如果预留资源会阻塞其他业务，则资源等效开销定义为：

`Bresource = Bactual + Bunused_reserved`

其中：

`Bunused_reserved = Breserved − Bused_reserved`

主要表格同时报告Bactual和Bresource。若二者导出的结论不同，以更保守的Bresource作为可靠性预算结论。

### 4.6 安装与恢复开销

分别统计：

- 初始完整安装bit；
- 重置后的重装bit；
- 码本身份失配恢复bit；
- 紧凑更新bit；
- 心跳请求与响应bit；
- ACK bit；
- 重复与ARQ bit。

不能把所有开销只合并成一个total而不提供分解。

### 4.7 Nominal transmitted bits

在统一数字链路编码、CRC、纠错、调制和有限ARQ后，可计算名义链路bit。

S6.7b第一主要指标仍为应用层实际bit，因为当前故障通过受控包丢失注入。名义链路bit作为第二层指标，只有在两种方法使用完全相同的链路编码和分包规则时才比较。

不得把应用层bit称为真实空口bit。

---

## 5. 方法与基线

### 5.1 语义系统

固定S6-FC0核心组件：

- 解析型ε任务等价码本；
- escape精确动作；
- 版本化任务与上下文codec；
- hard regret和状态年龄触发；
- 累计ACK信念恢复；
- 固定心跳候选；
- 冻结一步风险排序；
- 有预算的紧凑更新重复保护；
- fail-closed身份校验。

S6.7b只扫描工作点参数，不修改任务码本算法和动作定义。

### 5.2 精确动作基线

每scene发送多查询精确动作索引，帧自包含，不依赖码本、ACK或历史上下文。

精确基线必须允许自身可靠性工作点：

- 1次发送；
- 2次独立重复；
- 3次独立重复；
- 若实现有限ARQ，则使用与语义帧相同的确认和最大尝试规则。

不能只让语义系统调可靠性而固定精确基线，也不能让精确基线使用额外理想反馈。

### 5.3 辅助基线

- ideal feedback上界；
- 无状态语义包；
- S6-FC0去心跳；
- S6-FC0去预测保护；
- 每scene完整精确包。

occupancy和软功率基线保留用于表示层比较，但S6.7b主要公平系统比较为“有状态任务语义”与“自包含精确动作”。

---

## 6. 开发参数扫描

### 6.1 第一层：工作点曲线

在冻结混合故障条件下扫描：

| 参数 | 候选值 |
|---|---|
| 心跳间隔H | 10、20、40、80、关闭 |
| 最大状态年龄 | 30、60、120分钟 |
| 更新保护预算比例 | 0、0.05、0.10、0.20 |
| 任务帧最大尝试数 | 1、2、3 |
| ACK策略 | 累计ACK、关闭ACK的可定义对照 |
| 部署模式 | 预配置、在线安装 |

禁止直接运行全部笛卡尔积而不检查重复或不可定义组合。协议冻结前先生成配置数量、预计运行时间和存储预算。

### 6.2 第二层：故障稳健性

从第一层每种方法的Pareto前沿选择少量冻结工作点，再扫描：

| 故障参数 | 候选值 |
|---|---|
| task packet loss | 0、0.05、0.10、0.20 |
| ACK loss | 0、0.10、0.20、0.30 |
| install loss | 0、0.05、0.10、0.20 |
| delayed duplicate ACK | 0、0.05、0.10、0.20 |
| receiver reset | 0、0.005、0.01、0.02、0.05 |

第二层不再重新选择最优参数，只检验第一层冻结工作点的退化速度。

### 6.3 任务规模

- N ∈ {8, 16, 32, 64}；
- D/N ∈ {0.25, 0.50, 0.75}；
- ε第一轮固定0.2 dB；
- ε完整网格由S6.7c单独执行。

---

## 7. Pareto与matched-clean规则

### 7.1 Pareto前沿

若工作点A同时满足：

- `Bactual(A) ≤ Bactual(B)`；
- `Clean(A) ≥ Clean(B)`；
- 至少一项严格更优；

则B被A支配，不进入Pareto前沿。

Bresource重复构造一条更保守的前沿。

### 7.2 共同clean区间

只在两种方法开发曲线都能达到的clean区间内做matched-clean比较。

90%、95%、97%只是候选目标，不是强制目标。若其中某点不在共同区间，不得外推或人为增加不公平保护。

### 7.3 冻结工作点

在开发数据上，对目标c*选择：

> clean的95%置信区间下界达到c*的配置中，总bit最小的配置。

若没有配置达到该标准，该目标标记为不可达。

Final阶段直接运行这些冻结配置，不得根据Final clean重新匹配或重新选择参数。

### 7.4 插值

线性插值只用于绘图和描述完整曲线，不作为Final主要显著性检验。主要检验使用真实冻结配置。

---

## 8. 随机性与统计方法

### 8.1 共同随机数

所有方法在相同scene和轨迹编号下共享：

- task loss随机数；
- install loss随机数；
- ACK loss随机数；
- delayed ACK随机数；
- receiver reset随机数；
- 重复发送随机数。

不同方法没有定义的事件可以忽略对应随机流，但不得改变其他事件的随机顺序。

### 8.2 开发重复

第一版保持：

- 每个工作点60条链路轨迹；
- 固定主种子；
- 5,000次配对bootstrap；
- 95%置信区间。

运行前进行时间预算估计。如果配置数量过多，应先删去重复或理论支配配置，不得根据结果删配置。

### 8.3 数据不确定性

链路轨迹置信区间不能替代真实场景不确定性。

2022固定站点使用“站点×日期”作为外层分组，分别报告：

- 按链路轨迹的配对区间；
- 按站点×日期的cluster bootstrap区间；
- leave-one-site-out趋势。

### 8.4 多重比较

主要结论只指定一个主要工作点和一个主要指标：

`matched-clean下的Bresource节省率`

其他N、ε、故障强度和指标为次要或探索性。若对多个目标clean同时做显著性声明，应使用Holm校正或同时置信区间。

---

## 9. 成功与止损门槛

### 9.1 主要成功

在至少一个预注册共同clean工作点：

- Savings点估计 > 0；
- 95%配对区间下界 > 0；
- availability差异没有被隐藏；
- Bactual和Bresource结论方向一致。

点估计达到15%—20%以上时，可形成较强论文结论。

### 9.2 部分成功

若只在长会话或N ≥ 32成立，应明确收缩结论为：

> 任务等价语义适合码本安装可摊销、候选信道规模较大的长会话频谱控制。

### 9.3 失败与转向

出现以下任一情况，不进入外部Final：

- 没有共同clean区间；
- matched-clean节省率不为正；
- Bactual有优势但Bresource无优势；
- 语义系统主要通过不可用换取低bit；
- 结果只在单个随机种子或单个站点成立；
- 协议头部、重装或心跳抵消码本收益。

此时转入：

- 头部和身份信息摊销分析；
- 周期性自包含帧；
- 半状态语义；
- 更低成本上下文摘要；
- 长会话适用边界研究。

---

## 10. 需要实现的新结果字段

S6-FC0历史文件保持不变。S6.7b新模拟器或仪表层必须新增：

- `task_frame_bits`
- `exact_action_frame_bits`
- `escape_exact_bits`
- `initial_install_bits`
- `recovery_install_bits`
- `compact_update_bits`
- `duplicate_update_bits`
- `heartbeat_request_bits`
- `ack_bits`
- `heartbeat_response_bits`
- `other_feedback_bits`
- `actual_forward_bits`
- `actual_feedback_bits`
- `actual_total_application_bits`
- `reserved_capacity_bits`
- `used_reserved_capacity_bits`
- `unused_reserved_capacity_bits`
- `resource_equivalent_bits`
- `nominal_link_bits`，若已实现统一数字链路映射

并检查：

`actual_total_application_bits = actual_forward_bits + actual_feedback_bits`

`resource_equivalent_bits = actual_total_application_bits + unused_reserved_capacity_bits`

---

## 11. 仪表层dry-run结果与进入性能网格前的任务

预注册配置已经确定：

- 语义系统固定使用24-bit累计ACK；
- 精确基线使用1、2、3次自包含开环发送，不使用理想反馈；
- Final工作点只能由开发曲线冻结，禁止Final后重新匹配；
- 主要开销同时使用Bactual和更保守的Bresource。

2026-07-27已在永久排除的2023 pilot上完成小规模仪表层演练：

- N = 8，D = {2, 4, 6}；
- 20 scenes × 3条混合故障轨迹；
- 全部前向、反馈、总开销、预留容量和资源等效bit恒等式通过；
- 独立心跳探针演练实际产生1次32-bit请求和1次32-bit响应，反馈分解余项为0；
- 三条混合故障轨迹的availability均为100%，clean分别为95%、100%和95%；
- 演练只用于确认记账正确性，不构成算法性能结论；
- 外部Final信号值未读取，阶段6外部Final访问次数仍为0。

正式运行粗网格前仍需完成：

1. 将冻结的一步风险排序器接入预留比例工作点；
2. 完整粗网格运行器支持断点、分批和结果合并；
3. 建立2022固定站点流式紧凑缓存，禁止全量解压占满F盘；
4. 完成粗网格后只保留每个N至多5个冻结Pareto候选。

上述实现只能写入新文件，不得修改S6-FC0锁定文件。

## 12. 执行层smoke结果

执行层smoke覆盖：

- 预配置码本但动作状态为空；
- 在线安装且上下文为空；
- 1、2、3次嵌套开环任务发送；
- H = 10和关闭心跳；
- 自包含精确动作基线的1、2、3次发送；
- 11条固定用途共同随机流。

共运行17个工作点、5条轨迹和20 scenes，即1,700次scene评估。所有执行检查通过；新增代码的定向测试全部通过。

单进程实测吞吐量约为11,771次scene评估/秒。计入20、100和500 scenes三种会话长度后，预注册完整粗网格上界为95,135,040次scene评估；线性估算单进程约需8,082秒，即约2.25小时。该估计只用于安排运行批次，不能作为算法性能证据。

## 13. 完整粗网格执行状态

四个N分片均已完成：

- 每个N包含1,080个语义工作点和9个精确动作工作点；
- 每个工作点包含60条配对链路轨迹；
- 四个分片实测耗时约10.7—13.7分钟；
- 合并结果约312 MB；
- 333项项目回归测试全部通过；
- 95%配对bootstrap分析已完成；
- 2022分组验证候选已按控制器参数去重冻结。

主要结果和负结果见`docs/STAGE6_S6_7B_MATCHED_RELIABILITY_RESULTS.md`。后续不得返回粗网格重新选择参数。
