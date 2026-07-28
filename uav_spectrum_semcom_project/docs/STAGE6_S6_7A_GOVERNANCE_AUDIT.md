# 阶段6 S6.7a冻结证据与数据治理审计

版本：1.0  
日期：2026-07-27  
状态：S6.7a全部治理门槛通过  
Final状态：阶段6外部Final访问次数为0

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：execution planning + reproducibility audit
- Verification Status：ANALYZED AND HASH-VERIFIED
- Primary Result：`results/stage6/s6_7a_governance_audit_v1/governance_metadata_audit.json`
- Result SHA-256：`369b3f1822ab9e4b3995b0fb64ed1e2eca676379c8a9c190ce408ec20c2b5f8c`
- Audit Script：`scripts/audit_stage6_s6_7a_metadata.py`
- S6.7b Preflight：`results/stage6/matched_reliability_preflight_v1/preflight.json`
- S6.7b Preflight SHA-256：`eac6ffc37b72d99614c300a81f03ddc0833879e1b36dde09ef439ebb56adb6d1`
- S6.7b Instrumentation Dry-run：`results/stage6/matched_reliability_instrumentation_dry_run_v1/result.json`
- S6.7b Dry-run SHA-256：`2bb9caf207ba7135d00de477cbaf72139c26308a2d68a7ccf8b5ad0e2aa753a6`
- S6.7b Execution Smoke：`results/stage6/matched_reliability_execution_smoke_v1/result.json`
- S6.7b Smoke SHA-256：`deb4c0ee933498a8ee3344eb6fa2264463ffef3f24be06c226cb18d5a3b51411`
- Claim Boundary：只核验冻结文件、开发数据目录元数据、抽样频率轴和外部Final压缩包完整性；没有读取外部Final功率值，没有执行阶段6 Final

---

## 1. 审计目的

S6.7a用于在matched-reliability公平实验和外部Final之前建立不可歧义的证据边界：

1. S6-FC0是否仍与冻结时完全一致；
2. 开发数据和外部Final数据是否具有唯一角色；
3. CC1、CC2、LW1是否具备扩充时序研究的实际容量；
4. 外部Final压缩包是否完整且仍保持零信号访问；
5. 后续指标和通信开销是否可以统一复算。

本轮没有打开2024—2025外部Final压缩包的成员目录或SigMF内容。外部Final只进行了文件存在性、文件大小和整个压缩包SHA-256核验。

---

## 2. S6-FC0冻结证据核验

### 2.1 冻结身份

| 项目 | 结果 |
|---|---|
| freeze id | `stage6_candidate_architecture_freeze_v1` |
| 冻结清单 | `configs/stage6_candidate_architecture_freeze_v1.json` |
| 冻结清单SHA-256 | `8409170d391af04d04c1ba0e16f937ab01749d8248d63b20dc25be43634da3dd` |
| 锁定代码/配置文件 | 10项 |
| 锁定结果证据 | 12项 |
| 实际存在 | 22/22 |
| 实际哈希匹配 | 22/22 |
| 判定 | PASS |

### 2.2 锁定代码和配置

以下文件均重新计算SHA-256并与冻结清单一致：

1. `src/spectrum_semcom/stage6_task_codebook.py`
2. `src/spectrum_semcom/stage6_task_codec.py`
3. `src/spectrum_semcom/stage6_context_codec.py`
4. `src/spectrum_semcom/stage6_context_recovery.py`
5. `src/spectrum_semcom/stage6_context_heartbeat.py`
6. `src/spectrum_semcom/stage6_temporal_hazard.py`
7. `src/spectrum_semcom/stage5_cumulative_ack.py`
8. `scripts/run_stage6_context_recovery_development.py`
9. `scripts/run_stage6_joint_predictive_recovery_development.py`
10. `configs/stage6_joint_predictive_recovery_development_v1.json`

### 2.3 锁定结果证据

以下12项结果也全部与冻结清单一致：

1. 任务码本；
2. 真实任务codec；
3. 上下文事件；
4. 上下文恢复；
5. 固定心跳；
6. 一步风险预测；
7. 预测重复保护；
8. 联合预测恢复；
9. 直接下一码字分类；
10. 两阶段选择性码字分类；
11. 马尔可夫概率调度；
12. 马尔可夫token bucket终止结果。

### 2.4 开发验收

| 项目 | 结果 |
|---|---:|
| 开发验收门槛 | 全部通过 |
| 回归测试数 | 323 |
| pytest退出码 | 0 |
| 主验收结果SHA-256 | `48bdebfee9862807be9a980fcc21e8c09800e4559b20e42dc657d2d5ec3e84e9` |
| 独立复现结果SHA-256 | `48bdebfee9862807be9a980fcc21e8c09800e4559b20e42dc657d2d5ec3e84e9` |
| 主结果与复现 | 字节级一致 |
| 阶段6 Final访问次数 | 0 |

当前授权仍为：

- 可以继续完成S6.7a—S6.7d开发治理；
- 不授权执行外部Final；
- 不允许覆盖或修改S6-FC0；
- 新实现必须使用新版本号。

---

## 3. 数据使用账本

### 3.1 当前数据角色

| 数据 | 本地位置 | 已知访问状态 | 后续唯一角色 |
|---|---|---|---|
| Packapalooza 2023 helikite | `F:/uav_spectrum_final_data/aerpaw_helikite_external_v1/pilot/Packapalooza_2023` | 功率值已用于阶段5—6开发 | 永久开发、适配和消融 |
| February 2022 LW1 | `F:/uav_spectrum_final_data/aerpaw_sub6_feb2022/raw/ResultsLW1Feb2022_SigMF.zip` | 已访问并参与阶段4开发/历史Final体系 | 阶段6开发与分组验证 |
| February 2022 CC1 | `F:/uav_spectrum_final_data/aerpaw_sub6_feb2022/raw/ResultsCC1Feb2022_SigMF.zip` | 已访问；2月14—15日参与C1-v4 Final | 阶段6开发与分组验证 |
| February 2022 CC2 | `F:/uav_spectrum_final_data/aerpaw_sub6_feb2022/raw/ResultsCC2Feb2022_SigMF.zip` | 已访问；2月14—15日参与C1-v4 Final | 阶段6开发与分组验证 |
| Packapalooza 2024 helikite | `final/Packapalooza_2024/Packapalooza_2024_SigMF.zip` | 只核验文件大小和哈希 | 阶段6外部Final候选 |
| Lake Wheeler 2024 helikite | `final/Lake_Wheeler_2024/Lake_Wheeler_2024_SigMF.zip` | 只核验文件大小和哈希 | 阶段6外部Final候选 |
| Packapalooza 2025 helikite | `final/Packapalooza_2025/Dryad_submission_AERPAW_Pack2025_17Oct2025.zip` | 只核验文件大小和哈希 | 阶段6外部Final候选 |
| 未来合成数据 | 尚未生成 | 无 | 训练、稀有事件增强和压力测试 |

February 2022三站数据已经被工程访问，不能重新宣称为阶段6独立外部Final。即使其中部分时段此前没有进入某次实验，也只能作为隔离开发验证或准外部分析。

### 3.2 外部Final压缩包完整性

| 活动 | 字节数 | SHA-256 | 结果 |
|---|---:|---|---|
| Packapalooza 2024 | 745,874,507 | `d6367101b05a8161880029274d619e8e823637e92e2c47751102f9be0f987617` | MATCH |
| Lake Wheeler 2024 | 787,189,713 | `bee764cee4c99f7cddc35c9036ef5c38e6283516ebfe322f34d50297c3bbfa9b` | MATCH |
| Packapalooza 2025 | 438,557,314 | `30490681c9f791a8d66ea18ad40ce4ad02d0d30322ee6bfb774b33153ea012c9` | MATCH |

对应Dryad外层下载包也全部与既有登记哈希一致：

- Packapalooza 2024：`411613dc421a5d7e5b94321ec1c5a840e11d20657cb442f597cb379482bd4b21`
- Lake Wheeler 2024：`826df7e25567797536af11e5c94822cca3eb7951de1ce286721cd6fdfa600b36`
- Packapalooza 2025：`7f9db2a4a0ea097e24ddc524309acb0e0f35db912c5a5cb552e8287095fe7f47`

外部Final登记状态仍为：

`download_bundles_verified_inner_archives_ready_final_not_accessed`

### 3.3 存储状态

审计时F盘：

- 已使用约791.82 GB；
- 剩余约139.67 GB；
- `aerpaw_sub6_feb2022`约106.50 GB；
- `aerpaw_helikite_external_v1`约2.18 GB。

当前空间足以完成元数据索引和紧凑开发缓存，但不应无计划地把CC1和CC2全部解压，因为三站完整解压量约228 GB，超过当前剩余空间。后续应采用ZIP流式读取或只建立任务所需频带的紧凑缓存。

---

## 4. CC1、CC2、LW1开发数据元数据审计

### 4.1 场景容量

ZIP中央目录排除了`__MACOSX`镜像条目，并按`.sigmf-meta`和`.sigmf-data`同名配对：

| 站点 | 配对场景 | 日历日组 | 首个时间戳 | 末个时间戳 | 中位间隔 |
|---|---:|---:|---|---|---:|
| LW1 | 21,617 | 10 | 2022-02-08 12:57:56 | 2022-02-22 10:42:54 | 30 s |
| CC1 | 32,529 | 9 | 2022-02-08 12:50:34 | 2022-02-24 19:59:48 | 19 s |
| CC2 | 34,865 | 8 | 2022-02-08 12:48:18 | 2022-02-15 12:47:53 | 17 s |
| 合计 | 89,011 | 27个“站点×日期”组 | — | — | — |

这说明后续轻量时序预测不再局限于364景。真正限制从“场景总数过少”转为：

- 独立站点只有3个；
- 活动和设备体系单一；
- 时间自相关很强；
- 某些站点存在多日中断；
- 不能把滑动窗口当作独立样本。

### 4.2 数据完整性

三站均满足：

- `.sigmf-meta`与`.sigmf-data`一一配对；
- orphan meta = 0；
- orphan data = 0；
- 每个真实`.sigmf-data`均为395,472字节；
- `395,472 = 98,868 × 4`，与98,868个float32功率值一致。

### 4.3 频率网格

每站按首、四分位、中位、四分之三和末尾位置抽取5个元数据，共核验15个频率轴：

| 项目 | 结果 |
|---|---|
| datatype | `rf32_le` |
| 频率点数 | 98,868 |
| 起始频率 | 87.16000366210938 MHz |
| 终止频率 | 6019.18017578125 MHz |
| 中位频率步长 | 0.06005859375 MHz |
| float64频率轴SHA-256 | `df21d4081cde1c5aeb3d2809aaa49a16fc7396eea75fe8b037fa6c50f3d8e66f` |
| 站内抽样一致 | 3/3 PASS |
| 跨站抽样一致 | PASS |

该结果证明抽样位置的频率轴一致，不等价于读取了所有89,011个元数据频率数组。结合所有功率文件字节数一致，可支持先建立统一流式开发缓存；缓存构建时仍必须逐景检查频率轴摘要或严格字段。

### 4.4 时间结构风险

- CC2采集连续性最好，最大相邻间隔约19秒；
- LW1和CC1包含跨日或多日中断；
- LW1在2月13日后到2月19日存在长缺口；
- CC1在2月15日后到2月24日存在长缺口。

因此预测窗口不得跨越大间隔。建议将窗口断点定义为：

`相邻间隔 > max(3 × 站点中位间隔，120秒)`

任何跨断点窗口均不进入训练、验证或测试。

### 4.5 后续拆分原则

S6.7b公平通信曲线不需要训练复杂预测器，可使用现有冻结模型和分组场景。

后续S7预测研究必须：

1. 以“站点×日期”为最小分组；
2. 进行leave-one-day或leave-one-site-out验证；
3. CC1/CC2的2月14—15日标记为历史C1-v4 Final已消费区；
4. 同一原始景的N = 8、16、32、64聚合不得跨拆分；
5. 归一化、标签阈值和合成器只能使用训练分组；
6. 2024—2025外部Final不能参与当前缓存设计或归一化。

---

## 5. 当前S6.7a门槛状态

| 门槛 | 状态 | 证据 |
|---|---|---|
| S6-FC0可唯一定位 | PASS | 22/22锁定文件哈希匹配 |
| 开发验收可复现 | PASS | 主结果和复现字节级一致 |
| 2022固定站点元数据可用 | PASS | 89,011个配对场景 |
| 抽样频率网格一致 | PASS | 15/15抽样一致 |
| 外部Final压缩包完整 | PASS | 3/3哈希匹配 |
| 外部Final零访问 | PASS | signal values未读取，count = 0 |
| 数据角色唯一 | PASS | 本文档数据账本 |
| 指标与bit字典 | PASS | v1.0指标字典 |
| S6.7b公平协议 | PASS | 预注册配置与预检 |

S6.7a已完成。S6.7b预检的11项治理检查、bit仪表层的6项演练检查和执行层的8项smoke检查全部通过；四种N的完整开发粗网格及bootstrap分析已完成，外部Final访问次数仍为0。

---

## 6. 下一步

已完成：

1. 明确`total_application_bits`、前向bit、反馈bit、预留容量bit和实际链路bit的关系；
2. 修正现有`ack_application_bits`字段实际同时包含ACK与心跳响应的命名歧义；
3. 固定clean、availability、conditional regret、effective regret和CVaR定义；
4. 规定短、中、长会话的安装摊销方式；
5. 写出S6.7b matched-reliability参数扫描和配对统计协议；
6. 在2023排除pilot上验证bit恒等式、预留容量和心跳请求/响应分解。

会话初始化、共同随机流、冻结风险排序器和2023开发粗网格均已完成。下一步是不访问外部Final，只用2022固定站点开发数据验证已经冻结的11个去重控制器候选。

---

## 7. 复现命令

在项目根目录执行：

```powershell
D:\anaconda\python.exe scripts\audit_stage6_s6_7a_metadata.py
```

输出：

`results/stage6/s6_7a_governance_audit_v1/governance_metadata_audit.json`

该命令会读取S6-FC0文件、2022固定站点ZIP中央目录及少量频率轴元数据，并对外部Final压缩包计算整体哈希；不会解压或读取外部Final SigMF信号。
