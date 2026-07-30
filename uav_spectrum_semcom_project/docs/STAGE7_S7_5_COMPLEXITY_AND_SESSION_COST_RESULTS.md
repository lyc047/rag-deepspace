# 阶段7 S7.5 冻结架构复杂度与会话开销结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-07-30
- Verification Status：ANALYZED；门槛方向经独立复测一致，绝对时延存在系统抖动
- Version Labels：stage7_s7_5a_complexity_benchmark_v1；stage7_s7_5b_offline_fit_scaling_v1
- Evidence Role：当前Windows桌面CPU上的工程开发证据，不是无人机板载验证
- Raw External Signal Files Loaded：0

## 1. 测试边界

测试使用确定性合成频谱，只用于执行冻结算法路径和测量计算成本，不评价无线性能或泛化能力。

报告的峰值内存来自Python `tracemalloc`，不等于进程总RSS；时延来自当前电脑上的Anaconda Python，不能直接外推到ARM、Jetson或飞控处理器。

## 2. 在线紧凑编解码

| N | 编码加解码p95 | 独立复测p95 | 峰值Python分配 |
|---:|---:|---:|---:|
| 8 | 0.280 ms | 0.308 ms | 1.86 MiB |
| 16 | 0.524 ms | 0.598 ms | 2.32 MiB |
| 32 | 0.725 ms | 0.936 ms | 3.61 MiB |
| 64 | 0.710 ms | 0.802 ms | 5.83 MiB |

四种N两轮p95均低于预注册1 ms门槛，说明Python原型的单次紧凑编解码不是当前在线瓶颈。绝对时延两轮并未全部满足5%复现差异，因此只保留范围和门槛判断，不把小数点后的差异解释为算法效应。

## 3. 离线码本构造

S7.5A首先对1024景进行无界贪心覆盖，生成45—143个码字。N = 32和64耗时约9.84和14.37 s，超过5 s门槛；独立复测仍分别约10.17和15.56 s，失败方向一致。

这一压力条件不等于冻结K = 3码字预算。S7.5B按K = 3重新测试：

| N | 64景拟合 | 256景拟合 | 1024景拟合 | 1024景复测 |
|---:|---:|---:|---:|---:|
| 8 | 0.006 s | 0.055 s | 0.354 s | 0.409 s |
| 16 | 0.011 s | 0.131 s | 1.025 s | 1.096 s |
| 32 | 0.029 s | 0.192 s | 2.308 s | 2.543 s |
| 64 | 0.027 s | 0.259 s | 3.627 s | 3.683 s |

K = 3条件4/4种N通过：64景均低于1 s，1024景均低于5 s。结论应写成：

> 在线紧凑编解码开销较低；有限码字预算下的离线构造在当前CPU上可接受，但无界码本覆盖随候选动作数量增长明显，不适合在线重建。

## 4. 会话启动摊销

预装码本激活64 bit，加24 bit ACK，共88 bit。仅计算一次启动开销：

| 会话长度 | 启动摊销 |
|---:|---:|
| 10景 | 8.8 bit/景 |
| 100景 | 0.88 bit/景 |
| 1000景 | 0.088 bit/景 |
| 10000景 | 0.0088 bit/景 |

因此预装激活适合中长会话；对于只有约10景的极短会话，启动开销不可忽略。固定心跳、任务更新、ACK和恢复属于持续开销，已经在S7.3b—S7.4B完整协议结果中另行计入，不能与本表重复相加。

## 5. 工程结论

1. 当前桌面CPU上，在线codec p95低于1 ms，Python分配峰值低于6 MiB；
2. K = 3离线拟合即使1024景也低于4 s，适合作为任务前或低频校准；
3. 无界贪心码本构造在N = 32/64超过5 s，应明确限制码字预算并禁止在线无界重建；
4. 启动成本在1000景以上可忽略，在10景短会话中明显；
5. 尚未测量真实板载CPU、总RSS、功耗、无线栈时延和硬件在环表现，不能宣称“已经适配无人机部署”。

## 6. 统计与方法风险

总体置信度：CAUTION。

- Simpson悖论：四种N逐项报告，没有用平均值掩盖N = 32/64无界拟合失败。
- 生态谬误：不从桌面CPU推断无人机处理器。
- Berkson与碰撞偏差：不适用于合成计算基准，但合成负载代表性有限。
- 基准率忽视：同时报告在线、离线、内存和会话摊销。
- 均值回归与幸存者偏差：没有删除慢批次或失败N。
- 多重搜索与分叉路径：门槛、规模和重复方式运行前注册；S7.5B由预注册失败分支触发。
- 相关不等于因果与反向因果：结论仅限受控程序路径，不外推真实平台。

覆盖：11/11。

## 7. 证据

- S7.5A配置：`configs/stage7_s7_5a_complexity_benchmark_v1.json`
- S7.5A执行器：`scripts/run_stage7_s7_5a_complexity_benchmark.py`
- S7.5A主结果与复测：`results/stage7/s7_5a_complexity_benchmark_v1/`
- S7.5B配置：`configs/stage7_s7_5b_offline_fit_scaling_v1.json`
- S7.5B执行器：`scripts/run_stage7_s7_5b_offline_fit_scaling.py`
- S7.5B主结果与复测：`results/stage7/s7_5b_offline_fit_scaling_v1/`
