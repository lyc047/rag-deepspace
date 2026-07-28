# S6.6d：阶段六开发证据验收

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate + reproduce
- Origin Date：2026-07-26
- Verification Status：VERIFIED DEVELOPMENT ACCEPTANCE
- Data Role：仅审计冻结开发证据；未读取外部Final信号值
- Final Status：外部Final访问次数为0
- Protocol：`configs/stage6_development_acceptance_v1.json`
- Primary Result：`results/stage6/development_acceptance_v1/development_acceptance_result.json`
- Independent Reproduction：`results/stage6/development_acceptance_reproduction_v1/development_acceptance_result.json`
- Primary/Reproduction SHA-256：`48bdebfee9862807be9a980fcc21e8c09800e4559b20e42dc657d2d5ec3e84e9`

## 1. 验收结论

阶段六冻结候选通过全部14项开发验收门槛，结论为：

> **可以进入阶段六外部Final协议注册，但必须携带已知限制。**

该结论只授权编写、冻结和审计Final协议，不授权执行Final，不授权读取外部Final信号值，也不授权修改候选算法或参数。

## 2. 验收门槛

| 类别 | 门槛 | 结果 |
|---|---|---|
| 架构治理 | 架构冻结检查全部通过 | 通过 |
| 消融 | 六组消融检查全部通过 | 通过 |
| 学习模块 | 至少3种N获得配对区间支持 | 3/4，达到 |
| 故障压力 | 压力检查全部通过 | 通过 |
| 任务需求 | 三类需求全部满足ε = 0.2 dB | 通过 |
| 码本安全 | 错误码本动作执行次数为0 | 通过 |
| 确定性复现 | 消融、故障、需求主结果与复现哈希一致 | 通过 |
| 复杂度复现 | 至少23/24时间单元低于25%差异 | 23/24，达到 |
| 内存复现 | 四种N的峰值跟踪内存完全一致 | 通过 |
| 回归测试 | 至少322项且全部通过 | 323项，通过 |
| Final治理 | Final访问次数必须为0 | 0，满足 |

主验收与独立复现文件逐字节一致，证明验收判断本身可重复。

## 3. 已获得支持的开发结论

当前开发证据支持：

1. 联合任务码本在排除型开发活动上同时支持0.25N、0.50N和0.75N需求，最大regret不超过0.2 dB；
2. 学习型更新重复保护在3/4种N上得到区间支持，但增益只有0.100至0.319个百分点；
3. 固定静默心跳和上下文信念恢复是混合故障闭环的主要可靠性来源；
4. 冻结Python实现在N = 8至64范围内没有出现数量级复杂度爆炸；
5. 接收端上下文重置是当前系统最主要的合成故障弱点；
6. fail-closed码本身份检查在全部已运行压力单元中保持错误动作执行为0。

这些结论仍属于同一固定站点开发活动内的算法与协议证据。

## 4. Final前禁止声明的结论

在独立外部Final完成前，不得声明：

- 已经实现跨活动泛化或真实无人机移动泛化；
- 在相同clean rate下端到端bit降低20%以上；
- 合成故障概率等同于真实无人机链路故障概率；
- 机器学习是阶段六的主要性能来源；
- 已获得嵌入式时延、内存或能耗保证；
- 阶段六算法已通过外部Final。

其中，2-bit联合码字相对8至17 bit精确索引bundle降低75.0%至88.2%只属于**语义payload**结果，不能替代端到端应用层开销结论。

## 5. 必须携带到Final协议的限制

1. 完整系统相对逐场景精确动作包虽然节省28.46%至37.62%应用层bit，但clean rate低约11.4至14.2个百分点；
2. 有效CVaR₀.₉仍被不可用场景和10 dB掉线惩罚主导；
3. 重置概率从0升至0.05时，clean下降约28.4至30.3个百分点；
4. ACK丢失下维持clean需要额外28.24至36.84 bit/场景；
5. 学习模块只能作为辅助可靠性创新；
6. 复杂度数据只代表当前桌面Python环境；
7. 当前开发活动是固定站点频谱，不是真实无人机移动轨迹。

这些限制不是“待隐藏的问题”，而是Final协议分层报告和论文讨论章节的强制内容。

## 6. 证据等级

| 证据 | 等级 | 原因 |
|---|---|---|
| 码本任务约束与需求分解 | VERIFIED DEVELOPMENT | 确定性主结果与复现哈希一致 |
| 消融效应 | VERIFIED DEVELOPMENT | 配对蒙特卡洛、预注册、精确复现 |
| 故障概率曲线 | VERIFIED SYNTHETIC STRESS | 确定性复现，但故障为控制注入 |
| 复杂度 | PARTIALLY REPRODUCIBLE | 23/24时间单元达标，单点受系统抖动影响 |
| 跨活动泛化 | UNTESTED | 尚未执行阶段六外部Final |
| 真实无人机可靠性 | UNTESTED | 缺少真实移动链路和现场故障标签 |

## 7. 最终判断

阶段六开发工作量已经达到“能够注册一次独立外部Final”的条件，但尚未达到“研究结论已经最终成立”的条件。

下一步为S6.7：

1. 冻结外部Final数据清单和排除规则；
2. 固定一次性访问状态；
3. 固定完整候选及所有对照方法；
4. 固定主指标、置信区间、多重比较和失败判定；
5. 固定输出文件与哈希；
6. 在协议审计通过后，才允许进入S6.8的一次性外部Final。
