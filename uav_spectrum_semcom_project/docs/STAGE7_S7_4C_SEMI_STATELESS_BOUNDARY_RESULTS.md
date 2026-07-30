# 阶段7 S7.4C 半无状态帧头边界结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-07-30
- Verification Status：VERIFIED
- Version Label：stage7_s7_4c0_semi_stateless_header_audit_v1
- Evidence Role：基于已完成S7.4B轨迹统计的解析重计费，不是外部 Final
- Raw Signal Files Loaded：0

## 1. 目的

本实验判断：若会话预先固定码本目录，将事件更新的身份增量从实测24 bit缩短为16 bit或8 bit，是否值得实现新的半无状态codec。

同时计算0 bit身份的不可实现上界。该上界只用于排除研究分支，不能作为协议候选。

## 2. 方法

保持S7.4B的包投递、ACK、动作、clean和故障轨迹完全不变，只重计身份bit：

`调整后总bit = 原总bit − 原身份开销 × (1 − 新字段宽度 ÷ 24)`

扫描字段宽度0、8、16和24 bit，并保持12、16、24景三种心跳策略不变。候选必须在同一字段宽度和同一策略下至少通过3/4种N。

## 3. 关键结果

即使使用0 bit不可实现上界，所有策略仍为0/4种N通过：

- 12景策略的0-bit点估计bit降幅为3.39%—6.27%，但clean平均下降0.54—0.67个百分点，四种N的clean区间下界均低于−0.5个百分点；
- 16景策略的0-bit点估计bit降幅为6.57%—11.68%，但clean下降2.31—3.00个百分点；
- 24景策略的0-bit点估计bit降幅为10.16%—18.14%，但clean下降5.30—7.17个百分点；
- 8、16和24 bit候选不可能优于0 bit上界，因此均不需要实现。

这说明S7.4B失败的主因已经从“身份字段过长”排除为“心跳变稀后随机重置恢复不及时”。缩短帧头只能改善bit，不能恢复clean。

## 4. 决策

1. 不实现8-bit或16-bit半无状态codec；
2. 固定10景心跳S6R-FH10-v1冻结为阶段7可靠性工程边界；
3. 保留S7.3b、S7.4A、S7.4B和S7.4C的连续负结果，形成协议机制消融链；
4. 不打开内部开发测试集、储备集或新Final数据；
5. 阶段7不再扩展预测心跳、周期检查点、事件捎带或更短身份帧；
6. 下一步转入冻结架构的计算复杂度、会话开销和论文证据整合。

## 5. 统计与逻辑风险

总体置信度：SOLID用于开发分支排除，不能用于外部泛化主张。

本实验使用不可能实现的0 bit上界；若上界仍失败，则任何增加正身份bit的同轨迹方案都不能通过。该结论不依赖挑选某个N。未进行新的多重数据搜索，也未读取原始信号。

11项统计谬误已检查：无聚合反转式挑选、无个体层外推、无极端样本筛选、无轨迹删除、无结果后改门槛；因果解释仅限冻结仿真故障模型。覆盖11/11。

## 6. 证据

- 配置：`configs/stage7_s7_4c0_semi_stateless_header_audit_v1.json`
- 执行器：`scripts/run_stage7_s7_4c0_semi_stateless_header_audit.py`
- 主结果：`results/stage7/s7_4c0_semi_stateless_header_audit_v1/result.json`
- 复现结果：`results/stage7/s7_4c0_semi_stateless_header_audit_v1/reproduction.json`
- 主结果与复现结果除耗时字段外逐字段一致。
