# 阶段4 Batch 13：Final统计判决器冻结

日期：2026-07-16  
数据：仅合成统计自检数据，不含RF、标签或final scene  
final access count：0

## 1. 缺口

此前协议已经登记两个主检验族、10,000次bootstrap、非劣界和Holm校正，但尚未把scene聚合层级、p值计算、intersection-union族判决和输入完整性落实为冻结代码。若等看到final后再写统计脚本，仍可能出现条件选择或方法调整。

## 2. 固定统计流程

新增`stage4_final_statistics_v1`：

1. 接收`stage4_final_scene_metrics_v1`；
2. 两个family必须使用完全相同的至少200个独立scene；
3. 对每个方法每scene只允许一行，五seed和链路重复必须在上游scene内聚合；
4. 使用10,000次配对scene bootstrap生成95% CI；
5. C1 CVaR0.9在scene级regret分布上计算；
6. 使用单侧配对随机化检验计算component p值；
7. 每个intersection-union family取component p值最大值；
8. 两个family p值做Holm校正；
9. family必须同时满足全部CI门槛和Holm调整p值≤0.05；次指标不能补救失败主检验。

C1 bit仍按预注册规则要求95% CI包含0；它是匹配成本门槛，不以“不显著”替代等效性声明。选择性G2的regret门槛固定为0.0026813。

## 3. 完整性和访问绑定

正式入口要求：

- access state必须为`access_consumed/count=1`；
- 输入registry SHA与access receipt一致；
- receipt自身SHA一致；
- scene数、方法名、配对集合、数值范围和schema全部合法。

任何错误都会拒绝统计，不自动删除scene或补齐方法。

## 4. 合成分支自检

使用240个纯合成数值scene验证代码行为：

- 成功分支：C1与选择性G2均通过；
- 失败分支：C1与选择性G2均失败；
- 坏输入分支：删除一条配对方法记录后被拒绝；
- 10,000次bootstrap和随机化检验均按冻结seed复现。

该自检只证明统计代码能正确执行预注册分支，不构成任何通信、感知或final性能证据。

## 5. 验证与结论

- final统计定向测试及协议测试：8项通过；
- 全项目：148项通过；
- 开发验收：无failed项；
- final访问：0。

至此，final前的统计定义、代码和失败处理均已冻结。剩余无法由代码替代的条件仍是新增独立scene及其一次性推理。
