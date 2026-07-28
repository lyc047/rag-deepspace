# 阶段6 S6.7b固定站点缓存与分组验证候选冻结

版本：1.0  
日期：2026-07-27  
状态：缓存完成，候选已冻结，分组性能结果尚未访问

## Material Passport

- Verification Status：EXECUTED, HASHED AND FROZEN
- Cache：`results/stage6/fixed_site_stream_cache_v1/cache.npz`
- Cache SHA-256：`93c7a49548f05e2e53c6789fabc845de711fcf101869be5331af607c5b292305`
- Cache Result：`results/stage6/fixed_site_stream_cache_v1/result.json`
- Cache Result SHA-256：`d1c771bf0b3bc02ebade5d945f642f2b030ef4a899c0cb0b930efba0efc10caf`
- Candidate Freeze：`configs/stage6_grouped_validation_candidates_v1.json`
- Candidate Freeze SHA-256：`c52e0540b2533ca71ae7c8e02f3bf6fabfd88c1910702a0d9eff2dfe103ce7b7`
- External Final Access Count：0

## 1. 缓存构建结果

三份2022固定站点ZIP没有完整解压。脚本逐条流式读取功率成员，只保留3550—3700 MHz任务频带，并生成N = 8、16、32、64的平均信道功率。

| 站点 | 场景数 | 本地日期数 | 中位扫描间隔 |
|---|---:|---:|---:|
| LW1 | 21,617 | 10 | 30 s |
| CC1 | 32,529 | 9 | 19 s |
| CC2 | 34,865 | 8 | 17 s |
| 合计 | 89,011 | 27 | — |

缓存质量结果：

- 频带bin数：2,500；
- 原始频带功率范围：−137.57至−108.23 dBm；
- N = 8、16、32、64数组均为有限值；
- 27个站点×日期组对应27个连续会话段；
- 缓存大小约32.9 MB；
- 构建耗时约307秒；
- 外部Final文件未打开。

## 2. 候选冻结

粗网格产生的工作点先按控制器参数去重，会话长度和部署模式只作为评估条件。最终冻结：

- N = 8：3个控制器；
- N = 16：3个控制器；
- N = 32：3个控制器；
- N = 64：2个控制器；
- 合计11个控制器。

全部候选的最大状态年龄为30分钟，风险预留比例为0。候选只在H = 10或20、基础发送次数1—3之间变化。

## 3. 分组验证规则

后续固定执行：

1. 使用2023训练组确定的任务码本，不在2022数据上重拟合；
2. 不重新训练一步风险排序器；
3. 对每个候选分别评估L = 20、100、500；
4. 分别评估预配置与在线安装；
5. 精确动作基线固定为每景1、2、3次自包含发送；
6. 每个条件使用60条配对链路轨迹；
7. 以站点×日期作为外层统计组；
8. 报告leave-one-site-out；
9. 结果访问后不得增删候选、改变clean目标或切段规则。

本阶段仍属于开发验证。即使分组结果通过，也只能说明跨站点、跨日期稳定性增强，不能替代2024—2025一次性外部Final。
