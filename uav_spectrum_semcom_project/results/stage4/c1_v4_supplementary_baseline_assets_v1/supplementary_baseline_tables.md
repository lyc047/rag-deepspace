# C1-v4补充基线表（自动生成）

> 仅使用已经看过的开发验证缓存；属于Final后的解释性补充实验，
> 不读取Final测量值或Final指标，不改变已完成的单次Final结论。

## 表示基线

| 方法 | 应用层bit | 平均regret/dB | CVaR0.9/dB | 实际bit |
|---|---:|---:|---:|---:|
| hard_mean_energy_1bit | 160 | 0.644452 | 2.396475 | 690.751 |
| occupancy_fraction_4bit | 184 | 0.557435 | 2.384316 | 754.553 |
| soft_channel_power_4bit | 184 | 0.241049 | 0.826542 | 754.553 |
| soft_channel_power_8bit | 216 | 0.247873 | 0.852357 | 840.777 |
| proposed_best_block_indicator_1bit | 157 | 0.235225 | 0.806517 | 690.751 |

最优块指示相对各基线的直观变化：

- 相对硬能量判决：平均regret降低63.50%，CVaR降低66.35%；应用层少3 bit，但分包/FEC对齐后实际bit相同。
- 相对4-bit occupancy：平均regret降低57.80%，CVaR降低66.17%，实际bit降低8.46%。
- 相对4-bit软功率：平均regret降低2.42%，CVaR降低2.42%，实际bit降低8.46%。
- 相对8-bit软功率：平均regret降低5.10%，CVaR降低5.38%，实际bit降低17.84%。

## 可靠性基线

| 方法 | 平均regret/dB | CVaR0.9/dB | 实际bit | 链路时延/ms |
|---|---:|---:|---:|---:|
| no_arq_stateless | 0.239531 | 0.906824 | 476.000 | 1.288 |
| standard_arq_stateless | 0.171237 | 0.662148 | 690.575 | 1.869 |
| extra_arq_stateless | 0.136026 | 0.519891 | 842.123 | 2.279 |
| standard_arq_unlimited_hold | 0.122709 | 0.417162 | 690.575 | 1.869 |
| standard_arq_age60_hold | 0.137960 | 0.472561 | 690.575 | 1.869 |

- 60分钟保护相对标准ARQ无状态：平均regret降低19.43%，CVaR降低28.63%，实际bit不变。
- 60分钟保护相对额外ARQ：平均regret高1.42%，但CVaR低9.10%，实际bit低18.00%。
- 固定站点慢变化开发集上，无限保持的平均regret和CVaR比60分钟保护分别低11.05%和11.72%；这说明年龄保护付出了保守性代价，而当前数据没有覆盖其要防范的移动、突变或长失联风险。
