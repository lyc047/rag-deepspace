# C1-v4 Final论文表格（自动生成）

> 数据来自已冻结的单次Final结果；不得据此重新调参或重跑Final。

## 总体结果

| 方法 | 应用层bit | 平均regret/dB | CVaR0.9/dB | 实际发送bit | 帧成功率 |
|---|---:|---:|---:|---:|---:|
| occupancy_stateless | 184 | 0.708167 | 2.402559 | 755.448 | 0.675298 |
| indicator_stateless | 157 | 0.302646 | 0.944649 | 691.206 | 0.682321 |
| indicator_guarded_last_success | 157 | 0.217700 | 0.642915 | 691.206 | 0.682321 |

## 关键相对改善

- H1相对occupancy：平均regret降低57.26%，CVaR降低60.68%，实际bit降低8.50%。
- H2相对无状态块指示：平均regret继续降低28.07%，CVaR继续降低31.94%，bit差为0。
- 完整V4相对occupancy：平均regret降低69.26%，CVaR降低73.24%。
