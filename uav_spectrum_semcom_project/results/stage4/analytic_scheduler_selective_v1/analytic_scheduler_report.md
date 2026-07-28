# Stage 4 Analytic Scheduler Validation

Scheduler residual sketch used: True.

Train raw-score/value rank correlation: 0.1589; unique scores: 699; PAV knots: 4.

## Incremental budget 520 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| confidence_G3 | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| analytic_greedy | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| analytic_exact_knapsack | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| budget_oracle | 0.019945 | 0.125777 | 1889.31 | 0.6000 | 0.7800 |

## Incremental budget 600 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2363.38 | 0.5267 | 0.0000 |
| confidence_G3 | 0.026133 | 0.135445 | 2367.40 | 0.5267 | 0.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024436 | 0.131286 | 2294.81 | 0.5600 | 0.0000 |
| analytic_exact_knapsack | 0.024129 | 0.129044 | 2354.11 | 0.5467 | 0.0000 |
| budget_oracle | 0.013466 | 0.099042 | 1999.49 | 0.6733 | 0.5867 |

## Incremental budget 750 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2363.38 | 0.5267 | 0.0000 |
| confidence_G3 | 0.026133 | 0.135445 | 2367.40 | 0.5267 | 0.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024436 | 0.131286 | 2294.81 | 0.5600 | 0.0000 |
| analytic_exact_knapsack | 0.024129 | 0.129044 | 2354.11 | 0.5467 | 0.0000 |
| budget_oracle | 0.006864 | 0.056052 | 2050.01 | 0.7133 | 0.5200 |

## Incremental budget 950 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1775.34 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2363.38 | 0.5267 | 0.0000 |
| confidence_G3 | 0.026133 | 0.135445 | 2367.40 | 0.5267 | 0.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2293.37 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024436 | 0.131286 | 2294.81 | 0.5600 | 0.0000 |
| analytic_exact_knapsack | 0.024129 | 0.129044 | 2354.11 | 0.5467 | 0.0000 |
| budget_oracle | 0.006864 | 0.056052 | 2050.01 | 0.7133 | 0.5200 |

No calibration/validation labels were serialized and no final holdout was accessed.
