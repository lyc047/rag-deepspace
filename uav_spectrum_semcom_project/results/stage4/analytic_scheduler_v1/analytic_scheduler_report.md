# Stage 4 Analytic Scheduler Validation

Train raw-score/value rank correlation: undefined_constant_score; unique scores: 1; PAV knots: 2.

## Incremental budget 520 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| confidence_G3 | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_greedy | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_exact_knapsack | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| budget_oracle | 0.019945 | 0.125777 | 1748.42 | 0.6000 | 0.7800 |

## Incremental budget 600 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2222.50 | 0.5267 | 0.0000 |
| confidence_G3 | 0.028963 | 0.145418 | 1768.48 | 0.4733 | 0.7733 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_greedy | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_exact_knapsack | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| budget_oracle | 0.013466 | 0.099042 | 1858.61 | 0.6733 | 0.5867 |

## Incremental budget 750 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2222.50 | 0.5267 | 0.0000 |
| confidence_G3 | 0.028995 | 0.143371 | 1908.61 | 0.4733 | 0.5800 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_greedy | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_exact_knapsack | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| budget_oracle | 0.006864 | 0.056052 | 1909.13 | 0.7133 | 0.5200 |

## Incremental budget 950 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2222.50 | 0.5267 | 0.0000 |
| confidence_G3 | 0.028995 | 0.143371 | 1908.61 | 0.4733 | 0.5800 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_greedy | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| analytic_exact_knapsack | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| budget_oracle | 0.006864 | 0.056052 | 1909.13 | 0.7133 | 0.5200 |

No calibration/validation labels were serialized and no final holdout was accessed.
