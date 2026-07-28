# Stage 4 Analytic Scheduler Validation

Scheduler residual sketch used: True.

Train raw-score/value rank correlation: 0.0950; unique scores: 1257; PAV knots: 4.

## Incremental budget 520 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| confidence_G3 | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2530.84 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024717 | 0.136736 | 2403.06 | 0.5400 | 0.2467 |
| analytic_exact_knapsack | 0.024717 | 0.136736 | 2403.06 | 0.5400 | 0.2467 |
| budget_oracle | 0.019945 | 0.125777 | 2126.78 | 0.6000 | 0.7800 |

## Incremental budget 600 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2600.85 | 0.5267 | 0.0000 |
| confidence_G3 | 0.029047 | 0.145418 | 2115.28 | 0.4733 | 0.8267 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2530.84 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024436 | 0.131286 | 2497.75 | 0.5600 | 0.0667 |
| analytic_exact_knapsack | 0.024129 | 0.129044 | 2557.05 | 0.5467 | 0.0667 |
| budget_oracle | 0.013466 | 0.099042 | 2236.97 | 0.6733 | 0.5867 |

## Incremental budget 750 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2600.85 | 0.5267 | 0.0000 |
| confidence_G3 | 0.029078 | 0.143371 | 2255.41 | 0.4733 | 0.6333 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2530.84 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024114 | 0.129637 | 2526.95 | 0.5600 | 0.0200 |
| analytic_exact_knapsack | 0.023759 | 0.127246 | 2593.49 | 0.5467 | 0.0200 |
| budget_oracle | 0.006864 | 0.056052 | 2287.49 | 0.7133 | 0.5200 |

## Incremental budget 950 bit

| Method | regret | CVaR | total bit | clean | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2600.85 | 0.5267 | 0.0000 |
| confidence_G3 | 0.029078 | 0.143371 | 2255.41 | 0.4733 | 0.6333 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2530.84 | 0.5267 | 0.0000 |
| analytic_greedy | 0.024114 | 0.129637 | 2526.95 | 0.5600 | 0.0200 |
| analytic_exact_knapsack | 0.023759 | 0.127246 | 2593.49 | 0.5467 | 0.0200 |
| budget_oracle | 0.006864 | 0.056052 | 2287.49 | 0.7133 | 0.5200 |

No calibration/validation labels were serialized and no final holdout was accessed.
