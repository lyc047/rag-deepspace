# Stage 4 C2 Value Network Validation

Train groups/rows: 2700 / 18000; features: 55; five preregistered seeds, fixed final epoch.

| Method | regret | CVaR | expected bit | clean rate | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 1634.46 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2222.50 | 0.5267 | 0.0000 |
| confidence_G3 | 0.028772 | 0.139949 | 2553.54 | 0.4533 | 0.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2152.49 | 0.5267 | 0.0000 |
| learned_counterfactual_value_per_bit | 0.025075 | 0.114811 | 2215.33 | 0.4867 | 0.0067 |
| oracle | 0.005500 | 0.050784 | 1970.69 | 0.7400 | 0.4733 |

Learned ensemble validation regret/bit: 0.025075 / 2215.33.

No validation labels were serialized; no final holdout was created or accessed.
