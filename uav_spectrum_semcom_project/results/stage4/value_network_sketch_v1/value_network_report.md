# Stage 4 C2 Value Network Validation

G1 preview probability bits: 1.

Train groups/rows: 2700 / 18000; features: 55; five preregistered seeds, fixed final epoch.

| Method | regret | CVaR | expected bit | clean rate | no upgrade |
|---|---:|---:|---:|---:|---:|
| no_upgrade | 0.028172 | 0.145275 | 2012.81 | 0.4800 | 1.0000 |
| sensing_snr_G3 | 0.024979 | 0.134724 | 2600.85 | 0.5267 | 0.0000 |
| confidence_G3 | 0.027198 | 0.134602 | 2959.65 | 0.4733 | 0.0000 |
| stage3_prior_value_per_bit | 0.025368 | 0.136736 | 2530.84 | 0.5267 | 0.0000 |
| learned_counterfactual_value_per_bit | 0.025594 | 0.120926 | 2644.78 | 0.5067 | 0.0067 |
| oracle | 0.005500 | 0.050784 | 2349.05 | 0.7400 | 0.4733 |

Learned ensemble validation regret/bit: 0.025594 / 2644.78.

No validation labels were serialized; no final holdout was created or accessed.
