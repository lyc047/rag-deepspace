# Stage 4 Gate A Controlled Training

Five seeds; identical initialization and epoch permutations across methods. Development validation only.

| Method | Regret mean [95% CI] | CVaR0.9 | Brier | nominal bit |
|---|---:|---:|---:|---:|
| detection_only | 0.023624 [0.023152, 0.024095] | 0.130095 | 0.009204 | 505.7 |
| detection_plus_rate | 0.024987 [0.024353, 0.025621] | 0.135002 | 0.009608 | 485.8 |
| detection_plus_resource | 0.021895 [0.021482, 0.022308] | 0.123676 | 0.009223 | 506.4 |
| full_joint_loss | 0.023298 [0.022276, 0.024320] | 0.131178 | 0.014034 | 491.3 |

No final holdout was created or accessed; this report selects the next development decision only.
