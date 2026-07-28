# Stage 4 Gate A Paired Hierarchical Analysis

5 training seeds x 150 paired validation scenes; 10000 bootstrap repetitions.

| Method vs detection-only | Regret delta [95% CI] | CVaR0.9 delta [95% CI] | nominal bit delta [95% CI] |
|---|---:|---:|---:|
| detection_plus_rate | 0.001363 [-0.000566, 0.003513] | 0.004907 [-0.004821, 0.015191] | -19.86 [-28.75, -8.10] |
| detection_plus_resource | -0.001729 [-0.003082, -0.000534] | -0.006419 [-0.013153, -0.001264] | 0.78 [-1.83, 3.53] |
| full_joint_loss | -0.000326 [-0.002251, 0.001505] | 0.001442 [-0.007452, 0.010088] | -14.35 [-18.84, -10.01] |

Decision: `retain_resource_aware_loss_for_C1`.

This is validation-only evidence; no final holdout was created or accessed.
