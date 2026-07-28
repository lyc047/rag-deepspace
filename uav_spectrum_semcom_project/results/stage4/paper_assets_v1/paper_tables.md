# Stage-4 thesis tables (generated)

> Development evidence only; final holdout access count is 0.

## Gate A paired effects

| Method vs detection-only | Regret diff | Regret 95% CI | CVaR diff | CVaR 95% CI | Bit diff | Bit 95% CI |
|---|---|---|---|---|---|---|
| detection_plus_rate | 0.001363 | [-0.000566, 0.003513] | 0.004907 | [-0.004821, 0.015191] | -19.86 | [-28.75, -8.10] |
| detection_plus_resource | -0.001729 | [-0.003082, -0.000534] | -0.006419 | [-0.013153, -0.001264] | 0.78 | [-1.83, 3.53] |
| full_joint_loss | -0.000326 | [-0.002251, 0.001505] | 0.001442 | [-0.007452, 0.010088] | -14.35 | [-18.84, -10.01] |

## C2 value diagnostic

| Diagnostic | Value |
|---|---|
| Value/bit MAE | 2.5263892e-05 |
| Pooled Spearman | 0.00938 |
| Top-1 / Top-2 / Top-3 | 0.1200 / 0.2133 / 0.3000 |
| Selected regret reduction | 0.003097 |
| Oracle regret reduction | 0.022672 |

## Complexity

| Component | Parameters | FP32 bytes | Linear MACs | Median ms | p95 ms |
|---|---|---|---|---|---|
| C1 head / scene | 1740 | 6960 | 1664 | 0.4238 | 0.7075 |
| C2 / 8 candidates / 1 seed | 5697 | 22788 | 44800 | 0.0695 | 0.1201 |
| C2 / 8 candidates / 5 seeds | 28485 | 113940 | 224000 | 0.3502 | 0.5706 |

## Coverage status

| Status | Count |
|---|---|
| blocked | 2 |
| failed | 0 |
| passed | 19 |
| waived | 7 |
