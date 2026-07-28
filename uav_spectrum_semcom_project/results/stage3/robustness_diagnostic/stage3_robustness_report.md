# Stage 3 Robustness and Failure-Boundary Report

Fresh holdout: 20 temporally overlapping semi-synthetic scenes, five repeated link/noise realizations.
The first 800 previously consumed test sources are excluded; this run starts at frozen test-source offset 1200.
All fusion methods within a scenario use identical delivered reports and transmitted-bit traces.

## Confidence calibration

Validation samples: 320; raw quality MSE 0.005367; calibrated MSE 0.000379.

## Results

| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |
|---|---|---:|---:|---:|---:|
| baseline_4_nodes | majority | 0.6000 | 0.021500 | 0.1000 | 3226.3 |
| baseline_4_nodes | mean | 0.7000 | 0.016871 | 0.4000 | 3226.3 |
| baseline_4_nodes | or | 0.7500 | 0.015115 | 0.4500 | 3226.3 |
| baseline_4_nodes | quality_weighted | 0.7000 | 0.016871 | 0.4000 | 3226.3 |
| baseline_4_nodes | snr_weighted | 0.7000 | 0.016871 | 0.4000 | 3226.3 |
| best_node_failure | majority | 0.6000 | 0.021500 | 0.1000 | 2412.9 |
| best_node_failure | mean | 0.7000 | 0.018781 | 0.3000 | 2412.9 |
| best_node_failure | or | 0.7000 | 0.018781 | 0.3000 | 2412.9 |
| best_node_failure | quality_weighted | 0.7000 | 0.018781 | 0.3000 | 2412.9 |
| best_node_failure | snr_weighted | 0.7000 | 0.018781 | 0.3000 | 2412.9 |
| best_single_node | majority | 0.6500 | 0.020316 | 0.1500 | 813.4 |
| best_single_node | mean | 0.7500 | 0.015115 | 0.4500 | 813.4 |
| best_single_node | or | 0.7500 | 0.015115 | 0.4500 | 813.4 |
| best_single_node | quality_weighted | 0.7500 | 0.015115 | 0.4500 | 813.4 |
| best_single_node | snr_weighted | 0.7500 | 0.015115 | 0.4500 | 813.4 |
| combined_stale_failure_anomaly | majority | 0.6000 | 0.020994 | 0.1000 | 2526.3 |
| combined_stale_failure_anomaly | mean | 0.7000 | 0.018585 | 0.1500 | 2526.3 |
| combined_stale_failure_anomaly | or | 0.7000 | 0.018585 | 0.1500 | 2526.3 |
| combined_stale_failure_anomaly | quality_weighted | 0.7000 | 0.018585 | 0.1500 | 2526.3 |
| combined_stale_failure_anomaly | snr_weighted | 0.7000 | 0.018585 | 0.1500 | 2526.3 |
| heterogeneous_2_nodes | majority | 0.6500 | 0.020316 | 0.1500 | 1479.8 |
| heterogeneous_2_nodes | mean | 0.7500 | 0.015115 | 0.4500 | 1479.8 |
| heterogeneous_2_nodes | or | 0.7500 | 0.015115 | 0.4500 | 1479.8 |
| heterogeneous_2_nodes | quality_weighted | 0.7500 | 0.015115 | 0.4500 | 1479.8 |
| heterogeneous_2_nodes | snr_weighted | 0.7500 | 0.015115 | 0.4500 | 1479.8 |
| high_confidence_wrong | majority | 0.6000 | 0.021576 | 0.0500 | 3226.3 |
| high_confidence_wrong | mean | 0.7000 | 0.018347 | 0.2000 | 3226.3 |
| high_confidence_wrong | or | 0.7000 | 0.018347 | 0.2000 | 3226.3 |
| high_confidence_wrong | quality_weighted | 0.7000 | 0.018347 | 0.2000 | 3226.3 |
| high_confidence_wrong | snr_weighted | 0.7000 | 0.018347 | 0.2000 | 3226.3 |
| high_snr_stale | majority | 0.6000 | 0.021500 | 0.1000 | 3182.2 |
| high_snr_stale | mean | 0.7000 | 0.017773 | 0.2500 | 3182.2 |
| high_snr_stale | or | 0.6500 | 0.018957 | 0.2000 | 3182.2 |
| high_snr_stale | quality_weighted | 0.7000 | 0.017573 | 0.3000 | 3182.2 |
| high_snr_stale | snr_weighted | 0.7000 | 0.017773 | 0.2500 | 3182.2 |

## Paired proposed-minus-mean differences

Negative regret delta favors quality-aware fusion.

| Scenario | Regret delta (95% CI) | Clean-rate delta (95% CI) |
|---|---:|---:|
| baseline_4_nodes | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| best_node_failure | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| best_single_node | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| combined_stale_failure_anomaly | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| heterogeneous_2_nodes | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| high_confidence_wrong | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| high_snr_stale | -0.000200 [-0.000200, -0.000200] | 0.0000 [0.0000, 0.0000] |

## Claim boundary

These are controlled RadDet spectrogram perturbations and sliding max-hold scenes, not synchronized multi-receiver measurements.
A scenario supports H3 only when the paired regret-delta 95% interval is below zero at identical bit cost.
