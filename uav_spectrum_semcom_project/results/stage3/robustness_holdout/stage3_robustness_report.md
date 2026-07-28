# Stage 3 Robustness and Failure-Boundary Report

Fresh holdout: 200 temporally overlapping semi-synthetic scenes, 5 repeated link/noise realizations.
The first 800 previously consumed test sources are excluded; this run starts at frozen test-source offset 1200.
All fusion methods within a scenario use identical delivered reports and transmitted-bit traces.

## Confidence calibration

Validation samples: 320; raw quality MSE 0.307967; calibrated MSE 0.227169.
Fresh-holdout raw quality MSE 0.412519; calibrated MSE 0.266312.

## Results

| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |
|---|---|---:|---:|---:|---:|
| baseline_4_nodes | majority | 0.7770 | 0.009931 | 0.4520 | 3375.5 |
| baseline_4_nodes | mean | 0.7890 | 0.008752 | 0.5940 | 3375.5 |
| baseline_4_nodes | or | 0.7900 | 0.008747 | 0.5940 | 3375.5 |
| baseline_4_nodes | quality_weighted | 0.7900 | 0.008772 | 0.5950 | 3375.5 |
| baseline_4_nodes | snr_weighted | 0.7900 | 0.008679 | 0.5960 | 3375.5 |
| best_node_failure | majority | 0.7770 | 0.009983 | 0.4420 | 2531.1 |
| best_node_failure | mean | 0.7820 | 0.009429 | 0.5160 | 2531.1 |
| best_node_failure | or | 0.7820 | 0.009385 | 0.5170 | 2531.1 |
| best_node_failure | quality_weighted | 0.7800 | 0.009491 | 0.5150 | 2531.1 |
| best_node_failure | snr_weighted | 0.7790 | 0.009505 | 0.5160 | 2531.1 |
| best_single_node | majority | 0.7780 | 0.009655 | 0.5090 | 844.4 |
| best_single_node | mean | 0.7930 | 0.008666 | 0.5970 | 844.4 |
| best_single_node | or | 0.7930 | 0.008666 | 0.5970 | 844.4 |
| best_single_node | quality_weighted | 0.7930 | 0.008666 | 0.5970 | 844.4 |
| best_single_node | snr_weighted | 0.7930 | 0.008666 | 0.5970 | 844.4 |
| combined_stale_failure_anomaly | majority | 0.7710 | 0.010706 | 0.4080 | 2675.9 |
| combined_stale_failure_anomaly | mean | 0.7950 | 0.009808 | 0.4410 | 2675.9 |
| combined_stale_failure_anomaly | or | 0.7940 | 0.009898 | 0.4400 | 2675.9 |
| combined_stale_failure_anomaly | quality_weighted | 0.7930 | 0.009845 | 0.4390 | 2675.9 |
| combined_stale_failure_anomaly | snr_weighted | 0.7930 | 0.009845 | 0.4390 | 2675.9 |
| heterogeneous_2_nodes | majority | 0.7780 | 0.009655 | 0.5090 | 1580.3 |
| heterogeneous_2_nodes | mean | 0.7890 | 0.008889 | 0.5930 | 1580.3 |
| heterogeneous_2_nodes | or | 0.7900 | 0.008846 | 0.5940 | 1580.3 |
| heterogeneous_2_nodes | quality_weighted | 0.7910 | 0.008800 | 0.5950 | 1580.3 |
| heterogeneous_2_nodes | snr_weighted | 0.7900 | 0.008821 | 0.5940 | 1580.3 |
| high_confidence_wrong | majority | 0.7770 | 0.010096 | 0.4330 | 3375.5 |
| high_confidence_wrong | mean | 0.7930 | 0.008673 | 0.5350 | 3375.5 |
| high_confidence_wrong | or | 0.7960 | 0.008624 | 0.5360 | 3375.5 |
| high_confidence_wrong | quality_weighted | 0.7890 | 0.009084 | 0.5270 | 3375.5 |
| high_confidence_wrong | snr_weighted | 0.7900 | 0.008891 | 0.5320 | 3375.5 |
| high_snr_stale | majority | 0.7740 | 0.010193 | 0.4300 | 3372.8 |
| high_snr_stale | mean | 0.7810 | 0.010605 | 0.4850 | 3372.8 |
| high_snr_stale | or | 0.7790 | 0.010679 | 0.4810 | 3372.8 |
| high_snr_stale | quality_weighted | 0.7790 | 0.010617 | 0.4840 | 3372.8 |
| high_snr_stale | snr_weighted | 0.7820 | 0.010584 | 0.4860 | 3372.8 |

## Paired proposed-minus-mean differences

Negative regret delta favors quality-aware fusion.

| Scenario | Regret delta (95% CI) | Clean-rate delta (95% CI) |
|---|---:|---:|
| baseline_4_nodes | 0.000021 [-0.000277, 0.000319] | 0.0010 [-0.0027, 0.0047] |
| best_node_failure | 0.000061 [-0.000031, 0.000153] | -0.0020 [-0.0044, 0.0004] |
| best_single_node | 0.000000 [0.000000, 0.000000] | 0.0000 [0.0000, 0.0000] |
| combined_stale_failure_anomaly | 0.000037 [-0.000019, 0.000094] | -0.0020 [-0.0044, 0.0004] |
| heterogeneous_2_nodes | -0.000088 [-0.000213, 0.000037] | 0.0020 [-0.0004, 0.0044] |
| high_confidence_wrong | 0.000411 [0.000194, 0.000627] | -0.0040 [-0.0112, 0.0032] |
| high_snr_stale | 0.000012 [-0.000201, 0.000225] | -0.0020 [-0.0070, 0.0030] |

## Claim boundary

These are controlled RadDet spectrogram perturbations and sliding max-hold scenes, not synchronized multi-receiver measurements.
A scenario supports H3 only when the paired regret-delta 95% interval is below zero at identical bit cost.
