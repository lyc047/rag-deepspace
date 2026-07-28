# Stage 3 Adaptive ACK and Risk Fallback

Validation offset 3000; single fresh holdout offset 4000.
The scheduler starts with two value/bit-ranked nodes, adds candidates after delivery failure, and expands to three reports after disagreement or staleness.
All fallback transmissions are included in the bit count.

## Frozen policy

Pair-disagreement threshold: 0.0; stale trigger: 0.1 s; target reports: 3; fusion: robust_quality_weighted.
Validation node priors: [0.5375, 0.5, 0.5, 0.625]; expected bit/node: [1070.5, 787.2, 719.1, 816.6].

## Fresh holdout results

| Scenario | Method | Clean | Regret | bit/decision | attempted nodes | ACK fallback | risk expansion |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_4_nodes | adaptive_mean | 0.7630 | 0.013133 | 2397.8 | 3.06 | 0.000 | 1.000 |
| baseline_4_nodes | fixed_two_mean | 0.7660 | 0.013021 | 1578.6 | 2.00 | 0.000 | 0.000 |
| baseline_4_nodes | full_mean | 0.7630 | 0.013133 | 3427.9 | 4.00 | 0.000 | 0.000 |
| baseline_4_nodes | proposed_adaptive_robust | 0.7670 | 0.012965 | 2397.8 | 3.06 | 0.000 | 1.000 |
| best_node_failure | adaptive_mean | 0.7000 | 0.015039 | 2548.9 | 4.00 | 1.000 | 0.945 |
| best_node_failure | fixed_two_mean | 0.6980 | 0.014954 | 699.7 | 2.00 | 0.000 | 0.000 |
| best_node_failure | full_mean | 0.7000 | 0.015039 | 2548.9 | 4.00 | 0.000 | 0.000 |
| best_node_failure | proposed_adaptive_robust | 0.6990 | 0.015072 | 2548.9 | 4.00 | 1.000 | 0.945 |
| best_single_node | adaptive_mean | 0.7650 | 0.013796 | 878.9 | 1.00 | 1.000 | 0.000 |
| best_single_node | fixed_two_mean | 0.7650 | 0.013796 | 878.9 | 1.00 | 0.000 | 0.000 |
| best_single_node | full_mean | 0.7650 | 0.013796 | 878.9 | 1.00 | 0.000 | 0.000 |
| best_single_node | proposed_adaptive_robust | 0.7650 | 0.013796 | 878.9 | 1.00 | 1.000 | 0.000 |
| combined_stale_failure_anomaly | adaptive_mean | 0.7070 | 0.016741 | 2736.1 | 4.00 | 1.000 | 0.946 |
| combined_stale_failure_anomaly | fixed_two_mean | 0.7100 | 0.016453 | 889.9 | 2.00 | 0.000 | 0.000 |
| combined_stale_failure_anomaly | full_mean | 0.7070 | 0.016741 | 2736.1 | 4.00 | 0.000 | 0.000 |
| combined_stale_failure_anomaly | proposed_adaptive_robust | 0.7080 | 0.016711 | 2736.1 | 4.00 | 1.000 | 0.946 |
| heterogeneous_2_nodes | adaptive_mean | 0.7620 | 0.013881 | 1639.0 | 2.00 | 0.055 | 0.000 |
| heterogeneous_2_nodes | fixed_two_mean | 0.7620 | 0.013881 | 1639.0 | 2.00 | 0.000 | 0.000 |
| heterogeneous_2_nodes | full_mean | 0.7620 | 0.013881 | 1639.0 | 2.00 | 0.000 | 0.000 |
| heterogeneous_2_nodes | proposed_adaptive_robust | 0.7620 | 0.013881 | 1639.0 | 2.00 | 0.055 | 0.000 |
| high_confidence_wrong | adaptive_mean | 0.7040 | 0.016393 | 2397.8 | 3.06 | 0.000 | 1.000 |
| high_confidence_wrong | fixed_two_mean | 0.7080 | 0.016207 | 1578.6 | 2.00 | 0.000 | 0.000 |
| high_confidence_wrong | full_mean | 0.7040 | 0.016423 | 3427.9 | 4.00 | 0.000 | 0.000 |
| high_confidence_wrong | proposed_adaptive_robust | 0.7030 | 0.016530 | 2397.8 | 3.06 | 0.000 | 1.000 |
| high_snr_stale | adaptive_mean | 0.7230 | 0.016467 | 2408.8 | 3.06 | 0.055 | 1.000 |
| high_snr_stale | fixed_two_mean | 0.7000 | 0.015014 | 1459.8 | 2.00 | 0.000 | 0.000 |
| high_snr_stale | full_mean | 0.7230 | 0.016475 | 3438.8 | 4.00 | 0.000 | 0.000 |
| high_snr_stale | proposed_adaptive_robust | 0.7210 | 0.016281 | 2408.8 | 3.06 | 0.055 | 1.000 |

## Adaptive proposed minus full mean

| Scenario | Regret delta (95% CI) | bit delta |
|---|---:|---:|
| baseline_4_nodes | -0.000168 [-0.000401, 0.000065] | -1030.0 |
| best_node_failure | 0.000033 [-0.000055, 0.000121] | 0.0 |
| best_single_node | 0.000000 [0.000000, 0.000000] | 0.0 |
| combined_stale_failure_anomaly | -0.000030 [-0.000088, 0.000028] | 0.0 |
| heterogeneous_2_nodes | 0.000000 [0.000000, 0.000000] | 0.0 |
| high_confidence_wrong | 0.000107 [-0.000403, 0.000617] | -1030.0 |
| high_snr_stale | -0.000194 [-0.000914, 0.000525] | -1030.0 |

## Robust fusion minus adaptive mean

Both methods use identical adaptive scheduling and bits.

| Scenario | Regret delta (95% CI) |
|---|---:|
| baseline_4_nodes | -0.000168 [-0.000401, 0.000065] |
| best_node_failure | 0.000033 [-0.000055, 0.000121] |
| best_single_node | 0.000000 [0.000000, 0.000000] |
| combined_stale_failure_anomaly | -0.000030 [-0.000088, 0.000028] |
| heterogeneous_2_nodes | 0.000000 [0.000000, 0.000000] |
| high_confidence_wrong | 0.000136 [-0.000356, 0.000629] |
| high_snr_stale | -0.000186 [-0.000906, 0.000535] |

This remains semi-synthetic evidence and does not establish real spatial diversity.
