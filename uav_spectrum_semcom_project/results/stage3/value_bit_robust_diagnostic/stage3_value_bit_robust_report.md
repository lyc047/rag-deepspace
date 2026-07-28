# Stage 3 Value/Bit Node Selection and Robust Fusion

The policy was selected only on fresh validation sources beginning at offset 2000 and evaluated once on fresh test sources beginning at offset 2000.
Scheduling uses frozen validation node priors, declared link reliability and age, and expected transmitted bits. It does not inspect current occupancy or test truth.

## Frozen policy

Selected nodes: 2; disagreement scale: 0.02; maximum weight ratio: 1.5.
Validation node correctness priors: [0.2, 0.2, 0.45, 0.8].
Validation expected bit/node: [1083.6, 793.8, 668.5, 826.0].

## Fresh holdout results

| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |
|---|---|---:|---:|---:|---:|
| baseline_4_nodes | best_prior_single | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| baseline_4_nodes | full_mean | 0.6500 | 0.008113 | 0.6000 | 3283.0 |
| baseline_4_nodes | full_quality | 0.6500 | 0.008202 | 0.5500 | 3283.0 |
| baseline_4_nodes | full_robust | 0.6500 | 0.008202 | 0.5500 | 3283.0 |
| baseline_4_nodes | proposed_value_bit_robust | 0.7000 | 0.006452 | 0.5500 | 1481.9 |
| baseline_4_nodes | selected_mean | 0.7000 | 0.006452 | 0.5500 | 1481.9 |
| best_node_failure | best_prior_single | 0.7000 | 0.007573 | 0.6500 | 0.0 |
| best_node_failure | full_mean | 0.6500 | 0.009605 | 0.5500 | 2482.2 |
| best_node_failure | full_quality | 0.7000 | 0.007697 | 0.6000 | 2482.2 |
| best_node_failure | full_robust | 0.7000 | 0.007697 | 0.6000 | 2482.2 |
| best_node_failure | proposed_value_bit_robust | 0.7000 | 0.007785 | 0.5500 | 681.1 |
| best_node_failure | selected_mean | 0.7000 | 0.007785 | 0.5500 | 681.1 |
| best_single_node | best_prior_single | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| best_single_node | full_mean | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| best_single_node | full_quality | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| best_single_node | full_robust | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| best_single_node | proposed_value_bit_robust | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| best_single_node | selected_mean | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| combined_stale_failure_anomaly | best_prior_single | 0.7500 | 0.009792 | 0.4500 | 731.5 |
| combined_stale_failure_anomaly | full_mean | 0.7000 | 0.010879 | 0.4000 | 2538.9 |
| combined_stale_failure_anomaly | full_quality | 0.7000 | 0.010879 | 0.4000 | 2538.9 |
| combined_stale_failure_anomaly | full_robust | 0.7000 | 0.010879 | 0.4000 | 2538.9 |
| combined_stale_failure_anomaly | proposed_value_bit_robust | 0.7500 | 0.009792 | 0.4500 | 731.5 |
| combined_stale_failure_anomaly | selected_mean | 0.7500 | 0.009792 | 0.4500 | 731.5 |
| heterogeneous_2_nodes | best_prior_single | 0.7000 | 0.006579 | 0.5500 | 800.8 |
| heterogeneous_2_nodes | full_mean | 0.6500 | 0.008113 | 0.6000 | 1486.1 |
| heterogeneous_2_nodes | full_quality | 0.6500 | 0.008113 | 0.6000 | 1486.1 |
| heterogeneous_2_nodes | full_robust | 0.6500 | 0.008113 | 0.6000 | 1486.1 |
| heterogeneous_2_nodes | proposed_value_bit_robust | 0.6500 | 0.008113 | 0.6000 | 1486.1 |
| heterogeneous_2_nodes | selected_mean | 0.6500 | 0.008113 | 0.6000 | 1486.1 |
| high_confidence_wrong | best_prior_single | 0.6000 | 0.010859 | 0.3500 | 800.8 |
| high_confidence_wrong | full_mean | 0.6000 | 0.010771 | 0.4000 | 3283.0 |
| high_confidence_wrong | full_quality | 0.6000 | 0.010859 | 0.3500 | 3283.0 |
| high_confidence_wrong | full_robust | 0.6000 | 0.010859 | 0.3500 | 3283.0 |
| high_confidence_wrong | proposed_value_bit_robust | 0.6000 | 0.010859 | 0.3500 | 1481.9 |
| high_confidence_wrong | selected_mean | 0.6000 | 0.010859 | 0.3500 | 1481.9 |
| high_snr_stale | best_prior_single | 0.7000 | 0.007785 | 0.5500 | 681.1 |
| high_snr_stale | full_mean | 0.8000 | 0.003059 | 0.7000 | 3213.7 |
| high_snr_stale | full_quality | 0.8000 | 0.003059 | 0.7000 | 3213.7 |
| high_snr_stale | full_robust | 0.8000 | 0.003059 | 0.7000 | 3213.7 |
| high_snr_stale | proposed_value_bit_robust | 0.7500 | 0.004841 | 0.6500 | 1412.6 |
| high_snr_stale | selected_mean | 0.7500 | 0.004841 | 0.6500 | 1412.6 |

## Proposed minus full-mean paired differences

| Scenario | Regret delta (95% CI) | bit delta |
|---|---:|---:|
| baseline_4_nodes | -0.001661 [-0.001661, -0.001661] | -1801.1 |
| best_node_failure | -0.001820 [-0.001820, -0.001820] | -1801.1 |
| best_single_node | 0.000000 [0.000000, 0.000000] | 0.0 |
| combined_stale_failure_anomaly | -0.001087 [-0.001087, -0.001087] | -1807.4 |
| heterogeneous_2_nodes | 0.000000 [0.000000, 0.000000] | 0.0 |
| high_confidence_wrong | 0.000088 [0.000088, 0.000088] | -1801.1 |
| high_snr_stale | 0.001782 [0.001782, 0.001782] | -1801.1 |

## Claim boundary

This remains semi-synthetic same-scene evidence. Lower bits with non-inferior regret supports a scheduling/efficiency claim, not yet a real spatial-diversity claim.
