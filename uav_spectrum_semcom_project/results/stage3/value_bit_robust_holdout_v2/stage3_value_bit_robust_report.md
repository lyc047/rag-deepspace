# Stage 3 Value/Bit Node Selection and Robust Fusion

The policy was selected only on validation sources beginning at offset 2000 and evaluated once on fresh test sources beginning at offset 3000.
Scheduling uses frozen validation node priors, declared link reliability and age, and expected transmitted bits. It does not inspect current occupancy or test truth.

## Frozen policy

Selected nodes: 2; disagreement scale: 0.02; maximum weight ratio: 1.5.
Validation node correctness priors: [0.525, 0.5125, 0.5625, 0.6625].
Validation expected bit/node: [1076.8, 803.8, 734.8, 890.6].

## Fresh holdout results

| Scenario | Method | Clean rate | Regret | Oracle-equivalent | bit/decision |
|---|---|---:|---:|---:|---:|
| baseline_4_nodes | best_prior_single | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| baseline_4_nodes | full_mean | 0.7220 | 0.009366 | 0.5920 | 3297.9 |
| baseline_4_nodes | full_quality | 0.7220 | 0.009321 | 0.5920 | 3297.9 |
| baseline_4_nodes | full_robust | 0.7220 | 0.009427 | 0.5890 | 3297.9 |
| baseline_4_nodes | proposed_value_bit_robust | 0.7210 | 0.009642 | 0.5930 | 1486.8 |
| baseline_4_nodes | selected_mean | 0.7210 | 0.009710 | 0.5920 | 1486.8 |
| best_node_failure | best_prior_single | 0.6400 | 0.015338 | 0.3850 | 0.0 |
| best_node_failure | full_mean | 0.6650 | 0.012116 | 0.4720 | 2486.9 |
| best_node_failure | full_quality | 0.6670 | 0.012085 | 0.4750 | 2486.9 |
| best_node_failure | full_robust | 0.6670 | 0.012085 | 0.4750 | 2486.9 |
| best_node_failure | proposed_value_bit_robust | 0.6600 | 0.013057 | 0.4650 | 675.8 |
| best_node_failure | selected_mean | 0.6600 | 0.013057 | 0.4650 | 675.8 |
| best_single_node | best_prior_single | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| best_single_node | full_mean | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| best_single_node | full_quality | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| best_single_node | full_robust | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| best_single_node | proposed_value_bit_robust | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| best_single_node | selected_mean | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| combined_stale_failure_anomaly | best_prior_single | 0.6400 | 0.015338 | 0.3850 | 0.0 |
| combined_stale_failure_anomaly | full_mean | 0.6410 | 0.013529 | 0.4360 | 2620.5 |
| combined_stale_failure_anomaly | full_quality | 0.6440 | 0.013403 | 0.4370 | 2620.5 |
| combined_stale_failure_anomaly | full_robust | 0.6430 | 0.013450 | 0.4360 | 2620.5 |
| combined_stale_failure_anomaly | proposed_value_bit_robust | 0.6410 | 0.013701 | 0.4320 | 808.0 |
| combined_stale_failure_anomaly | selected_mean | 0.6410 | 0.013701 | 0.4320 | 808.0 |
| heterogeneous_2_nodes | best_prior_single | 0.7230 | 0.010188 | 0.5810 | 811.0 |
| heterogeneous_2_nodes | full_mean | 0.7250 | 0.009828 | 0.5840 | 1528.7 |
| heterogeneous_2_nodes | full_quality | 0.7240 | 0.009890 | 0.5830 | 1528.7 |
| heterogeneous_2_nodes | full_robust | 0.7240 | 0.009890 | 0.5830 | 1528.7 |
| heterogeneous_2_nodes | proposed_value_bit_robust | 0.7240 | 0.009890 | 0.5830 | 1528.7 |
| heterogeneous_2_nodes | selected_mean | 0.7250 | 0.009828 | 0.5840 | 1528.7 |
| high_confidence_wrong | best_prior_single | 0.6610 | 0.013096 | 0.4600 | 811.0 |
| high_confidence_wrong | full_mean | 0.6600 | 0.011991 | 0.4780 | 3297.9 |
| high_confidence_wrong | full_quality | 0.6610 | 0.012114 | 0.4800 | 3297.9 |
| high_confidence_wrong | full_robust | 0.6630 | 0.012119 | 0.4850 | 3297.9 |
| high_confidence_wrong | proposed_value_bit_robust | 0.6630 | 0.012354 | 0.4790 | 1486.8 |
| high_confidence_wrong | selected_mean | 0.6620 | 0.012378 | 0.4790 | 1486.8 |
| high_snr_stale | best_prior_single | 0.6600 | 0.013057 | 0.4650 | 675.8 |
| high_snr_stale | full_mean | 0.6710 | 0.010771 | 0.4800 | 3294.9 |
| high_snr_stale | full_quality | 0.6690 | 0.010795 | 0.4790 | 3294.9 |
| high_snr_stale | full_robust | 0.6710 | 0.010765 | 0.4800 | 3294.9 |
| high_snr_stale | proposed_value_bit_robust | 0.6660 | 0.012231 | 0.4750 | 1393.5 |
| high_snr_stale | selected_mean | 0.6640 | 0.012261 | 0.4720 | 1393.5 |

## Proposed minus full-mean paired differences

| Scenario | Regret delta (95% CI) | bit delta |
|---|---:|---:|
| baseline_4_nodes | 0.000276 [-0.000342, 0.000894] | -1811.1 |
| best_node_failure | 0.000941 [0.000565, 0.001317] | -1811.1 |
| best_single_node | 0.000000 [0.000000, 0.000000] | 0.0 |
| combined_stale_failure_anomaly | 0.000172 [-0.000113, 0.000457] | -1812.5 |
| heterogeneous_2_nodes | 0.000062 [-0.000050, 0.000173] | 0.0 |
| high_confidence_wrong | 0.000363 [-0.000130, 0.000856] | -1811.1 |
| high_snr_stale | 0.001460 [0.000273, 0.002646] | -1901.4 |

## Robust weighting minus selected-mean differences

Both methods use the same selected nodes and identical bits.

| Scenario | Regret delta (95% CI) |
|---|---:|
| baseline_4_nodes | -0.000068 [-0.000318, 0.000183] |
| best_node_failure | 0.000000 [0.000000, 0.000000] |
| best_single_node | 0.000000 [0.000000, 0.000000] |
| combined_stale_failure_anomaly | 0.000000 [0.000000, 0.000000] |
| heterogeneous_2_nodes | 0.000062 [-0.000050, 0.000173] |
| high_confidence_wrong | -0.000024 [-0.000361, 0.000313] |
| high_snr_stale | -0.000031 [-0.000268, 0.000206] |

## Claim boundary

This remains semi-synthetic same-scene evidence. Lower bits with non-inferior regret supports a scheduling/efficiency claim, not yet a real spatial-diversity claim.
