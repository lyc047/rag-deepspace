# DQN Multimodal Semantic Transmission Policy

## Purpose

This experiment upgrades the previous rule-based decision layer to a lightweight trainable DQN/contextual-bandit policy. The state combines spectrum quality and visual semantic demand; the action selects how much visual information to transmit.

## State and action design

- State: clean-channel rate, packet loss, BER, high-priority visual flag, priority-object density, object density, ROI area, and normalized payload sizes.
- Actions: summary_only, semantic_only, semantic_plus_roi, lowres_plus_semantic, jpeg_full.
- Reward: the same task utility used by the formal decision-layer simulator, so DQN is directly comparable with spectrum_rule and oracle_policy.

## Representative test results

Packet loss shown below: 0.2.

| Spectrum | Policy | bits/frame | Utility | Semantic score | Priority detail | Detail/Mbit | Action mix |
|---|---|---:|---:|---:|---:|---:|---|
| random | semantic_only | 5296.4 | 1.1431 | 0.6354 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | oracle_policy | 5296.4 | 1.1431 | 0.6354 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | dqn | 5395.5 | 1.1416 | 0.6296 | 0.0000 | 1.07 | semantic_only:0.99, semantic_plus_roi:0.01 |
| random | spectrum_rule | 5269.4 | 1.0365 | 0.5762 | 0.0000 | 0.00 | summary_only:0.08, semantic_only:0.92 |
| random | summary_only | 177.0 | 0.2479 | 0.1377 | 0.0000 | 0.00 | summary_only:1.00 |
| semantic_hard_rep3 | oracle_policy | 6617.9 | 1.3699 | 0.7460 | 0.0075 | 2.40 | semantic_only:0.98, semantic_plus_roi:0.02 |
| semantic_hard_rep3 | dqn | 7039.0 | 1.3697 | 0.7352 | 0.0117 | 3.73 | semantic_only:0.95, semantic_plus_roi:0.05 |
| semantic_hard_rep3 | spectrum_rule | 6540.9 | 1.3605 | 0.7487 | 0.0075 | 1.08 | semantic_only:0.99, semantic_plus_roi:0.01 |
| semantic_hard_rep3 | semantic_only | 6414.9 | 1.3540 | 0.7526 | 0.0000 | 0.00 | semantic_only:1.00 |
| semantic_hard_rep3 | summary_only | 177.0 | 0.2586 | 0.1437 | 0.0000 | 0.00 | summary_only:1.00 |
| spectrogram8_partial | dqn | 5706.7 | 1.3313 | 0.7318 | 0.0058 | 0.99 | semantic_only:0.99, semantic_plus_roi:0.01 |
| spectrogram8_partial | oracle_policy | 5706.7 | 1.3313 | 0.7318 | 0.0058 | 0.99 | semantic_only:0.99, semantic_plus_roi:0.01 |
| spectrogram8_partial | semantic_only | 5541.8 | 1.3259 | 0.7370 | 0.0000 | 0.00 | semantic_only:1.00 |
| spectrogram8_partial | spectrum_rule | 5541.8 | 1.3259 | 0.7370 | 0.0000 | 0.00 | semantic_only:1.00 |
| spectrogram8_partial | summary_only | 177.0 | 0.2562 | 0.1424 | 0.0000 | 0.00 | summary_only:1.00 |

## Link-pressure sensitivity under semantic_hard_rep3

| Packet loss | Policy | bits/frame | Utility | Priority detail | Action mix |
|---:|---|---:|---:|---:|---|
| 0.00 | dqn | 47583.5 | 3.8463 | 0.9061 | semantic_plus_roi:0.15, lowres_plus_semantic:0.85 |
| 0.00 | oracle_policy | 47457.4 | 3.8473 | 0.9064 | semantic_plus_roi:0.13, lowres_plus_semantic:0.87 |
| 0.00 | semantic_only | 5135.1 | 1.7810 | 0.0000 | semantic_only:1.00 |
| 0.00 | spectrum_rule | 44928.7 | 3.7214 | 0.9064 | semantic_only:0.09, semantic_plus_roi:0.08, lowres_plus_semantic:0.83 |
| 0.05 | dqn | 44024.3 | 2.4626 | 0.5599 | semantic_only:0.02, semantic_plus_roi:0.16, lowres_plus_semantic:0.81 |
| 0.05 | oracle_policy | 44024.3 | 2.4626 | 0.5599 | semantic_only:0.02, semantic_plus_roi:0.16, lowres_plus_semantic:0.81 |
| 0.05 | semantic_only | 4934.7 | 1.6852 | 0.0000 | semantic_only:1.00 |
| 0.05 | spectrum_rule | 43073.3 | 2.4241 | 0.5599 | semantic_only:0.07, semantic_plus_roi:0.13, lowres_plus_semantic:0.80 |
| 0.10 | dqn | 17584.6 | 1.7352 | 0.1832 | semantic_only:0.58, semantic_plus_roi:0.10, lowres_plus_semantic:0.32 |
| 0.10 | oracle_policy | 17978.4 | 1.7354 | 0.1871 | semantic_only:0.57, semantic_plus_roi:0.10, lowres_plus_semantic:0.33 |
| 0.10 | semantic_only | 5583.1 | 1.5744 | 0.0000 | semantic_only:1.00 |
| 0.10 | spectrum_rule | 17636.0 | 1.7112 | 0.1871 | semantic_only:0.60, semantic_plus_roi:0.07, lowres_plus_semantic:0.33 |
| 0.20 | dqn | 7039.0 | 1.3697 | 0.0117 | semantic_only:0.95, semantic_plus_roi:0.05 |
| 0.20 | oracle_policy | 6617.9 | 1.3699 | 0.0075 | semantic_only:0.98, semantic_plus_roi:0.02 |
| 0.20 | semantic_only | 6414.9 | 1.3540 | 0.0000 | semantic_only:1.00 |
| 0.20 | spectrum_rule | 6540.9 | 1.3605 | 0.0075 | semantic_only:0.99, semantic_plus_roi:0.01 |

## Interpretation

DQN is not used here to replace the spectrum detector. It optimizes the cross-layer decision: whether the UAV should send only visual semantics, add ROI/detail, or fall back to a compact summary under poor links. If DQN approaches the oracle and exceeds spectrum_rule on utility at similar or lower payload, it supports the next research claim: semantic communication should jointly decide content granularity and radio resource state, not only compress images or classify spectrum.
