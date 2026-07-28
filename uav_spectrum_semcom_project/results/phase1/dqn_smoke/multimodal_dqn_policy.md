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
| random | semantic_only | 6069.8 | 1.0980 | 0.6104 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | spectrum_rule | 6069.8 | 1.0980 | 0.6104 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | dqn | 6069.8 | 1.0980 | 0.6104 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | oracle_policy | 6069.8 | 1.0980 | 0.6104 | 0.0000 | 0.00 | semantic_only:1.00 |
| random | summary_only | 177.0 | 0.2479 | 0.1377 | 0.0000 | 0.00 | summary_only:1.00 |
| semantic_hard_rep3 | semantic_only | 3474.7 | 1.5184 | 0.8438 | 0.0000 | 0.00 | semantic_only:1.00 |
| semantic_hard_rep3 | spectrum_rule | 3474.7 | 1.5184 | 0.8438 | 0.0000 | 0.00 | semantic_only:1.00 |
| semantic_hard_rep3 | oracle_policy | 3474.7 | 1.5184 | 0.8438 | 0.0000 | 0.00 | semantic_only:1.00 |
| semantic_hard_rep3 | dqn | 5006.5 | 1.4951 | 0.8039 | 0.0299 | 5.11 | semantic_only:0.93, lowres_plus_semantic:0.07 |
| semantic_hard_rep3 | summary_only | 177.0 | 0.2586 | 0.1437 | 0.0000 | 0.00 | summary_only:1.00 |
| spectrogram8_partial | semantic_only | 4789.3 | 1.3833 | 0.7688 | 0.0000 | 0.00 | semantic_only:1.00 |
| spectrogram8_partial | spectrum_rule | 4789.3 | 1.3833 | 0.7688 | 0.0000 | 0.00 | semantic_only:1.00 |
| spectrogram8_partial | oracle_policy | 4789.3 | 1.3833 | 0.7688 | 0.0000 | 0.00 | semantic_only:1.00 |
| spectrogram8_partial | dqn | 6373.8 | 1.3653 | 0.7300 | 0.0216 | 3.13 | semantic_only:0.92, lowres_plus_semantic:0.08 |
| spectrogram8_partial | summary_only | 177.0 | 0.2562 | 0.1424 | 0.0000 | 0.00 | summary_only:1.00 |

## Link-pressure sensitivity under semantic_hard_rep3

| Packet loss | Policy | bits/frame | Utility | Priority detail | Action mix |
|---:|---|---:|---:|---:|---|
| 0.00 | dqn | 16808.8 | 2.7119 | 0.3726 | semantic_only:0.60, lowres_plus_semantic:0.40 |
| 0.00 | oracle_policy | 84434.5 | 3.7325 | 0.8487 | lowres_plus_semantic:1.00 |
| 0.00 | semantic_only | 5191.4 | 1.7808 | 0.0000 | semantic_only:1.00 |
| 0.00 | spectrum_rule | 84434.5 | 3.7325 | 0.8487 | lowres_plus_semantic:1.00 |
| 0.05 | dqn | 30445.7 | 2.5944 | 0.5818 | semantic_only:0.10, lowres_plus_semantic:0.90 |
| 0.05 | oracle_policy | 35071.5 | 2.6237 | 0.6513 | lowres_plus_semantic:1.00 |
| 0.05 | semantic_only | 3254.6 | 1.7188 | 0.0000 | semantic_only:1.00 |
| 0.05 | spectrum_rule | 28203.5 | 2.5531 | 0.6513 | semantic_only:0.20, lowres_plus_semantic:0.80 |
| 0.10 | dqn | 20426.8 | 1.8365 | 0.3239 | semantic_only:0.38, lowres_plus_semantic:0.62 |
| 0.10 | oracle_policy | 12871.8 | 1.9755 | 0.2817 | semantic_only:0.54, semantic_plus_roi:0.15, lowres_plus_semantic:0.31 |
| 0.10 | semantic_only | 2535.2 | 1.6855 | 0.0000 | semantic_only:1.00 |
| 0.10 | spectrum_rule | 12418.7 | 1.8918 | 0.2817 | semantic_only:0.62, semantic_plus_roi:0.08, lowres_plus_semantic:0.31 |
| 0.20 | dqn | 5006.5 | 1.4951 | 0.0299 | semantic_only:0.93, lowres_plus_semantic:0.07 |
| 0.20 | oracle_policy | 3474.7 | 1.5184 | 0.0000 | semantic_only:1.00 |
| 0.20 | semantic_only | 3474.7 | 1.5184 | 0.0000 | semantic_only:1.00 |
| 0.20 | spectrum_rule | 3474.7 | 1.5184 | 0.0000 | semantic_only:1.00 |

## Interpretation

DQN is not used here to replace the spectrum detector. It optimizes the cross-layer decision: whether the UAV should send only visual semantics, add ROI/detail, or fall back to a compact summary under poor links. If DQN approaches the oracle and exceeds spectrum_rule on utility at similar or lower payload, it supports the next research claim: semantic communication should jointly decide content granularity and radio resource state, not only compress images or classify spectrum.
