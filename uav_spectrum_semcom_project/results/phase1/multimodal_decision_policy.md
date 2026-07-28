# Multimodal Decision Policy Simulation

## Setup

- Visual dataset: VisDrone2019-DET-val, 548 frames.
- Spectrum source: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\resource_optimization_simulation.json`
- BER: 0.0001
- Policies: summary_only, semantic_only, roi_always, lowres_always, jpeg_always, spectrum_rule, oracle_policy

## Action space

| Action | Meaning |
|---|---|
| summary_only | transmit only a compact state summary |
| semantic_only | transmit visual semantic boxes only |
| semantic_plus_roi | transmit visual semantics plus estimated ROI patches |
| lowres_plus_semantic | transmit visual semantics plus low-resolution image |
| jpeg_full | transmit full JPEG image |

## Representative results at packet_loss=0.2

| Spectrum | Policy | bits/frame | Utility | Semantic score | Priority detail | Fail rate | Main action ratios |
|---|---|---:|---:|---:|---:|---:|---|
| random | oracle_policy | 5295.4 | 1.1390 | 0.6279 | 0.0025 | 0.3996 | semantic_only:0.99 |
| random | semantic_only | 5305.6 | 1.1364 | 0.6317 | 0.0000 | 0.3522 | semantic_only:1.00 |
| random | spectrum_rule | 5278.8 | 1.0599 | 0.5892 | 0.0000 | 0.3814 | summary_only:0.05, semantic_only:0.95 |
| random | summary_only | 177.0 | 0.2479 | 0.1377 | 0.0000 | 0.0912 | summary_only:1.00 |
| random | lowres_always | 52727.9 | 0.0991 | 0.0253 | 0.0243 | 0.9745 | lowres_plus_semantic:1.00 |
| random | roi_always | 161836.6 | 0.0459 | 0.0176 | 0.0102 | 0.9781 | semantic_plus_roi:1.00 |
| random | jpeg_always | 1185557.3 | -0.1423 | 0.0000 | 0.0000 | 1.0000 | jpeg_full:1.00 |
| semantic_hard_rep3 | oracle_policy | 5511.7 | 1.4241 | 0.7774 | 0.0074 | 0.2117 | semantic_only:0.98, semantic_plus_roi:0.02 |
| semantic_hard_rep3 | spectrum_rule | 5445.9 | 1.4182 | 0.7798 | 0.0074 | 0.2354 | semantic_only:0.99, semantic_plus_roi:0.01 |
| semantic_hard_rep3 | semantic_only | 5305.6 | 1.4102 | 0.7838 | 0.0000 | 0.1934 | semantic_only:1.00 |
| semantic_hard_rep3 | lowres_always | 52727.9 | 0.5367 | 0.1283 | 0.1266 | 0.8522 | lowres_plus_semantic:1.00 |
| semantic_hard_rep3 | summary_only | 177.0 | 0.2586 | 0.1437 | 0.0000 | 0.0274 | summary_only:1.00 |
| semantic_hard_rep3 | roi_always | 161836.6 | 0.1806 | 0.0510 | 0.0383 | 0.9507 | semantic_plus_roi:1.00 |
| semantic_hard_rep3 | jpeg_always | 1185557.3 | -0.1423 | 0.0000 | 0.0000 | 1.0000 | jpeg_full:1.00 |
| spectrogram8_partial | oracle_policy | 5480.6 | 1.3534 | 0.7404 | 0.0059 | 0.2664 | semantic_only:0.98, semantic_plus_roi:0.02 |
| spectrogram8_partial | semantic_only | 5305.6 | 1.3435 | 0.7467 | 0.0000 | 0.2591 | semantic_only:1.00 |
| spectrogram8_partial | spectrum_rule | 5305.6 | 1.3435 | 0.7467 | 0.0000 | 0.2755 | semantic_only:1.00 |
| spectrogram8_partial | lowres_always | 52727.9 | 0.3651 | 0.0880 | 0.0864 | 0.9069 | lowres_plus_semantic:1.00 |
| spectrogram8_partial | summary_only | 177.0 | 0.2562 | 0.1424 | 0.0000 | 0.0511 | summary_only:1.00 |
| spectrogram8_partial | roi_always | 161836.6 | 0.1284 | 0.0383 | 0.0270 | 0.9489 | semantic_plus_roi:1.00 |
| spectrogram8_partial | jpeg_always | 1185557.3 | -0.1423 | 0.0000 | 0.0000 | 1.0000 | jpeg_full:1.00 |

## Link-pressure sensitivity under semantic_hard_rep3 spectrum reporting

| Packet loss | Policy | bits/frame | Utility | Priority detail | Fail rate | Action mix |
|---:|---|---:|---:|---:|---:|---|
| 0.00 | oracle_policy | 50341.6 | 3.8662 | 0.9028 | 0.1004 | semantic_plus_roi:0.10, lowres_plus_semantic:0.90 |
| 0.00 | semantic_only | 5305.6 | 1.7804 | 0.0000 | 0.0055 | semantic_only:1.00 |
| 0.00 | spectrum_rule | 48748.5 | 3.7905 | 0.9028 | 0.0821 | semantic_only:0.05, semantic_plus_roi:0.07, lowres_plus_semantic:0.87 |
| 0.05 | oracle_policy | 45888.9 | 2.3665 | 0.5339 | 0.4526 | semantic_only:0.04, semantic_plus_roi:0.10, lowres_plus_semantic:0.86 |
| 0.05 | semantic_only | 5305.6 | 1.6785 | 0.0000 | 0.0602 | semantic_only:1.00 |
| 0.05 | spectrum_rule | 44761.9 | 2.3317 | 0.5339 | 0.3777 | semantic_only:0.09, semantic_plus_roi:0.07, lowres_plus_semantic:0.84 |
| 0.10 | oracle_policy | 16487.8 | 1.7292 | 0.1741 | 0.2774 | semantic_only:0.61, semantic_plus_roi:0.08, lowres_plus_semantic:0.31 |
| 0.10 | semantic_only | 5305.6 | 1.5830 | 0.0000 | 0.1131 | semantic_only:1.00 |
| 0.10 | spectrum_rule | 15998.3 | 1.7118 | 0.1741 | 0.2774 | semantic_only:0.64, semantic_plus_roi:0.05, lowres_plus_semantic:0.31 |
| 0.20 | oracle_policy | 5511.7 | 1.4241 | 0.0074 | 0.2117 | semantic_only:0.98, semantic_plus_roi:0.02 |
| 0.20 | semantic_only | 5305.6 | 1.4102 | 0.0000 | 0.1934 | semantic_only:1.00 |
| 0.20 | spectrum_rule | 5445.9 | 1.4182 | 0.0074 | 0.2354 | semantic_only:0.99, semantic_plus_roi:0.01 |

## Interpretation

This script formalizes the decision layer before reinforcement learning. The rule policy uses spectrum clean rate and visual priority to choose between semantic-only, ROI, low-resolution image, and full JPEG transmission. The sensitivity table shows that the rule policy transmits more image detail when the link is clean, but falls back to semantic-only as packet loss grows. The oracle policy is not deployable; it selects the action with the highest utility under the simulated channel and is used as an upper-bound reference.

Trade-off plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multimodal_decision_policy_tradeoff.png`
CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multimodal_decision_policy.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multimodal_decision_policy.json`
