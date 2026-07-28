# Weak Visual Semantics in the Closed Loop

## Purpose

This experiment keeps the current lightweight visual ROI-mask model and plugs its predicted visual semantics into the spectrum-aware DQN decision loop. It is a hardware-safe baseline for the current 2GB-GPU machine; stronger YOLO-style detectors can replace only the visual front-end later.

## Representative closed-loop result

Spectrum scheme: `semantic_hard_rep3`, packet loss: 0.2.

| Visual source | Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gt_annotations | oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 1.0000 | 70.8 | semantic_only:0.98, semantic_plus_roi:0.02 |
| gt_annotations | dqn | 5635.6 | 1.3794 | 0.7740 | 0.0083 | 1.0000 | 70.8 | semantic_only:0.97, semantic_plus_roi:0.03 |
| gt_annotations | spectrum_rule | 5445.9 | 1.3763 | 0.7798 | 0.0070 | 1.0000 | 70.8 | semantic_only:0.99, semantic_plus_roi:0.01 |
| gt_annotations | semantic_only | 5305.6 | 1.3683 | 0.7838 | 0.0000 | 1.0000 | 70.8 | semantic_only:1.00 |
| roi_mask_128_class_agnostic | oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.2308 | 28.3 | semantic_only:0.98, semantic_plus_roi:0.02 |
| roi_mask_128_class_agnostic | dqn | 2489.3 | 0.3466 | 0.2028 | 0.0015 | 0.2308 | 28.3 | semantic_only:0.97, semantic_plus_roi:0.03 |
| roi_mask_128_class_agnostic | semantic_only | 2243.8 | 0.3451 | 0.2045 | 0.0000 | 0.2308 | 28.3 | semantic_only:1.00 |
| roi_mask_128_class_agnostic | spectrum_rule | 2243.8 | 0.3451 | 0.2045 | 0.0000 | 0.2308 | 28.3 | semantic_only:1.00 |
| roi_mask_128_priority_proxy | oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.2308 | 28.3 | semantic_only:0.98, semantic_plus_roi:0.02 |
| roi_mask_128_priority_proxy | dqn | 2445.4 | 0.3469 | 0.2030 | 0.0015 | 0.2308 | 28.3 | semantic_only:0.98, semantic_plus_roi:0.02 |
| roi_mask_128_priority_proxy | spectrum_rule | 2313.2 | 0.3457 | 0.2042 | 0.0005 | 0.2308 | 28.3 | semantic_only:0.99 |
| roi_mask_128_priority_proxy | semantic_only | 2243.8 | 0.3451 | 0.2045 | 0.0000 | 0.2308 | 28.3 | semantic_only:1.00 |

## Interpretation

- `gt_annotations` is the visual-semantic upper bound and should not be claimed as a deployed detector.
- `roi_mask_128_class_agnostic` is the current safest real visual front-end: it predicts ROI boxes but no reliable category, so semantic quality and task utility are limited.
- `roi_mask_128_priority_proxy` treats predicted ROI boxes as priority-like objects. It is not a final model; it estimates what happens if the weak ROI detector is used as a coarse priority trigger.
- The result makes the current bottleneck explicit: the system closed loop works, but visual semantic quality is now the limiting factor.

## Hardware note

The current ROI-mask model is a tiny CNN and is safe for a 2GB MX450 GPU. A 256x256 retraining run completed without memory overflow but did not improve F1, so the 128x128 checkpoint remains the current weak visual baseline.
