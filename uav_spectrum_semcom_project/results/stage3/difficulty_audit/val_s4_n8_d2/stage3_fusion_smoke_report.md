# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

60 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.7833 [0.7833, 0.7833] | 0.022176 [0.022122, 0.022229] | 0.5367 | 3345.0 |
| mean | 0.8267 [0.7982, 0.8551] | 0.017129 [0.013707, 0.020552] | 0.6333 | 3345.0 |
| or | 0.8267 [0.7982, 0.8551] | 0.017129 [0.013707, 0.020552] | 0.6333 | 3345.0 |
| quality_weighted | 0.8233 [0.8012, 0.8455] | 0.017302 [0.014107, 0.020496] | 0.6300 | 3345.0 |
| snr_weighted | 0.8233 [0.8012, 0.8455] | 0.017302 [0.014107, 0.020496] | 0.6300 | 3345.0 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
