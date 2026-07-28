# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

50 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.9480 [0.9323, 0.9637] | 0.002260 [0.001866, 0.002655] | 0.8160 | 3220.3 |
| mean | 1.0000 [1.0000, 1.0000] | 0.000103 [0.000004, 0.000203] | 0.9840 | 3220.3 |
| or | 1.0000 [1.0000, 1.0000] | 0.000103 [0.000004, 0.000203] | 0.9840 | 3220.3 |
| quality_weighted | 1.0000 [1.0000, 1.0000] | 0.000103 [0.000004, 0.000203] | 0.9840 | 3220.3 |
| snr_weighted | 1.0000 [1.0000, 1.0000] | 0.000103 [0.000004, 0.000203] | 0.9840 | 3220.3 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
