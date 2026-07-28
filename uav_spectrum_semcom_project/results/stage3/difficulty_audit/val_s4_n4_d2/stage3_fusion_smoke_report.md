# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

60 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.7300 [0.7235, 0.7365] | 0.016799 [0.016562, 0.017036] | 0.5100 | 3345.0 |
| mean | 0.7733 [0.7537, 0.7929] | 0.012877 [0.011040, 0.014715] | 0.6100 | 3345.0 |
| or | 0.7733 [0.7537, 0.7929] | 0.012874 [0.011034, 0.014715] | 0.6133 | 3345.0 |
| quality_weighted | 0.7767 [0.7522, 0.8011] | 0.012618 [0.010457, 0.014780] | 0.6167 | 3345.0 |
| snr_weighted | 0.7767 [0.7522, 0.8011] | 0.012621 [0.010464, 0.014779] | 0.6133 | 3345.0 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
