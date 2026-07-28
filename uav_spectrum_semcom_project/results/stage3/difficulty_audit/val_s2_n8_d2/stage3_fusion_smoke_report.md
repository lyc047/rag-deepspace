# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

60 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.9200 [0.9135, 0.9265] | 0.008909 [0.007061, 0.010756] | 0.7867 | 3262.2 |
| mean | 0.9167 [0.8988, 0.9346] | 0.007304 [0.005106, 0.009501] | 0.8400 | 3262.2 |
| or | 0.9167 [0.8988, 0.9346] | 0.007304 [0.005106, 0.009501] | 0.8400 | 3262.2 |
| quality_weighted | 0.9167 [0.8988, 0.9346] | 0.007304 [0.005106, 0.009501] | 0.8400 | 3262.2 |
| snr_weighted | 0.9167 [0.8988, 0.9346] | 0.007304 [0.005106, 0.009501] | 0.8400 | 3262.2 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
