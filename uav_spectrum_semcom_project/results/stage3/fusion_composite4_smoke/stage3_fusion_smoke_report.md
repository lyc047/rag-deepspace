# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

50 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.6720 [0.6563, 0.6877] | 0.013705 [0.012987, 0.014423] | 0.4960 | 3298.9 |
| mean | 0.8280 [0.8080, 0.8480] | 0.006227 [0.004778, 0.007675] | 0.6960 | 3298.9 |
| or | 0.8280 [0.8080, 0.8480] | 0.006242 [0.004800, 0.007683] | 0.6920 | 3298.9 |
| quality_weighted | 0.8320 [0.8085, 0.8555] | 0.006033 [0.004573, 0.007494] | 0.7040 | 3298.9 |
| snr_weighted | 0.8320 [0.8085, 0.8555] | 0.006085 [0.004545, 0.007625] | 0.7000 | 3298.9 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
