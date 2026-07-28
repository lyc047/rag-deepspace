# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

200 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.7040 [0.6983, 0.7097] | 0.014402 [0.014004, 0.014800] | 0.4970 | 3347.2 |
| mean | 0.7920 [0.7829, 0.8011] | 0.009309 [0.008504, 0.010114] | 0.6270 | 3347.2 |
| or | 0.7910 [0.7832, 0.7988] | 0.009358 [0.008646, 0.010070] | 0.6260 | 3347.2 |
| quality_weighted | 0.7910 [0.7815, 0.8005] | 0.009306 [0.008673, 0.009939] | 0.6270 | 3347.2 |
| snr_weighted | 0.7900 [0.7818, 0.7982] | 0.009383 [0.008721, 0.010044] | 0.6260 | 3347.2 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
