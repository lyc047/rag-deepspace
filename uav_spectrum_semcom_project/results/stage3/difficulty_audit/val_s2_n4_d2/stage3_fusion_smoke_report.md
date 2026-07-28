# Stage 3 Correlated Multi-UAV Fusion Smoke Report

## Protocol

60 underlying RadDet scenes are each observed by 4 controlled node views. All fusion methods receive exactly the same successfully delivered reports and therefore have identical transmitted-bit cost.

This is same-scene semi-synthetic evidence, not synchronized multi-SDR or real UAV evidence.

## Results

| Fusion | Clean rate (95% CI) | Regret (95% CI) | Oracle-equivalent | bit/decision |
|---|---:|---:|---:|---:|
| majority | 0.8867 [0.8801, 0.8932] | 0.007873 [0.006980, 0.008766] | 0.6300 | 3262.2 |
| mean | 0.8833 [0.8654, 0.9012] | 0.008866 [0.008403, 0.009328] | 0.7500 | 3262.2 |
| or | 0.8900 [0.8733, 0.9067] | 0.008702 [0.008256, 0.009148] | 0.7567 | 3262.2 |
| quality_weighted | 0.8833 [0.8654, 0.9012] | 0.008860 [0.008390, 0.009331] | 0.7500 | 3262.2 |
| snr_weighted | 0.8833 [0.8654, 0.9012] | 0.008860 [0.008390, 0.009331] | 0.7500 | 3262.2 |

## Claim boundary

- The node views share one underlying scene but are generated from spectrogram-domain controlled perturbations.
- Quality weights use declared sensing SNR, reporting reliability and age; they never use ground-truth occupancy.
- A learned fusion model is not yet evaluated.
- H3 remains preliminary until record-replay or synchronized multi-receiver evidence is available.
