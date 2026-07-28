# Multi-UAV Spectrum Pressure Sweep

## Setup

- Packet loss: 0.2
- BER: 0.0001
- Candidate channels: 4
- Demand: contiguous block of 2 channel(s)
- Switch cost: 0.005
- Decision steps per trial: 1000
- Trials: 5

## Clean resource selection rate

| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9142 | 0.9138 | 0.9800 | 0.9974 | 0.9968 | 0.9992 |
| 2 | 0.8370 | 0.8392 | 0.9586 | 0.9828 | 0.9820 | 0.9882 |
| 4 | 0.6846 | 0.6868 | 0.8844 | 0.9284 | 0.9296 | 0.9396 |
| 8 | 0.4306 | 0.4314 | 0.6894 | 0.7316 | 0.7300 | 0.7554 |

## Mean occupancy regret vs oracle

| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0048 | 0.0050 | 0.0012 | 0.0001 | 0.0002 |
| 2 | 0.0087 | 0.0093 | 0.0021 | 0.0006 | 0.0008 |
| 4 | 0.0149 | 0.0166 | 0.0040 | 0.0013 | 0.0013 |
| 8 | 0.0248 | 0.0277 | 0.0055 | 0.0028 | 0.0029 |

## Net utility after switch cost

| UAVs | Random | Fixed | Semantic hard | Semantic hard rep3 | Semantic soft rep3 | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9918 | 0.9949 | 0.9954 | 0.9965 | 0.9964 | 0.9967 |
| 2 | 0.9870 | 0.9898 | 0.9937 | 0.9952 | 0.9951 | 0.9958 |
| 4 | 0.9778 | 0.9795 | 0.9888 | 0.9915 | 0.9915 | 0.9929 |
| 8 | 0.9587 | 0.9591 | 0.9780 | 0.9808 | 0.9807 | 0.9836 |

## Interpretation

At 8 simultaneous UAV observations, semantic hard repetition improves clean resource selection by 0.3010 absolute over random while remaining 3817.9x smaller than 12-bit I/Q forwarding.
This sweep is intended to show that the resource-optimization value of spectrum semantics becomes clearer as the spectrum-sharing pressure increases.

Clean-rate plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multi_uav_pressure_clean_rate.png`
Regret plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multi_uav_pressure_regret.png`
CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multi_uav_pressure_sweep.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multi_uav_pressure_sweep.json`
