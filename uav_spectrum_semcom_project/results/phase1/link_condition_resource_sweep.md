# Link-Condition Resource Optimization Sweep

## Setup

- UAV observations per decision: 4
- Decision steps/trial: 600
- Trials: 3
- Large-payload partial recovery: FEC overhead 1.25, minimum recovery fraction 0.25

## Representative results at BER=0.0001

| Packet loss | Random clean | Semantic hard rep3 clean | Spectrogram partial clean | I/Q partial clean | Semantic hard rep3 net utility |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6928 | 0.9394 | 0.9233 | 0.9194 | 0.9918 |
| 0.03 | 0.7094 | 0.9339 | 0.9072 | 0.9144 | 0.9916 |
| 0.05 | 0.6933 | 0.9333 | 0.9067 | 0.8994 | 0.9917 |
| 0.1 | 0.6928 | 0.9267 | 0.8989 | 0.8850 | 0.9916 |
| 0.2 | 0.6856 | 0.9217 | 0.8644 | 0.8583 | 0.9904 |
| 0.3 | 0.7094 | 0.9328 | 0.8617 | 0.8561 | 0.9916 |

## Interpretation

This sweep evaluates whether the spectrum resource decision remains robust under changing packet loss and BER. The partial large-payload baselines are intentionally more favorable than the earlier all-or-nothing gate, so they provide a fairer comparison point for semantic reporting.

Clean-rate plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\link_condition_clean_rate.png`
Net-utility plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\link_condition_net_utility.png`
CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\link_condition_resource_sweep.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\link_condition_resource_sweep.json`
