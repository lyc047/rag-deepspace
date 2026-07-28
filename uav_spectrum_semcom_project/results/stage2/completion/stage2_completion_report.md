# Stage 2 Completion: Strong PSD Baseline and Full-Split Robustness

## Validation-only PSD selection

Selected PSD: `{"threshold_sigma": 3.0, "bins_per_channel": 8, "quantization_bits": 1, "mode": "cfar", "clean_resource_rate": 0.708, "mean_occupancy_regret": 0.013996832072734833, "mean_selected_occupancy": 0.01806991919875145, "oracle_equivalent_rate": 0.462}`. Candidate resolution/reducer choices were selected on validation data only; all task-link rows below use the frozen choice.

## Full frozen-test results

| Condition | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Latency |
|---|---|---:|---:|---:|---:|---:|
| awgn_3dB_250ms | hard_semantic | 0.3820 | 0.8405 [0.8374, 0.8436] | 0.006062 [0.005831, 0.006294] | 4335.9 | 2.82 ms |
| awgn_3dB_250ms | selected_strong_psd | n/a | 0.7048 [0.7046, 0.7050] | 0.015052 [0.014984, 0.015121] | 3598.2 | 2.63 ms |
| burst_6dB_25ms | hard_semantic | 0.4827 | 0.9257 [0.9246, 0.9268] | 0.000849 [0.000766, 0.000932] | 2745.2 | 1.60 ms |
| burst_6dB_25ms | selected_strong_psd | n/a | 0.7055 [0.7053, 0.7058] | 0.014994 [0.014966, 0.015021] | 2357.8 | 1.47 ms |
| rayleigh_6dB_250ms | hard_semantic | 0.4454 | 0.8919 [0.8900, 0.8938] | 0.002876 [0.002823, 0.002930] | 3788.3 | 2.66 ms |
| rayleigh_6dB_250ms | selected_strong_psd | n/a | 0.7060 [0.7051, 0.7069] | 0.014932 [0.014902, 0.014962] | 3237.8 | 2.51 ms |
| rician_6dB_250ms | hard_semantic | 0.4710 | 0.9139 [0.9116, 0.9161] | 0.001557 [0.001440, 0.001674] | 3367.5 | 2.39 ms |
| rician_6dB_250ms | selected_strong_psd | n/a | 0.7065 [0.7052, 0.7077] | 0.014961 [0.014873, 0.015049] | 2874.2 | 2.26 ms |

## Claim boundary

- PSD remains normalized spectrogram intensity, not calibrated dBm receiver power.
- Rayleigh/Rician use ideal receiver equalization; burst uses a declared Gilbert--Elliott packet-attempt model.
- Four unrelated frames form a pressure decision, so these are H1/H4 results rather than same-scene multi-UAV H3 evidence.
- RadDet remains synthetic radar data.
