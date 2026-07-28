# Stage 3 LoRaIQ Real Four-Receiver Pilot

## Protocol

Two UAV-transmitter LoRa frames are used: one LoS and one NLoS event. Each event has four aligned real RRH IQ recordings.
Each node sends one 241-bit application semantic containing four 8-bit occupancy probabilities and fixed metadata. The shared fair link adds header, CRC, FEC, padding, and retransmissions.
The off-frame Welch detector uses a fixed 1.5 dB excess threshold for this pilot.

## Local receiver evidence

| Transmission | Scene | RRH | annotated SNR | excess dB by channel | local F1 |
|---:|---|---:|---:|---|---:|
| 27429 | drone_los | 1 | 44.10 | [24.46, 23.35, 23.90, 24.46] | 0.6667 |
| 27429 | drone_los | 2 | 14.73 | [1.78, 13.47, 14.65, 1.24] | 0.8000 |
| 27429 | drone_los | 3 | 15.34 | [1.76, 14.26, 14.26, 1.37] | 0.8000 |
| 27429 | drone_los | 4 | 5.16 | [0.49, 6.33, 7.50, 0.59] | 1.0000 |
| 23039 | drone_nlos | 1 | -20.21 | [0.08, 0.28, 0.33, 0.06] | 0.0000 |
| 23039 | drone_nlos | 2 | -13.69 | [0.07, 0.16, 0.11, 0.01] | 0.0000 |
| 23039 | drone_nlos | 3 | -0.18 | [0.29, 2.89, 2.35, 0.20] | 1.0000 |
| 23039 | drone_nlos | 4 | -16.55 | [0.02, 0.22, 0.16, -0.01] | 0.0000 |

## End-to-end results

The confidence intervals below reflect reporting-link randomness only; two sensing events are not an independent statistical test set.

| Policy | F1 | Brier | Clean | Regret | uplink bit | scheduler bit | latency | reports |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_single | 0.8889 | 0.0948 | 1.0000 | 0.003000 | 630.0 | 0.0 | 1.36 ms | 1.00 |
| full_mean | 0.5417 | 0.1848 | 1.0000 | 0.003000 | 2803.5 | 0.0 | 1.98 ms | 3.75 |
| full_snr_weighted | 0.8500 | 0.1124 | 1.0000 | 0.003000 | 2803.5 | 0.0 | 1.98 ms | 3.75 |
| full_robust_quality | 0.6161 | 0.1380 | 1.0000 | 0.003000 | 2803.5 | 0.0 | 1.98 ms | 3.75 |
| proposed_adaptive_value_bit | 0.7500 | 0.1233 | 1.0000 | 0.003000 | 1890.0 | 96.0 | 4.05 ms | 3.00 |

## Interpretation boundary

This run establishes that aligned real multi-receiver IQ can pass through local sensing, quantized semantic reporting, the fair digital link, fusion, and resource selection.
The resource task is saturated because both LoRa frames occupy the same central half-band; clean rate and regret cannot distinguish the policies.
The detector threshold was inspected on these same two pilot events, the event count is two, and the receiving nodes are fixed RRHs rather than UAVs. These results are diagnostic and do not establish H3.
