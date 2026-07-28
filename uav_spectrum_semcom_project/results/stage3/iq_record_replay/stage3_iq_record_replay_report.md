# Stage 3 Same-Source SigMF I/Q Record Replay

One real 5G NR SigMF recording is sliced into 14 correlated frames and replayed through four independent controlled sensing channels.
This validates the raw-I/Q-to-STFT-to-semantic-report pipeline. It is not independent multi-receiver capture and cannot establish spatial diversity.

## Frozen detector and node sensing

Detector: sigma=7.0, min_cells=16, enhancement=freq_median.

| Node | Sensing SNR | Detection F1 |
|---:|---:|---:|
| 0 | -3.0 dB | 0.0000 |
| 1 | 3.0 dB | 0.0000 |
| 2 | 9.0 dB | 0.0000 |
| 3 | 15.0 dB | 0.3380 |

## Task and communication results

Energy is a 1 W reference RF-airtime estimate, not measured hardware energy.

| Policy | Clean | Regret | Oracle-equivalent | uplink bit | scheduler bit | link latency | RF energy | nodes | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_ack_robust | 0.7143 | 0.002459 | 0.2571 | 2744.0 | 97.4 | 5.04 ms | 1.372 mJ | 3.04 | 1.04 |
| fixed_two_mean | 0.7143 | 0.002459 | 0.2571 | 1973.2 | 0.0 | 1.87 ms | 0.987 mJ | 2.00 | 0.00 |
| full_mean | 0.7143 | 0.002459 | 0.2571 | 3769.0 | 0.0 | 2.66 ms | 1.884 mJ | 4.00 | 0.00 |
| full_robust | 0.7143 | 0.002459 | 0.2571 | 3769.0 | 0.0 | 2.66 ms | 1.884 mJ | 4.00 | 0.00 |

## Claim boundary

All frames originate from one short recording. Detector parameters were previously selected on the same recording, so detection and task numbers are diagnostic only.
The contribution of this run is same-source raw-I/Q replay and explicit fallback latency/energy accounting, not a generalization or real-flight claim.
