# Stage 2 Fair Digital Reporting Link

## 1. Scope and claim boundary

This stage replaces the earlier asymmetric gate model with one packetization and digital-link rule shared by hard semantics, quantized soft semantics, compressed spectrograms, uint8 spectrograms, and the theoretical IQ capacity reference.

The current report establishes link-level fairness and validates the scalable abstraction against a waveform chain. Payload delivery ratio is not treated as downstream sensing accuracy. Full H4 evidence requires the next resource-selection adapter to decode received fragments and evaluate occupancy regret/clean rate.

## 2. Frozen fairness protocol

- Modulation: `qpsk`.
- Channel: `awgn`; Eb/N0 sweep: [0.0, 3.0, 6.0, 9.0, 12.0] dB.
- FEC: `hamming74`; maximum retransmissions: 1.
- Packet application payload: 1024 bit; link header: 96 bit; CRC: 16 bit.
- Symbol rate: 1000000 baud; stop-and-wait latency budget: 0.250 s/decision.
- Repeats: 30; uncertainty: sample standard deviation and normal-approximation 95% CI.

All representations use the same packet, CRC, FEC, modulation, ARQ, channel realization policy, and latency budget. Total transmitted bits include link header, CRC, FEC padding, modulation padding, and retransmissions.

## 3. Main link results

| Eb/N0 | Representation | App bits | Delivery ratio (95% CI) | Frame success | Tx bits | Latency (ms) | Goodput (kbit/s) | Budget exhausted |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | hard_semantic | 262 | 0.0000 [0.0000, 0.0000] | 0.000 | 1316.0 | 2.76 | 0.00 | 0.000 |
| 0.0 | quantized_soft_semantic | 327 | 0.0000 [0.0000, 0.0000] | 0.000 | 1540.0 | 2.87 | 0.00 | 0.000 |
| 0.0 | compressed_spectrogram_png | 116541 | 0.0000 [0.0000, 0.0000] | 0.000 | 242536.0 | 249.37 | 0.00 | 1.000 |
| 0.0 | spectrogram_uint8 | 131072 | 0.0000 [0.0000, 0.0000] | 0.000 | 242536.0 | 249.37 | 0.00 | 1.000 |
| 0.0 | iq_int12_complex | 24000000 | 0.0000 [0.0000, 0.0000] | 0.000 | 242536.0 | 249.37 | 0.00 | 1.000 |
| 3.0 | hard_semantic | 262 | 0.6333 [0.4579, 0.8087] | 0.633 | 987.0 | 2.07 | 107.66 | 0.000 |
| 3.0 | quantized_soft_semantic | 327 | 0.7000 [0.5332, 0.8668] | 0.700 | 1180.7 | 2.20 | 132.93 | 0.000 |
| 3.0 | compressed_spectrogram_png | 116541 | 0.0527 [0.0443, 0.0611] | 0.000 | 242536.0 | 249.37 | 24.64 | 1.000 |
| 3.0 | spectrogram_uint8 | 131072 | 0.0490 [0.0413, 0.0566] | 0.000 | 242536.0 | 249.37 | 25.73 | 1.000 |
| 3.0 | iq_int12_complex | 24000000 | 0.0003 [0.0002, 0.0003] | 0.000 | 242536.0 | 249.37 | 27.51 | 1.000 |
| 6.0 | hard_semantic | 262 | 1.0000 [1.0000, 1.0000] | 1.000 | 658.0 | 1.38 | 189.99 | 0.000 |
| 6.0 | quantized_soft_semantic | 327 | 1.0000 [1.0000, 1.0000] | 1.000 | 770.0 | 1.43 | 227.87 | 0.000 |
| 6.0 | compressed_spectrogram_png | 116541 | 0.9980 [0.9965, 0.9995] | 0.800 | 233651.6 | 240.41 | 483.91 | 0.033 |
| 6.0 | spectrogram_uint8 | 131072 | 0.9195 [0.9141, 0.9250] | 0.000 | 242536.0 | 249.37 | 483.32 | 1.000 |
| 6.0 | iq_int12_complex | 24000000 | 0.0050 [0.0050, 0.0051] | 0.000 | 242536.0 | 249.37 | 485.92 | 1.000 |
| 9.0 | hard_semantic | 262 | 1.0000 [1.0000, 1.0000] | 1.000 | 658.0 | 1.38 | 189.99 | 0.000 |
| 9.0 | quantized_soft_semantic | 327 | 1.0000 [1.0000, 1.0000] | 1.000 | 770.0 | 1.43 | 227.87 | 0.000 |
| 9.0 | compressed_spectrogram_png | 116541 | 1.0000 [1.0000, 1.0000] | 1.000 | 226296.0 | 232.85 | 500.50 | 0.000 |
| 9.0 | spectrogram_uint8 | 131072 | 0.9531 [0.9531, 0.9531] | 0.000 | 242536.0 | 249.37 | 500.98 | 1.000 |
| 9.0 | iq_int12_complex | 24000000 | 0.0052 [0.0052, 0.0052] | 0.000 | 242536.0 | 249.37 | 500.98 | 1.000 |
| 12.0 | hard_semantic | 262 | 1.0000 [1.0000, 1.0000] | 1.000 | 658.0 | 1.38 | 189.99 | 0.000 |
| 12.0 | quantized_soft_semantic | 327 | 1.0000 [1.0000, 1.0000] | 1.000 | 770.0 | 1.43 | 227.87 | 0.000 |
| 12.0 | compressed_spectrogram_png | 116541 | 1.0000 [1.0000, 1.0000] | 1.000 | 226296.0 | 232.85 | 500.50 | 0.000 |
| 12.0 | spectrogram_uint8 | 131072 | 0.9531 [0.9531, 0.9531] | 0.000 | 242536.0 | 249.37 | 500.98 | 1.000 |
| 12.0 | iq_int12_complex | 24000000 | 0.0052 [0.0052, 0.0052] | 0.000 | 242536.0 | 249.37 | 500.98 | 1.000 |

## 4. Analytic abstraction versus waveform simulation

The validation chain generates random packet bits, appends CRC-16/CCITT, applies the configured FEC, performs modulation, block fading/AWGN, hard demodulation, FEC decoding, and CRC checking. Eb/N0 is defined per transmitted coded bit; FEC therefore improves reliability by spending extra channel bits/energy rather than by receiving a free coding gain.

| Eb/N0 | Analytic packet success | Waveform success (95% CI) | Absolute error | Trials |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0018 | 0.0015 [0.0000, 0.0032] | 0.0003 | 2000 |
| 3.0 | 0.5412 | 0.5385 [0.5167, 0.5603] | 0.0027 | 2000 |
| 6.0 | 0.9929 | 0.9930 [0.9893, 0.9967] | 0.0001 | 2000 |
| 9.0 | 1.0000 | 1.0000 [1.0000, 1.0000] | 0.0000 | 2000 |

## 5. Interpretation rules

1. A small representation completing within the deadline demonstrates latency feasibility, not superior sensing accuracy.
2. Partial PNG/spectrogram/IQ delivery is counted only as recovered link payload; it is not assumed to be directly usable by a detector until a fragment-aware decoder is evaluated.
3. The IQ value is a theoretical 1,000,000-sample, 12+12-bit capacity reference because the local RadDet copy contains spectrograms and labels rather than raw IQ.
4. Hamming(7,4) is an auditable research baseline, not a claim of 5G-grade coding. A later hardware/standard-aligned stage should add LDPC or Polar coding.
5. AWGN results are the controlled first step. Rician/Rayleigh, burst errors, HARQ feedback delay, and task-level resource selection are required before the final thesis claim.

## 6. Reproducibility

- Configuration: `configs/stage2_digital_link.json`.
- Raw repeated trials: `digital_link_trials.csv`.
- Aggregated statistics: `digital_link_summary.csv`.
- Waveform validation: `waveform_validation.csv`.
- Machine-readable record: `stage2_digital_link_result.json`.
