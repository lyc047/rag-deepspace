# Stage 2 Task-Level Digital Link Validation

## Scope

This experiment connects the fair digital link to the frozen RadDet occupancy detector and a controlled resource-pressure task. All schemes share packetization, CRC, Hamming(7,4), QPSK, AWGN, ARQ, symbol rate, and the 0.25 s decision deadline.

The uint8 spectrogram baseline uses independently delivered row-major fragments; missing bytes are concealed with the mean of received bytes before the frozen detector runs. PNG requires complete delivery because arbitrary PNG fragments are not independently decodable. Raw IQ is excluded from task metrics because the local RadDet copy has no raw IQ samples.

## No-link frozen detector reference

- Frames: 200 from frozen `test` split.
- Pressure decisions: 50, each formed from 4 unrelated observations; this is not spatial cooperation evidence.
- Detection F1: 0.5484; precision: 0.4626; recall: 0.6733.
- Clean resource rate: 0.9400; mean occupancy regret: 0.000031.

## Fair-link task results

| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | Tx bits/decision | Parallel-link latency (ms) | Delivered payload |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3.0 | compressed_spectrogram_png | 0.0000 | 0.6600 [0.6600, 0.6600] | 0.014281 [0.014281, 0.014281] | 970144.0 | 249.37 | 0.0592 |
| 3.0 | hard_semantic | 0.4266 | 0.8440 [0.8362, 0.8518] | 0.004155 [0.003518, 0.004793] | 4383.1 | 2.86 | 0.6050 |
| 3.0 | quantized_soft_semantic | 0.3461 | 0.8120 [0.7778, 0.8462] | 0.006137 [0.004582, 0.007692] | 5258.7 | 3.15 | 0.5265 |
| 3.0 | spectrogram_uint8_fragment | 0.0150 | 0.6600 [0.6600, 0.6600] | 0.012425 [0.012425, 0.012425] | 970144.0 | 249.37 | 0.0523 |
| 6.0 | compressed_spectrogram_png | 0.5073 | 0.9120 [0.8920, 0.9320] | 0.002071 [0.001019, 0.003123] | 918678.0 | 244.50 | 0.9793 |
| 6.0 | hard_semantic | 0.5495 | 0.9400 [0.9400, 0.9400] | 0.000031 [0.000031, 0.000031] | 2698.8 | 1.51 | 0.9990 |
| 6.0 | quantized_soft_semantic | 0.5484 | 0.9400 [0.9400, 0.9400] | 0.000031 [0.000031, 0.000031] | 3168.6 | 1.67 | 1.0000 |
| 6.0 | spectrogram_uint8_fragment | 0.4786 | 0.9400 [0.9400, 0.9400] | 0.000026 [0.000016, 0.000036] | 970144.0 | 249.37 | 0.9215 |
| 9.0 | compressed_spectrogram_png | 0.5419 | 0.9400 [0.9400, 0.9400] | 0.000429 [0.000429, 0.000429] | 893552.7 | 239.39 | 0.9839 |
| 9.0 | hard_semantic | 0.5484 | 0.9400 [0.9400, 0.9400] | 0.000031 [0.000031, 0.000031] | 2666.4 | 1.45 | 1.0000 |
| 9.0 | quantized_soft_semantic | 0.5484 | 0.9400 [0.9400, 0.9400] | 0.000031 [0.000031, 0.000031] | 3123.1 | 1.61 | 1.0000 |
| 9.0 | spectrogram_uint8_fragment | 0.4906 | 0.9400 [0.9400, 0.9400] | 0.000031 [0.000031, 0.000031] | 970144.0 | 249.37 | 0.9531 |

## Claim boundary

- This is a controlled H4/H1 pressure validation. Aggregated observations are unrelated and therefore provide no evidence for multi-UAV spatial cooperation H3.
- The detector and threshold are frozen; no test-set tuning is performed.
- Spectrogram fragment concealment is an explicit baseline choice and must be ablated against tiling/erasure-aware training before the final thesis.
- The reported confidence intervals quantify link Monte Carlo variation on the fixed frame subset; a final thesis experiment must repeat across a larger frozen test set and additional channel models.
