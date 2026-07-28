# Stage 2 Classical Soft-Information and Tile Baselines

## Protocol

This experiment adds two missing fair baselines: quantized per-channel PSD/energy reports and independently decodable 16x16 uint8 spectrogram tiles. The frozen detector, source frames, resource task, packetization, CRC, Hamming(7,4), QPSK, AWGN, ARQ, and 0.25 s deadline are unchanged.

No-link detector reference: F1=0.5441, clean rate=0.9600, regret=0.000051.

PSD baselines average quantized channel energy from successfully received observations and select the minimum-energy contiguous block. They do not produce boxes, so detection F1 is not applicable. Tile transmission uses a seeded random tile order and a 32-bit tile header; only completely decoded tiles are restored before running the frozen detector.

## Results

| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency | Payload/tile delivery |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3.0 | hard_semantic | 0.4383 | 0.8160 [0.7477, 0.8843] | 0.005819 [0.003248, 0.008390] | 4453.2 | 2.86 ms | 0.5980 |
| 3.0 | quantized_psd_4bit | n/a | 0.6480 [0.6187, 0.6773] | 0.012267 [0.010616, 0.013919] | 3819.1 | 2.69 ms | 0.6520 |
| 3.0 | quantized_psd_8bit | n/a | 0.5680 [0.5523, 0.5837] | 0.020063 [0.019711, 0.020416] | 4016.9 | 2.71 ms | 0.6340 |
| 3.0 | spectrogram_tile16 | 0.0000 | 0.5760 [0.5446, 0.6074] | 0.017852 [0.016605, 0.019099] | 848839.5 | 249.65 ms | 0.0042 |
| 6.0 | hard_semantic | 0.5441 | 0.9600 [0.9600, 0.9600] | 0.000051 [0.000051, 0.000051] | 2737.3 | 1.51 ms | 1.0000 |
| 6.0 | quantized_psd_4bit | n/a | 0.6800 [0.6800, 0.6800] | 0.011121 [0.011121, 0.011121] | 2446.5 | 1.43 ms | 1.0000 |
| 6.0 | quantized_psd_8bit | n/a | 0.6000 [0.6000, 0.6000] | 0.019511 [0.019511, 0.019511] | 2545.2 | 1.42 ms | 1.0000 |
| 6.0 | spectrogram_tile16 | 0.2886 | 0.9600 [0.9600, 0.9600] | 0.000569 [0.000397, 0.000740] | 803972.7 | 249.33 ms | 0.7139 |
| 9.0 | hard_semantic | 0.5441 | 0.9600 [0.9600, 0.9600] | 0.000051 [0.000051, 0.000051] | 2709.3 | 1.46 ms | 1.0000 |
| 9.0 | quantized_psd_4bit | n/a | 0.6800 [0.6800, 0.6800] | 0.011121 [0.011121, 0.011121] | 2408.0 | 1.35 ms | 1.0000 |
| 9.0 | quantized_psd_8bit | n/a | 0.6000 [0.6000, 0.6000] | 0.019511 [0.019511, 0.019511] | 2520.0 | 1.36 ms | 1.0000 |
| 9.0 | spectrogram_tile16 | 0.2995 | 0.9520 [0.9363, 0.9677] | 0.000640 [0.000293, 0.000987] | 802816.0 | 249.45 ms | 0.7344 |

## Interpretation boundary

- PSD energy is derived from RadDet max-hold grayscale spectrograms, not calibrated receiver power in dBm; it is a classical soft-information proxy.
- PSD uses only four candidate-channel values. A later sweep must study frequency resolution and quantizer bits under matched total budgets.
- Independent tiles remove the unfair whole-image gate but add explicit per-tile metadata and require complete tile delivery.
- The 100-frame unrelated-observation pressure subset tests H1/H4 only, not H3 multi-UAV cooperation.
- No test-set threshold is tuned in this experiment.
