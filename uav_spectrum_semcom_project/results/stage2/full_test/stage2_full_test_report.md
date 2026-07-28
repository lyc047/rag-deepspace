# Stage 2 Full Frozen Test-Split Validation

## Protocol

All 20001 frames in the frozen local `test` directory are used; 20000 frames form 5000 complete four-observation pressure decisions and the remainder is excluded deterministically.

The PSD quantizer is frozen at 1 bit/value from the independent validation-split selection. The test split does not select detector thresholds, quantizer resolution, FEC, ARQ, packet size, channel conditions, or latency budget.

No-link detector reference: F1=0.4858, clean rate=0.9290, regret=0.000661.

## Full-split fair-link results

| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency | Payload delivered |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3.0 | frozen_quantized_psd | n/a | 0.7018 [0.7011, 0.7024] | 0.015494 [0.015389, 0.015598] | 3616.1 | 2.63 ms | 0.6731 |
| 3.0 | hard_semantic | 0.3868 | 0.8402 [0.8394, 0.8411] | 0.006046 [0.005989, 0.006103] | 4340.3 | 2.82 ms | 0.6171 |
| 6.0 | frozen_quantized_psd | n/a | 0.6994 [0.6994, 0.6994] | 0.015757 [0.015757, 0.015757] | 2318.5 | 1.39 ms | 1.0000 |
| 6.0 | hard_semantic | 0.4857 | 0.9289 [0.9288, 0.9290] | 0.000666 [0.000660, 0.000672] | 2699.3 | 1.51 ms | 0.9998 |
| 9.0 | frozen_quantized_psd | n/a | 0.6994 [0.6994, 0.6994] | 0.015757 [0.015757, 0.015757] | 2296.0 | 1.34 ms | 1.0000 |
| 9.0 | hard_semantic | 0.4858 | 0.9290 [0.9290, 0.9290] | 0.000661 [0.000661, 0.000661] | 2663.9 | 1.44 ms | 1.0000 |

## Claim boundary

- The local directory named `test` contains 20,001 frames. Stage 1 froze this observed mapping but did not verify the upstream provenance of the nonstandard train/val/test counts; the directory is therefore not relabeled.
- Four unrelated frames form a resource-pressure decision. This is full-split H1/H4 evidence, not H3 spatially correlated multi-UAV evidence.
- RadDet is synthetic wideband radar data, not real UAV RF capture.
- PSD is normalized grayscale energy rather than calibrated dBm power.
- Hamming(7,4) is an auditable coding baseline rather than a 5G-grade channel code.
