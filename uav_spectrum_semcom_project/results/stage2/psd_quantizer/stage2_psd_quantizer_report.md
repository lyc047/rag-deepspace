# Stage 2 PSD Quantizer Selection

## Selection protocol

Quantizer resolution is selected only on the frozen validation split. The rule is lexicographic: minimum occupancy regret, then maximum clean rate, then minimum bit width. The selected width is subsequently frozen and evaluated once on the test split and through the fair digital reporting link.

The PSD values are four normalized mean grayscale energies from RadDet max-hold spectrograms. They are a soft-information proxy, not calibrated receiver power in dBm.

## Validation selection

| Quantizer bits/value | Clean rate | Regret |
|---:|---:|---:|
| 1 | 0.8000 | 0.011455 |
| 2 | 0.8000 | 0.011455 |
| 3 | 0.7600 | 0.012334 |
| 4 | 0.7600 | 0.015889 |
| 6 | 0.6000 | 0.029816 |
| 8 | 0.5600 | 0.026059 |

Selected resolution: **1 bit/value**.

## Frozen test result without reporting errors

- Clean rate: 0.6400.
- Mean occupancy regret: 0.013573.

## Frozen quantizer through the digital link

| Eb/N0 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Link latency | Payload delivered |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 3577.2 | 2.63 ms | 0.6800 |
| 6.0 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 2319.0 | 1.39 ms | 1.0000 |
| 9.0 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 2296.0 | 1.34 ms | 1.0000 |

## Claim boundary

- Candidate bit widths and the selection rule were declared before this run; test results do not alter the selected width.
- The 100-frame validation and test subsets are diagnostic. Final thesis evidence requires the complete frozen splits.
- The quantizer comparison isolates representation resolution; it does not tune energy thresholds or frequency aggregation on the test set.
- Selection on grayscale energy may not transfer to calibrated IQ/PSD measurements and must be repeated when real receiver power is available.
