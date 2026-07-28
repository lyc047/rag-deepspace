# Stage 2 Enhanced Robustness and Pareto Report

## Protocol

The frozen detector, source frames, packetization, CRC, FEC, modulation, ARQ, resource task, and random-seed policy are unchanged. Robustness varies only the reporting channel model and Eb/N0. Pareto experiments vary only the shared latency budget under AWGN at 6 dB.

No-link reference: F1=0.5441, clean rate=0.9600, regret=0.000051.

## Reporting-channel robustness

| Channel | Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Payload delivered |
|---|---:|---|---:|---:|---:|---:|---:|
| awgn | 3.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.0569 |
| awgn | 3.0 | hard_semantic | 0.4355 | 0.8320 [0.8064, 0.8576] | 0.004080 [0.002670, 0.005491] | 4481.9 | 0.6010 |
| awgn | 6.0 | compressed_spectrogram_png | 0.5123 | 0.9120 [0.8892, 0.9348] | 0.002641 [0.001551, 0.003732] | 920753.3 | 0.9798 |
| awgn | 6.0 | hard_semantic | 0.5451 | 0.9600 [0.9600, 0.9600] | 0.000051 [0.000051, 0.000051] | 2748.2 | 0.9990 |
| awgn | 9.0 | compressed_spectrogram_png | 0.5439 | 0.9600 [0.9600, 0.9600] | 0.000051 [0.000051, 0.000051] | 894759.2 | 0.9844 |
| awgn | 9.0 | hard_semantic | 0.5441 | 0.9600 [0.9600, 0.9600] | 0.000051 [0.000051, 0.000051] | 2709.3 | 1.0000 |
| rayleigh | 3.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.2827 |
| rayleigh | 3.0 | hard_semantic | 0.4113 | 0.8720 [0.8394, 0.9046] | 0.003345 [0.002348, 0.004341] | 4514.6 | 0.5620 |
| rayleigh | 6.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.5481 |
| rayleigh | 6.0 | hard_semantic | 0.4948 | 0.9240 [0.9057, 0.9423] | 0.000854 [0.000494, 0.001213] | 3842.0 | 0.8140 |
| rayleigh | 9.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.7632 |
| rayleigh | 9.0 | hard_semantic | 0.5409 | 0.9480 [0.9360, 0.9600] | 0.000297 [0.000087, 0.000506] | 3368.8 | 0.9550 |
| rician | 3.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.2787 |
| rician | 3.0 | hard_semantic | 0.4389 | 0.8920 [0.8657, 0.9183] | 0.002448 [0.001509, 0.003386] | 4357.4 | 0.6260 |
| rician | 6.0 | compressed_spectrogram_png | 0.0000 | 0.6400 [0.6400, 0.6400] | 0.013573 [0.013573, 0.013573] | 970144.0 | 0.7027 |
| rician | 6.0 | hard_semantic | 0.5317 | 0.9400 [0.9225, 0.9575] | 0.000559 [0.000192, 0.000926] | 3356.2 | 0.9210 |
| rician | 9.0 | compressed_spectrogram_png | 0.0789 | 0.6680 [0.6476, 0.6884] | 0.012228 [0.011309, 0.013148] | 965990.0 | 0.9395 |
| rician | 9.0 | hard_semantic | 0.5417 | 0.9560 [0.9482, 0.9638] | 0.000133 [0.000016, 0.000249] | 2937.6 | 0.9950 |

## Shared-latency-budget Pareto sweep

| Budget | Scheme | F1 | Clean rate | Regret | bit/decision | Link latency | Budget exhaustion |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 ms | compressed_spectrogram_png | 0.0000 | 0.6400 | 0.013573 | 15904.0 | 4.09 ms | 1.000 |
| 5 ms | hard_semantic | 0.5441 | 0.9600 | 0.000051 | 2743.3 | 1.53 ms | 0.000 |
| 5 ms | spectrogram_uint8_fragment | 0.0000 | 0.7200 | 0.013615 | 15904.0 | 4.09 ms | 1.000 |
| 25 ms | compressed_spectrogram_png | 0.0000 | 0.6400 | 0.013573 | 95424.0 | 24.53 ms | 1.000 |
| 25 ms | hard_semantic | 0.5441 | 0.9600 | 0.000051 | 2727.9 | 1.50 ms | 0.000 |
| 25 ms | spectrogram_uint8_fragment | 0.0000 | 0.7200 | 0.013615 | 95424.0 | 24.53 ms | 1.000 |
| 100 ms | compressed_spectrogram_png | 0.0000 | 0.6400 | 0.013573 | 381696.0 | 98.11 ms | 1.000 |
| 100 ms | hard_semantic | 0.5441 | 0.9600 | 0.000051 | 2750.9 | 1.53 ms | 0.000 |
| 100 ms | spectrogram_uint8_fragment | 0.0280 | 0.6000 | 0.017135 | 381696.0 | 98.11 ms | 1.000 |
| 250 ms | compressed_spectrogram_png | 0.5387 | 0.9067 | 0.002369 | 920646.2 | 244.80 ms | 0.143 |
| 250 ms | hard_semantic | 0.5441 | 0.9600 | 0.000051 | 2779.7 | 1.59 ms | 0.000 |
| 250 ms | spectrogram_uint8_fragment | 0.5047 | 0.9600 | 0.000017 | 970144.0 | 249.37 ms | 1.000 |

## Claim boundary

- The 100-frame subset is a diagnostic enhancement experiment, not the final full-test thesis table.
- Rayleigh/Rician are independent block-fading reporting channels with perfect receiver-side equalization; burst errors and channel-estimation error remain future work.
- Unrelated four-observation pressure aggregation tests H1/H4 only and does not establish H3 spatial cooperation.
- Pareto points are produced by changing a shared deadline, not by granting method-specific coding or retransmission rules.
- PNG and row-major spectrogram fragmentation represent explicit codec/recovery baselines; future tile and erasure-aware baselines are still required.
