# Stage 2 Actual LDPC Waveform and Full-Task Validation

## Code and energy protocol

A deterministic regular LDPC code is generated with pyldpc 0.7.9: n=1200, k=602, rate=0.5017, dv=3, dc=6. Each information block contains a 96-bit application header and CRC-16, leaving 490 application bits per codeword.

Both LDPC and Hamming waveform baselines use unit-energy BPSK over AWGN. Eb/N0 is defined per transmitted coded bit; the lower-rate code spends more transmitted bits/energy rather than receiving an artificial SNR shift. Packet success requires CRC validity and exact application/header recovery.

## Actual waveform packet success

| Eb/N0 | LDPC success (95% CI) | Hamming success (95% CI) |
|---:|---:|---:|
| 0.0 | 0.0000 [0.0000, 0.0714] | 0.0000 [0.0000, 0.0714] |
| 1.0 | 0.1800 [0.0977, 0.3080] | 0.0000 [0.0000, 0.0714] |
| 2.0 | 0.7600 [0.6259, 0.8570] | 0.0200 [0.0035, 0.1050] |
| 3.0 | 1.0000 [0.9286, 1.0000] | 0.1600 [0.0834, 0.2851] |
| 4.0 | 1.0000 [0.9286, 1.0000] | 0.6200 [0.4815, 0.7414] |
| 5.0 | 1.0000 [0.9286, 1.0000] | 0.8400 [0.7149, 0.9166] |
| 6.0 | 1.0000 [0.9286, 1.0000] | 1.0000 [0.9286, 1.0000] |

## Full cached task validation (20001 frames, frozen PSD=1 bit/value)

The measured LDPC packet-success probability at each Eb/N0 is used for scalable packet/ARQ Monte Carlo on the frozen full-test features. Both hard semantic and PSD use the same LDPC codeword, CRC, retransmission limit, symbol rate, and deadline.

| Eb/N0 | Scheme | F1 | Clean rate (95% CI) | Regret (95% CI) | bit/decision | Parallel latency |
|---:|---|---:|---:|---:|---:|---:|
| 2.0 | frozen_psd_ldpc | n/a | 0.6998 [0.6992, 0.7004] | 0.015699 [0.015640, 0.015757] | 5951.0 | 3.75 ms |
| 2.0 | hard_semantic_ldpc | 0.4740 | 0.9161 [0.9139, 0.9182] | 0.001431 [0.001297, 0.001564] | 6052.6 | 3.89 ms |
| 3.0 | frozen_psd_ldpc | n/a | 0.6994 [0.6994, 0.6994] | 0.015757 [0.015757, 0.015757] | 4800.0 | 2.25 ms |
| 3.0 | hard_semantic_ldpc | 0.4858 | 0.9290 [0.9290, 0.9290] | 0.000661 [0.000661, 0.000661] | 4867.0 | 2.37 ms |
| 4.0 | frozen_psd_ldpc | n/a | 0.6994 [0.6994, 0.6994] | 0.015757 [0.015757, 0.015757] | 4800.0 | 2.25 ms |
| 4.0 | hard_semantic_ldpc | 0.4858 | 0.9290 [0.9290, 0.9290] | 0.000661 [0.000661, 0.000661] | 4867.0 | 2.37 ms |

## Claim boundary

- This is an actual regular LDPC/BP implementation, but it is not a 3GPP NR base graph, rate-matching, or HARQ implementation.
- The full task run reuses frozen detector/PSD features; waveform decoding is measured separately and transferred only as an empirical packet-success curve.
- AWGN assumes perfect synchronization and channel knowledge. Burst errors, fading, finite-length interleaving, and decoder complexity/energy remain future work.
- The RadDet and unrelated-pressure limitations from the full-test report remain unchanged.
