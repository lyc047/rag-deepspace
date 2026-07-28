# Semantic Link Simulation

## Baseline detector before communication

- F1: 0.5160
- Precision: 0.4389
- Recall: 0.6260
- Mean IoU: 0.5033

## Representative operating points

### packet_loss=0, BER=0

| Scheme | F1 | Precision | Recall | bits/frame | vs 8-bit spectrogram | vs 12-bit IQ |
|---|---:|---:|---:|---:|---:|---:|
| hard | 0.5160 | 0.4389 | 0.6260 | 262.0 | 500.3x | 91605.9x |
| hard_rep3 | 0.5160 | 0.4389 | 0.6260 | 786.0 | 166.8x | 30535.3x |
| soft | 0.5160 | 0.4389 | 0.6260 | 326.8 | 401.1x | 73448.4x |
| soft_rep3 | 0.5160 | 0.4389 | 0.6260 | 980.3 | 133.7x | 24482.8x |
| spectrogram8 | 0.5160 | 0.4389 | 0.6260 | 131072.0 | 1.0x | 183.1x |
| iq12 | 0.5160 | 0.4389 | 0.6260 | 24000000.0 | 0.0x | 1.0x |

### packet_loss=0.05, BER=1e-05

| Scheme | F1 | Precision | Recall | bits/frame | vs 8-bit spectrogram | vs 12-bit IQ |
|---|---:|---:|---:|---:|---:|---:|
| hard | 0.5060 | 0.4404 | 0.5945 | 262.0 | 500.3x | 91605.9x |
| hard_rep3 | 0.5159 | 0.4388 | 0.6259 | 786.0 | 166.8x | 30535.3x |
| soft | 0.5056 | 0.4403 | 0.5939 | 326.8 | 401.1x | 73448.4x |
| soft_rep3 | 0.5159 | 0.4388 | 0.6259 | 980.3 | 133.7x | 24482.8x |
| spectrogram8 | 0.0010 | 0.1708 | 0.0005 | 131072.0 | 1.0x | 183.1x |
| iq12 | 0.0000 | 0.0000 | 0.0000 | 24000000.0 | 0.0x | 1.0x |

### packet_loss=0.1, BER=0.0001

| Scheme | F1 | Precision | Recall | bits/frame | vs 8-bit spectrogram | vs 12-bit IQ |
|---|---:|---:|---:|---:|---:|---:|
| hard | 0.4895 | 0.4348 | 0.5602 | 262.0 | 500.3x | 91605.9x |
| hard_rep3 | 0.5159 | 0.4389 | 0.6257 | 786.0 | 166.8x | 30535.3x |
| soft | 0.4870 | 0.4342 | 0.5546 | 326.8 | 401.1x | 73448.4x |
| soft_rep3 | 0.5159 | 0.4389 | 0.6257 | 980.3 | 133.7x | 24482.8x |
| spectrogram8 | 0.0000 | 0.0000 | 0.0000 | 131072.0 | 1.0x | 183.1x |
| iq12 | 0.0000 | 0.0000 | 0.0000 | 24000000.0 | 0.0x | 1.0x |

### packet_loss=0.2, BER=0.0001

| Scheme | F1 | Precision | Recall | bits/frame | vs 8-bit spectrogram | vs 12-bit IQ |
|---|---:|---:|---:|---:|---:|---:|
| hard | 0.4687 | 0.4409 | 0.5005 | 262.0 | 500.3x | 91605.9x |
| hard_rep3 | 0.5145 | 0.4390 | 0.6214 | 786.0 | 166.8x | 30535.3x |
| soft | 0.4670 | 0.4414 | 0.4960 | 326.8 | 401.1x | 73448.4x |
| soft_rep3 | 0.5145 | 0.4390 | 0.6214 | 980.3 | 133.7x | 24482.8x |
| spectrogram8 | 0.0000 | 0.0000 | 0.0000 | 131072.0 | 1.0x | 183.1x |
| iq12 | 0.0000 | 0.0000 | 0.0000 | 24000000.0 | 0.0x | 1.0x |

## Interpretation

This simulation turns the detector output into a communication object. Hard semantic packets have the smallest payload; repeated semantic packets trade a small payload increase for much better packet-erasure robustness. Large-payload spectrogram/IQ forwarding is modeled conservatively and becomes fragile without FEC/ARQ because each frame spans many packets.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\semantic_link_simulation.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\semantic_link_simulation.json`
