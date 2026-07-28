# RadDet Occupancy-Mask SNR Evaluation

- split: test
- frames: 1000
- selected threshold: 0.85
- IoU threshold: 0.30

| SNR (dB) | Frames | Precision | Recall | F1 | Mean IoU | Semantic bits/frame |
|---:|---:|---:|---:|---:|---:|---:|
| -20 | 181 | 0.7238 | 0.7600 | 0.7415 | 0.5531 | 250.8 |
| -12 | 163 | 0.5747 | 0.6098 | 0.5917 | 0.5548 | 247.4 |
| -4 | 141 | 0.5176 | 0.5714 | 0.5432 | 0.4795 | 252.4 |
| 4 | 172 | 0.4375 | 0.6049 | 0.5078 | 0.4280 | 255.9 |
| 12 | 167 | 0.2759 | 0.4819 | 0.3509 | 0.4490 | 271.5 |
| 20 | 176 | 0.3168 | 0.6882 | 0.4339 | 0.5118 | 291.6 |

## Interpretation

This table separates the payload question from the semantic-extraction reliability question. If low-SNR F1 collapses while high-SNR F1 remains usable, the next contribution should focus on soft semantic packets, multi-view fusion, and SNR-aware confidence rather than only reducing bits.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\raddet_occupancy_mask_by_snr.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\raddet_occupancy_mask_by_snr.json`
