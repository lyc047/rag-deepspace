# Multi-UAV Spectrum Map Fusion Evaluation

- Packet loss: 0.2
- BER: 0.0001
- Decision steps/trial: 600
- Trials: 3

| UAVs | Scheme | Block MAE | Channel MAE | Clean-block accuracy | bits/decision |
|---:|---|---:|---:|---:|---:|
| 1 | iq12_partial | 0.0049 | 0.0051 | 0.9420 | 30000000.0 |
| 1 | semantic_hard | 0.0051 | 0.0052 | 0.9402 | 262.1 |
| 1 | semantic_hard_rep3 | 0.0053 | 0.0054 | 0.9461 | 786.4 |
| 1 | spectrogram8_partial | 0.0050 | 0.0052 | 0.9385 | 163840.0 |
| 2 | iq12_partial | 0.0098 | 0.0101 | 0.8835 | 60000000.0 |
| 2 | semantic_hard | 0.0101 | 0.0104 | 0.8815 | 526.5 |
| 2 | semantic_hard_rep3 | 0.0111 | 0.0112 | 0.8878 | 1579.4 |
| 2 | spectrogram8_partial | 0.0099 | 0.0103 | 0.8846 | 327680.0 |
| 4 | iq12_partial | 0.0171 | 0.0183 | 0.7919 | 120000000.0 |
| 4 | semantic_hard | 0.0179 | 0.0190 | 0.7944 | 1047.3 |
| 4 | semantic_hard_rep3 | 0.0206 | 0.0208 | 0.7904 | 3142.0 |
| 4 | spectrogram8_partial | 0.0172 | 0.0185 | 0.8102 | 655360.0 |
| 8 | iq12_partial | 0.0314 | 0.0350 | 0.7500 | 240000000.0 |
| 8 | semantic_hard | 0.0334 | 0.0364 | 0.7422 | 2099.2 |
| 8 | semantic_hard_rep3 | 0.0413 | 0.0420 | 0.7348 | 6297.6 |
| 8 | spectrogram8_partial | 0.0313 | 0.0346 | 0.7413 | 1310720.0 |

## Interpretation

This evaluates the spectrum-map layer directly, before resource selection. Lower MAE means the fused received reports better approximate the true multi-UAV occupancy map.

Block-MAE plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\spectrum_map_fusion_block_mae.png`
CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\spectrum_map_fusion_eval.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\spectrum_map_fusion_eval.json`
