# Spectrum-Aware UAV Visual Semantic Policy Simulation

## Setup

- Visual dataset: VisDrone2019-DET-val, 548 frames.
- Spectrum source: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\resource_optimization_simulation.json`
- Packet loss: 0.2
- BER: 0.0001
- High-priority visual frame: at least 10 priority-class objects.

## Results

| Spectrum scheme | Clean rate | Full JPEG bits/frame | Semantic-only bits/frame | Adaptive bits/frame | Adaptive/full ratio | Priority detail rate | Detail/Mbit |
|---|---:|---:|---:|---:|---:|---:|---:|
| iq12_gate | 0.6974 | 1185557.3 | 5305.6 | 113632.4 | 10.4x | 0.6950 | 6.12 |
| random | 0.6962 | 1185557.3 | 5305.6 | 114271.9 | 10.4x | 0.6931 | 6.06 |
| spectrogram8_gate | 0.7020 | 1185557.3 | 5305.6 | 118991.3 | 10.0x | 0.7375 | 6.20 |
| fixed_ch0 | 0.7006 | 1185557.3 | 5305.6 | 122652.1 | 9.7x | 0.7259 | 5.92 |
| semantic_hard | 0.8846 | 1185557.3 | 5305.6 | 149072.8 | 8.0x | 0.9015 | 6.05 |
| semantic_soft_rep3 | 0.9316 | 1185557.3 | 5305.6 | 149778.1 | 7.9x | 0.9305 | 6.21 |
| semantic_soft | 0.8854 | 1185557.3 | 5305.6 | 150150.0 | 7.9x | 0.9131 | 6.08 |
| semantic_hard_rep3 | 0.9314 | 1185557.3 | 5305.6 | 152934.9 | 7.8x | 0.9421 | 6.16 |

## Interpretation

This is the first bridge from the completed spectrum system to UAV visual semantic transmission. The adaptive policy uses spectrum resource quality to decide whether to transmit ROI image detail or only object-level visual semantics. The result should be treated as a policy-level feasibility check, not yet as a final trained multimodal semantic communication system.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multimodal_policy_simulation.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\multimodal_policy_simulation.json`
