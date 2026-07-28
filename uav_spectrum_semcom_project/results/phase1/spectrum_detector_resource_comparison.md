# Spectrum Detector Resource Comparison

- Packet loss: 0.2
- BER: 0.0001
- UAV observations per decision: 4

| Detector | Scheme | Clean rate | Net utility | Regret | bits/decision |
|---|---|---:|---:|---:|---:|
| mask_baseline | oracle | 0.9400 | 0.9931 | 0.0000 | 0.0 |
| mask_baseline | random | 0.7056 | 0.9795 | 0.0134 | 0.0 |
| mask_baseline | semantic_hard | 0.8911 | 0.9895 | 0.0035 | 1050.2 |
| mask_baseline | semantic_hard_rep3 | 0.9300 | 0.9918 | 0.0012 | 3150.5 |
| mask_baseline | semantic_soft_rep3 | 0.9300 | 0.9918 | 0.0012 | 3935.7 |
| mask_baseline | spectrogram8_partial | 0.8750 | 0.9892 | 0.0038 | 655360.0 |
| grid16_yolo_style | oracle | 0.9400 | 0.9931 | 0.0000 | 0.0 |
| grid16_yolo_style | random | 0.7000 | 0.9792 | 0.0135 | 0.0 |
| grid16_yolo_style | semantic_hard | 0.8361 | 0.9863 | 0.0066 | 990.7 |
| grid16_yolo_style | semantic_hard_rep3 | 0.8739 | 0.9881 | 0.0049 | 2972.0 |
| grid16_yolo_style | semantic_soft_rep3 | 0.8672 | 0.9884 | 0.0046 | 3539.2 |
| grid16_yolo_style | spectrogram8_partial | 0.8522 | 0.9876 | 0.0053 | 655360.0 |

## Interpretation

This compares whether the improved lightweight YOLO-style detector changes the downstream resource decision, not only the detection F1.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\spectrum_detector_resource_comparison.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\spectrum_detector_resource_comparison.json`
