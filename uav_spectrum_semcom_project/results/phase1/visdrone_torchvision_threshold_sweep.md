# VisDrone Torchvision Threshold Sweep

- Frames: 80
- IoU threshold: 0.5

| Score threshold | Precision | Recall | F1 | pred boxes/frame | semantic bits/frame | JPEG/semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.1415 | 0.1465 | 0.1439 | 50.8 | 3866.6 | 291.9x |
| 0.05 | 0.1803 | 0.1398 | 0.1575 | 38.1 | 2949.5 | 382.7x |
| 0.10 | 0.2662 | 0.1276 | 0.1725 | 23.5 | 1902.8 | 593.2x |
| 0.15 | 0.3408 | 0.1167 | 0.1738 | 16.8 | 1418.6 | 795.7x |
| 0.25 | 0.4626 | 0.0930 | 0.1548 | 9.9 | 919.1 | 1228.1x |
| 0.35 | 0.5827 | 0.0736 | 0.1307 | 6.2 | 655.4 | 1722.2x |
| 0.50 | 0.7563 | 0.0537 | 0.1004 | 3.5 | 460.1 | 2453.2x |

## Best threshold by F1

- Threshold: 0.15
- Precision: 0.3408
- Recall: 0.1167
- F1: 0.1738

Plot: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_torchvision_threshold_sweep.png`
CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_torchvision_threshold_sweep.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_torchvision_threshold_sweep.json`
