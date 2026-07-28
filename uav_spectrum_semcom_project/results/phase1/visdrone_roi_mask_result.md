# VisDrone Visual ROI Mask Baseline

## Setup

- Train/val/test frames: 360/80/108
- Image size: 128x128
- Epochs: 6
- Selected threshold: 0.75
- Box IoU threshold: 0.1

## Test performance

| Precision | Recall | F1 | Mask IoU | Mean matched IoU | bits/frame | JPEG/semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4341 | 0.1986 | 0.2725 | 0.3424 | 0.3725 | 2388.3 | 503.6x |

## Interpretation

This is the first trainable visual-semantic extractor in the project. It targets ROI/occupancy semantics for adaptive UAV image transmission, not fine-grained object category semantics. Its output can drive whether to transmit only object/ROI semantics or request ROI image patches when spectrum resources are clean.

Checkpoint: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_roi_mask.pt`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_roi_mask_result.json`
