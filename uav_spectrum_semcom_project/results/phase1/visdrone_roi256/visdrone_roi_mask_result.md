# VisDrone Visual ROI Mask Baseline

## Setup

- Train/val/test frames: 360/80/108
- Image size: 256x256
- Epochs: 10
- Selected threshold: 0.55
- Box IoU threshold: 0.1

## Test performance

| Precision | Recall | F1 | Mask IoU | Mean matched IoU | bits/frame | JPEG/semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2277 | 0.1900 | 0.2071 | 0.3139 | 0.3749 | 4437.0 | 271.1x |

## Interpretation

This is the first trainable visual-semantic extractor in the project. It targets ROI/occupancy semantics for adaptive UAV image transmission, not fine-grained object category semantics. Its output can drive whether to transmit only object/ROI semantics or request ROI image patches when spectrum resources are clean.

Checkpoint: `uav_spectrum_semcom_project\results\phase1\visdrone_roi256\visdrone_roi_mask.pt`
JSON: `uav_spectrum_semcom_project\results\phase1\visdrone_roi256\visdrone_roi_mask_result.json`
