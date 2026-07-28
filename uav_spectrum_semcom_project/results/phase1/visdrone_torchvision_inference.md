# VisDrone Torchvision Visual Semantic Inference

## Setup

- Detector: torchvision fasterrcnn_mobilenet_v3_large_320_fpn COCO pretrained
- Frames: 60
- Score threshold: 0.05
- IoU threshold: 0.5

## Detection performance

| Precision | Recall | F1 | Matched class acc | Mean matched IoU | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2019 | 0.1629 | 0.1803 | 0.7914 | 0.6787 | 441 | 1743 | 2267 |

## Payload

| Payload | Mean bits/frame | Ratio |
|---|---:|---:|
| JPEG image | 1247287.7 | 440.8x vs predicted semantics |
| Predicted visual semantic boxes | 2829.8 | 1.0x |
| Ground-truth visual semantic boxes | 3458.6 | - |

## Interpretation

The zero-shot COCO detector provides a first model-generated visual-semantic baseline. Because VisDrone contains many small aerial-view objects, recall is expected to be limited without dataset-specific fine-tuning. This result is mainly used to establish the next training/fine-tuning target and to quantify the payload of predicted visual semantic packets.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_torchvision_inference.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_torchvision_inference.json`
