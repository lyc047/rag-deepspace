# VisDrone Visual Semantic Payload Report

## Dataset

- Dataset: VisDrone2019-DET-val
- Frames: 548
- Mean image size: 1291.1 × 726.2
- Mean boxes/frame: 70.79
- Median boxes/frame: 65.0
- Nonempty frame ratio: 1.000

## Payload comparison

| Payload | Mean bits/frame | Ratio vs visual semantics |
|---|---:|---:|
| JPEG image | 1185557.3 | 223.5x |
| Raw RGB 8-bit image | 23063124.1 | 4346.9x |
| Visual semantic boxes | 5305.6 | 1.0x |

## Class distribution

| Class | Count |
|---|---:|
| pedestrian | 8844 |
| people | 5125 |
| bicycle | 1287 |
| car | 14064 |
| van | 1975 |
| truck | 750 |
| tricycle | 1045 |
| awning-tricycle | 532 |
| bus | 251 |
| motor | 4886 |
| others | 32 |

## Interpretation

This report establishes the visual side of the later multimodal semantic system. For UAV visual monitoring tasks, transmitting object-level semantics can be much smaller than forwarding JPEG or raw RGB images, but this only preserves task-level information rather than reconstructable imagery.

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_payload_summary.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\visdrone_payload_summary.json`
