# YOLOv8n Smoke Training on 2GB GPU

## Setup

- Environment: `D:\anaconda\envs\pytorch\python.exe`
- GPU: NVIDIA GeForce MX450, 2GB VRAM
- Ultralytics: 8.4.90
- Torch: 2.7.1+cu118
- Dataset: `data/processed/visdrone_yolo_priority/visdrone_uav_semantic.yaml`
- Model: `yolov8n.pt`
- Image size: 256
- Batch size: 1
- Epochs: 1
- Workers: 0
- AMP: enabled
- Output directory: `C:\yolo_visdrone_runs\yolov8n_priority_smoke_2gb_lts`

## Compatibility fixes

Three environment issues were found and handled:

1. `python -m ultralytics` is not available in the installed Ultralytics version, so the Python API is used instead.
2. Ultralytics failed to save under the project path containing Chinese characters, so the run output is written to `C:\yolo_visdrone_runs`.
3. The default `polars` package raised `unknown feature flag: 'sse3'`, so it was replaced with `polars-lts-cpu`.

The reusable smoke training script is:

```text
scripts/train_visdrone_yolov8n_smoke.py
```

## Training result

The 1-epoch smoke training completed successfully.

| Item | Result |
|---|---:|
| Peak GPU memory reported by Ultralytics | about 0.15 GB |
| Training images | 411 |
| Validation images | 68 |
| Validation instances | 4,830 |
| Precision | 0.757 |
| Recall | 0.0302 |
| mAP50 | 0.0197 |
| mAP50-95 | 0.00722 |
| `best.pt` size | about 6.2 MB |
| `last.pt` size | about 6.2 MB |

The low accuracy is expected because this is only a 1-epoch smoke test at small image size. The important conclusion is that YOLOv8n training is feasible on the current 2GB GPU under conservative settings.

## Closed-loop evaluation with smoke YOLO predictions

Predictions were exported with confidence threshold 0.05:

```text
C:\yolo_visdrone_runs\yolov8n_priority_smoke_2gb_lts_predict_conf005\labels
```

These predictions were then evaluated in the existing spectrum-aware DQN closed loop:

```text
results/phase1/yolo_smoke_closed_loop/yolo_visual_closed_loop.md
```

Representative result under `semantic_hard_rep3`, packet loss = 0.2:

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 2,403.4 | 0.3907 | 0.2231 | 0.0028 | 0.2522 | 21.7 |
| semantic_only | 1,769.9 | 0.3875 | 0.2257 | 0.0000 | 0.2522 | 21.7 |
| spectrum_rule | 1,769.9 | 0.3875 | 0.2257 | 0.0000 | 0.2522 | 21.7 |

Compared with the previous ROI-mask baseline, this one-epoch YOLO smoke model is only slightly better in visual semantic quality. It should be treated as a feasibility result, not as a final visual detector.

## Storage footprint

| Directory | Size |
|---|---:|
| `C:\yolo_visdrone_runs\yolov8n_priority_smoke_2gb_lts` | 11.82 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_smoke_2gb_lts_predict_conf005` | 0.56 MB |
| `results/phase1/yolo_smoke_closed_loop` | 0.05 MB |

## Conclusion

YOLOv8n can be trained on the current 2GB MX450 GPU if the configuration is conservative. The current smoke model is weak, but it proves that the visual semantic front-end can be upgraded without waiting for new hardware. Longer training can be attempted later with the same settings, but the expected improvement may be limited by image size, batch size, and the very small GPU.

## 5-epoch follow-up test

The epoch count was increased from 1 to 5 while keeping the same 2GB-safe setting:

```text
imgsz = 256
batch = 1
workers = 0
amp = True
```

Training completed successfully. The peak GPU memory reported by Ultralytics was about 0.215 GB, still far below the 2GB limit.

| Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.7576 | 0.0303 | 0.0198 | 0.0073 |
| 5 | 0.6472 | 0.0415 | 0.0296 | 0.0116 |

The 5-epoch model improved recall and mAP, although the detector is still weak for dense small UAV objects.

Predictions were exported with confidence threshold 0.05:

```text
C:\yolo_visdrone_runs\yolov8n_priority_5ep_2gb_predict_conf005\labels
```

Closed-loop result under `semantic_hard_rep3`, packet loss = 0.2:

| Visual front-end | Policy | bits/frame | Utility | Semantic score | Visual quality | Pred boxes |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n 1 epoch | DQN | 2,403.4 | 0.3907 | 0.2231 | 0.2522 | 21.7 |
| YOLOv8n 5 epochs | DQN | 2,451.9 | 0.5097 | 0.2920 | 0.3299 | 25.8 |
| ROI-mask 128 | DQN | 2,489.3 | 0.3466 | 0.2028 | 0.2308 | 28.3 |
| GT annotation upper bound | DQN | 5,635.6 | 1.3794 | 0.7740 | 1.0000 | 70.8 |

The 5-epoch YOLO model is now clearly better than the ROI-mask baseline in closed-loop utility, but it is still far from the GT semantic upper bound. This supports the current implementation strategy: use lightweight YOLO as the deployable visual front-end baseline on the current machine, while treating stronger YOLO training as a later upgrade path.

Additional storage footprint:

| Directory | Size |
|---|---:|
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_2gb` | 11.82 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_2gb_predict_conf005` | 0.67 MB |
| `results/phase1/yolo_5ep_closed_loop` | 0.05 MB |

## 320-pixel input follow-up test

Because the 256-pixel 5-epoch model still missed many dense small objects, the input size was increased from 256 to 320 while keeping the same 2GB-safe batch size:

```text
imgsz = 320
batch = 1
epochs = 5
workers = 0
amp = True
```

Training completed successfully without OOM. The peak GPU memory reported by Ultralytics was about 0.238 GB.

| Setting | Precision | Recall | mAP50 | mAP50-95 | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| 256, 5 epochs | 0.6472 | 0.0415 | 0.0296 | 0.0116 | 0.215 GB |
| 320, 5 epochs | 0.2548 | 0.0601 | 0.0451 | 0.0190 | 0.238 GB |

The 320-pixel model has lower precision but better recall and mAP, which is more useful for the current semantic-transmission task because missed UAV visual objects directly reduce semantic score.

Predictions were exported with confidence threshold 0.05:

```text
C:\yolo_visdrone_runs\yolov8n_priority_5ep_320_2gb_predict_conf005\labels
```

Closed-loop result under `semantic_hard_rep3`, packet loss = 0.2:

| Visual front-end | Policy | bits/frame | Utility | Semantic score | Visual quality | Pred boxes |
|---|---|---:|---:|---:|---:|---:|
| ROI-mask 128 | DQN | 2,489.3 | 0.3466 | 0.2028 | 0.2308 | 28.3 |
| YOLOv8n 1 epoch, 256 | DQN | 2,403.4 | 0.3907 | 0.2231 | 0.2522 | 21.7 |
| YOLOv8n 5 epochs, 256 | DQN | 2,451.9 | 0.5097 | 0.2920 | 0.3299 | 25.8 |
| YOLOv8n 5 epochs, 320 | DQN | 3,080.3 | 0.6078 | 0.3467 | 0.4007 | 33.3 |
| GT annotation upper bound | DQN | 5,635.6 | 1.3794 | 0.7740 | 1.0000 | 70.8 |

The 320-pixel model gives a clear improvement over both ROI-mask and the 256-pixel YOLO models. It was kept as a safe intermediate checkpoint, but later 512/640 tests show that this setting is still too conservative for the current 2GB GPU.

Additional storage footprint:

| Directory | Size |
|---|---:|
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_320_2gb` | 11.83 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_320_2gb_predict_conf005` | 0.87 MB |
| `results/phase1/yolo_5ep_320_closed_loop` | 0.05 MB |

## More aggressive 512/640 follow-up tests

Because the 320-pixel run used only about 0.238 GB GPU memory, two more aggressive settings were tested:

```text
YOLOv8n, imgsz = 512, batch = 2, epochs = 5
YOLOv8n, imgsz = 640, batch = 2, epochs = 5
```

Both settings completed successfully on the NVIDIA MX450 2GB GPU without OOM.

| Setting | Precision | Recall | mAP50 | mAP50-95 | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| 320, batch 1, 5 epochs | 0.2548 | 0.0601 | 0.0451 | 0.0190 | 0.238 GB |
| 512, batch 2, 5 epochs | 0.3220 | 0.1530 | 0.1000 | 0.0490 | 0.625 GB |
| 640, batch 2, 5 epochs | 0.2390 | 0.1940 | 0.1310 | 0.0664 | 0.830 GB |

The larger input sizes substantially improve recall and mAP. This is important for VisDrone-style dense small objects because missed objects directly reduce visual semantic quality in the closed-loop semantic transmission system.

Predictions were exported with confidence threshold 0.05:

```text
C:\yolo_visdrone_runs\yolov8n_priority_5ep_512_b2_2gb_predict_conf005\labels
C:\yolo_visdrone_runs\yolov8n_priority_5ep_640_b2_2gb_predict_conf005\labels
```

Closed-loop result under `semantic_hard_rep3`, packet loss = 0.2:

| Visual front-end | Policy | bits/frame | Utility | Semantic score | Visual quality | Pred boxes |
|---|---|---:|---:|---:|---:|---:|
| ROI-mask 128 | DQN | 2,489.3 | 0.3466 | 0.2028 | 0.2308 | 28.3 |
| YOLOv8n 5 epochs, 320 | DQN | 3,080.3 | 0.6078 | 0.3467 | 0.4007 | 33.3 |
| YOLOv8n 5 epochs, 512, batch 2 | DQN | 3,925.0 | 0.7802 | 0.4414 | 0.5308 | 45.7 |
| YOLOv8n 5 epochs, 640, batch 2 | DQN | 4,871.3 | 0.8948 | 0.5084 | 0.6348 | 60.6 |
| GT annotation upper bound | DQN | 5,635.6 | 1.3794 | 0.7740 | 1.0000 | 70.8 |

The 640-pixel model is now the strongest deployable visual front-end tested on the current machine. Although it transmits more semantic boxes and therefore increases bits/frame, the closed-loop utility still improves, meaning the added visual semantic information is useful rather than redundant.

Additional storage footprint:

| Directory | Size |
|---|---:|
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_512_b2_2gb` | 11.85 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_512_b2_2gb_predict_conf005` | 1.19 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_640_b2_2gb` | 11.87 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_5ep_640_b2_2gb_predict_conf005` | 1.58 MB |
| `results/phase1/yolo_5ep_512_b2_closed_loop` | 0.05 MB |
| `results/phase1/yolo_5ep_640_b2_closed_loop` | 0.05 MB |

## 10-epoch 640-pixel follow-up test

The epoch count was then increased from 5 to 10 while keeping the strongest safe input configuration:

```text
YOLOv8n, imgsz = 640, batch = 2, epochs = 10
```

Training completed successfully on the NVIDIA MX450 2GB GPU. The peak GPU memory reported by Ultralytics was about 0.662 GB.

| Setting | Precision | Recall | mAP50 | mAP50-95 | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| 640, batch 2, 5 epochs | 0.2390 | 0.1940 | 0.1310 | 0.0664 | 0.830 GB |
| 640, batch 2, 10 epochs | 0.2760 | 0.2360 | 0.1580 | 0.0809 | 0.662 GB |

The 10-epoch model improves recall, mAP50 and mAP50-95 compared with the 5-epoch 640 model.

Predictions were exported with confidence threshold 0.05:

```text
C:\yolo_visdrone_runs\yolov8n_priority_10ep_640_b2_2gb_predict_conf005\labels
```

Closed-loop result under `semantic_hard_rep3`, packet loss = 0.2:

| Visual front-end | Policy | bits/frame | Utility | Semantic score | Visual quality | Pred boxes |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n 5 epochs, 640, batch 2 | DQN | 4,871.3 | 0.8948 | 0.5084 | 0.6348 | 60.6 |
| YOLOv8n 10 epochs, 640, batch 2 | DQN | 6,658.8 | 0.9719 | 0.5535 | 0.7432 | 85.4 |
| GT annotation upper bound | DQN | 5,635.6 | 1.3794 | 0.7740 | 1.0000 | 70.8 |

The 10-epoch model improves closed-loop utility, semantic score and visual quality, so increasing epochs is useful. However, it also produces more boxes than the GT annotation average and increases semantic payload above the GT upper-bound payload. This indicates that the next step should not simply keep increasing epochs. A better next step is to tune the detection confidence threshold or add a top-k/priority filtering rule so that the system keeps useful object semantics while suppressing redundant low-confidence boxes.

Additional storage footprint:

| Directory | Size |
|---|---:|
| `C:\yolo_visdrone_runs\yolov8n_priority_10ep_640_b2_2gb` | 11.87 MB |
| `C:\yolo_visdrone_runs\yolov8n_priority_10ep_640_b2_2gb_predict_conf005` | 2.23 MB |
| `results/phase1/yolo_10ep_640_b2_closed_loop` | 0.05 MB |

Current recommendation: use `YOLOv8n, imgsz=640, batch=2, epochs=10` as the best visual-semantic front-end tested so far. The next controlled experiment should tune `conf` and top-k filtering before further increasing epochs or switching to a larger model.

## Semantic packet budget sweep

The 10-epoch/640-pixel predictions were reused to test semantic-packet filtering without retraining. All rows use `semantic_hard_rep3`, packet loss = 0.2, and the DQN policy.

| Packet filtering | Pred boxes | bits/frame | Utility | Semantic score | Visual quality |
|---|---:|---:|---:|---:|---:|
| No cap, conf=0.05 | 85.4 | 6,658.8 | 0.9719 | 0.5535 | 0.7432 |
| Confidence only, conf=0.08 | 63.4 | 4,983.4 | 0.9352 | 0.5328 | 0.6685 |
| Confidence only, conf=0.15 | 41.3 | 3,656.2 | 0.8155 | 0.4606 | 0.5458 |
| conf=0.05 + top-k=40 | 37.1 | 3,176.4 | 0.8059 | 0.4613 | 0.5453 |
| conf=0.05 + top-k=60 | 52.0 | 4,253.5 | 0.9126 | 0.5206 | 0.6349 |
| conf=0.05 + top-k=70 | 58.3 | 4,708.6 | 0.9310 | 0.5307 | 0.6636 |
| **conf=0.05 + top-k=80** | **63.8** | **5,105.5** | **0.9613** | **0.5476** | **0.6861** |

The top-k=80 packet budget lowers the semantic payload by 23.3% compared with the uncapped model while retaining 98.9% of its closed-loop utility. It is therefore the current deployment recommendation. The evaluator now supports `--max-boxes-per-frame`, enabling later spectrum- and queue-aware dynamic packet budgets.

An additional inference-side NMS test changed the IoU threshold from 0.70 to 0.50 at `conf=0.05`. It produced identical mean box count, visual quality and closed-loop metrics, both with and without the top-k=80 cap. The current excess payload is therefore not dominated by highly overlapping boxes removable by this NMS change; top-k packet budgeting remains the useful control knob.

## Final 2 GB visual training: YOLOv8s

The final detector was trained with `YOLOv8s, imgsz=640, batch=1, epochs=30`. AMP was disabled because the MX450 produced NaN losses with mixed precision for this model; full-precision training was stable and used about 1.25 GB of GPU memory. The best checkpoint was saved at epoch 27.

| Model | Precision | Recall | mAP50 | mAP50-95 | Weight size |
|---|---:|---:|---:|---:|---:|
| YOLOv8n, 640, batch 2, 10 epochs | 0.2760 | 0.2360 | 0.1580 | 0.0809 | 6.2 MB |
| **YOLOv8s, 640, batch 1, 30 epochs, AMP off** | **0.4872** | **0.3553** | **0.3443** | **0.1834** | **21.5 MB** |

The larger detector produces many low-confidence candidates at `conf=0.05`, so packet budgeting remains essential. Closed-loop DQN results under `semantic_hard_rep3`, packet loss = 0.2:

| Packet filtering | Pred boxes | bits/frame | Utility | Semantic score | Visual quality |
|---|---:|---:|---:|---:|---:|
| YOLOv8s uncapped | 127.1 | 9,712.3 | 0.9998 | 0.5696 | 0.8614 |
| YOLOv8s top-k=60 | 54.2 | 4,361.5 | 0.9713 | 0.5539 | 0.6810 |
| **YOLOv8s top-k=80** | **69.5** | **5,474.9** | **1.0305** | **0.5870** | **0.7471** |

The final recommended visual-semantic configuration is `YOLOv8s, 640, batch=1, AMP off, best.pt + conf=0.05 + top-k=80`. Compared with the prior YOLOv8n top-k=80 baseline, it raises closed-loop utility from 0.9613 to 1.0305 with only a 7.2% increase in semantic payload. This is the strongest real visual front-end and packetization setting tested within the 2 GB constraint.
