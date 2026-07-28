# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8n_priority_5ep_320_2gb_predict_conf005\labels`
- Confidence threshold: 0.05

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.4007 | 33.3 | semantic_only:0.98, semantic_plus_roi:0.02 |
| dqn | 3080.3 | 0.6078 | 0.3467 | 0.0034 | 0.4007 | 33.3 | semantic_only:0.95, semantic_plus_roi:0.05 |
| spectrum_rule | 2663.6 | 0.6048 | 0.3491 | 0.0005 | 0.4007 | 33.3 | semantic_only:0.99 |
| semantic_only | 2607.6 | 0.6041 | 0.3494 | 0.0000 | 0.4007 | 33.3 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
