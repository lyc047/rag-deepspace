# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `uav_spectrum_semcom_project\data\processed\visdrone_yolo_priority\predictions\gt_priority_all`
- Confidence threshold: 0.15

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| dqn | 5522.2 | 1.4031 | 0.7786 | 0.0137 | 1.0000 | 65.5 | semantic_only:0.95, semantic_plus_roi:0.04 |
| spectrum_rule | 5147.0 | 1.3997 | 0.7894 | 0.0096 | 1.0000 | 65.5 | semantic_only:0.98, semantic_plus_roi:0.02 |
| semantic_only | 4925.1 | 1.3899 | 0.7958 | 0.0000 | 1.0000 | 65.5 | semantic_only:1.00 |
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 1.0000 | 65.5 | semantic_only:0.98, semantic_plus_roi:0.02 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
