# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8s_priority_final_640_b1_noamp_2gb_predict_conf005\labels`
- Confidence threshold: 0.05
- Maximum semantic boxes per frame: 60

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.6810 | 54.2 | semantic_only:0.98, semantic_plus_roi:0.02 |
| dqn | 4361.5 | 0.9713 | 0.5539 | 0.0026 | 0.6810 | 54.2 | semantic_only:0.98, semantic_plus_roi:0.01 |
| spectrum_rule | 4259.0 | 0.9711 | 0.5561 | 0.0032 | 0.6810 | 54.2 | semantic_only:0.99 |
| semantic_only | 4111.2 | 0.9691 | 0.5593 | 0.0000 | 0.6810 | 54.2 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
