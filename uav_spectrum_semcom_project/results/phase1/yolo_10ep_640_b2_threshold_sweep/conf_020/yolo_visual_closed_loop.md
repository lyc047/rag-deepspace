# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8n_priority_10ep_640_b2_2gb_predict_conf005\labels`
- Confidence threshold: 0.2

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.4804 | 33.3 | semantic_only:0.98, semantic_plus_roi:0.02 |
| dqn | 3199.8 | 0.7358 | 0.4137 | 0.0064 | 0.4804 | 33.3 | semantic_only:0.94, semantic_plus_roi:0.05 |
| spectrum_rule | 2775.0 | 0.7296 | 0.4193 | 0.0020 | 0.4804 | 33.3 | semantic_only:0.99, semantic_plus_roi:0.01 |
| semantic_only | 2609.6 | 0.7273 | 0.4207 | 0.0000 | 0.4804 | 33.3 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
