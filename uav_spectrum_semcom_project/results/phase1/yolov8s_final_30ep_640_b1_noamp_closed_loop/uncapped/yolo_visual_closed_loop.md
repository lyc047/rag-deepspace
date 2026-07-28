# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8s_priority_final_640_b1_noamp_2gb_predict_conf005\labels`
- Confidence threshold: 0.05
- Maximum semantic boxes per frame: unlimited

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.8614 | 127.1 | semantic_only:0.98, semantic_plus_roi:0.02 |
| spectrum_rule | 9475.9 | 1.0007 | 0.5737 | 0.0025 | 0.8614 | 127.1 | semantic_only:0.99 |
| dqn | 9712.3 | 0.9998 | 0.5696 | 0.0029 | 0.8614 | 127.1 | semantic_only:0.98, semantic_plus_roi:0.01 |
| semantic_only | 9360.8 | 0.9991 | 0.5763 | 0.0000 | 0.8614 | 127.1 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
