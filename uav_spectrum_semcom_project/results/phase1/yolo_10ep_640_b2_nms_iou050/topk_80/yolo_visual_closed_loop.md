# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8n_priority_10ep_640_b2_2gb_predict_conf005_iou050\labels`
- Confidence threshold: 0.05
- Maximum semantic boxes per frame: 80

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.6861 | 63.8 | semantic_only:0.98, semantic_plus_roi:0.02 |
| dqn | 5105.5 | 0.9613 | 0.5476 | 0.0030 | 0.6861 | 63.8 | semantic_only:0.98, semantic_plus_roi:0.02 |
| spectrum_rule | 4965.6 | 0.9607 | 0.5496 | 0.0029 | 0.6861 | 63.8 | semantic_only:0.99 |
| semantic_only | 4805.8 | 0.9594 | 0.5523 | 0.0000 | 0.6861 | 63.8 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
