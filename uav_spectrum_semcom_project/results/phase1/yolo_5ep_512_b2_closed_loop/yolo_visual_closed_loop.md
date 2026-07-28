# YOLO Visual Semantics Closed-Loop Evaluation

## Purpose

This report evaluates YOLO-format visual semantic predictions inside the existing spectrum-aware DQN closed loop. It is designed for future lightweight YOLO detectors trained on the prepared VisDrone priority dataset.

- Prediction directory: `C:\yolo_visdrone_runs\yolov8n_priority_5ep_512_b2_2gb_predict_conf005\labels`
- Confidence threshold: 0.05

## Representative result

| Policy | bits/frame | Utility | Semantic score | Priority detail | Visual quality | Pred boxes | Action mix |
|---|---:|---:|---:|---:|---:|---:|---|
| oracle_policy | 5511.7 | 1.3796 | 0.7774 | 0.0070 | 0.5308 | 45.7 | semantic_only:0.98, semantic_plus_roi:0.02 |
| dqn | 3925.0 | 0.7802 | 0.4414 | 0.0044 | 0.5308 | 45.7 | semantic_only:0.96, semantic_plus_roi:0.03 |
| spectrum_rule | 3631.3 | 0.7778 | 0.4457 | 0.0017 | 0.5308 | 45.7 | semantic_only:0.99 |
| semantic_only | 3502.5 | 0.7759 | 0.4470 | 0.0000 | 0.5308 | 45.7 | semantic_only:1.00 |

## Interpretation

The adapter expects YOLO label/prediction text files named `<frame_stem>.txt`. Each line may be either `class x y w h` or `class x y w h confidence`. Classes follow the priority dataset mapping: pedestrian, people, car, van, truck, bus, motor. The closed-loop metric then evaluates whether the predicted visual semantics improve task utility under the spectrum-aware transmission policy.
