# Payload Scheme Comparison

## 1. 对比说明

本表不改变检测结果，只比较同一组预测框在不同上报 payload 表示下的通信开销。

| 方法 | Payload 方案 | 平均预测框数 | 平均 bit/frame | 平均速率 | 平均 frame F1 |
|---|---|---:|---:|---:|---:|
| frame_cnn | `feature_token_16d_6bit` | 2.33 | 571.7 | 57.167 kbps | 0.6111 |
| frame_cnn | `feature_token_32d_8bit` | 2.33 | 945.0 | 94.500 kbps | 0.6111 |
| frame_cnn | `hard_box` | 2.33 | 377.0 | 37.700 kbps | 0.6111 |
| frame_cnn | `semantic_box_current` | 2.33 | 395.7 | 39.567 kbps | 0.6111 |
| frame_cnn | `soft_box_8class` | 2.33 | 526.3 | 52.633 kbps | 0.6111 |
| tuned_traditional | `feature_token_16d_6bit` | 6.33 | 1083.7 | 132.308 kbps | 0.7152 |
| tuned_traditional | `feature_token_32d_8bit` | 6.33 | 2097.0 | 248.123 kbps | 0.7152 |
| tuned_traditional | `hard_box` | 6.33 | 665.0 | 82.476 kbps | 0.7152 |
| tuned_traditional | `semantic_box_current` | 6.33 | 715.7 | 88.266 kbps | 0.7152 |
| tuned_traditional | `soft_box_8class` | 6.33 | 1070.3 | 128.802 kbps | 0.7152 |

## 2. 阶段性结论

1. `hard_box` 是最低开销方案，但只适合地面端不需要复核、不需要类别概率的场景。
2. `soft_box_8class` 增加类别概率，适合多 UAV 融合时做软信息融合。
3. `feature_token` 开销高于纯候选框，但仍是 kbps 级，后续可作为语义通信主线。
4. 当前结果再次说明：即便上传更丰富的语义 token，速率仍远低于 12+12 bit I/Q 的 184.32 Mbps。
