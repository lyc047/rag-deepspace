# Frame-Level Phase 1 Report

## 1. 实验设置

- 数据：短 5G NR SigMF 真实 I/Q 录制
- 切分方式：按 SigMF 中的 `Frame N` 标注切分为 frame-level 样本
- 划分：train 8 frames，val 3 frames，test 3 frames
- 目的：避免在同一整帧上随机采 patch 后又在同一整帧测试，建立更规范的第一阶段基准

## 2. 方法对比

| 方法 | 参数选择 | Val F1 | Test Precision | Test Recall | Test F1 | Test 平均语义速率 |
|---|---|---:|---:|---:|---:|---:|
| Tuned traditional baseline | `freq_median`, sigma=5.0, min_cells=32 | 0.7619 | 0.5263 | 0.7692 | 0.6250 | 88.266 kbps |
| Tiny STFT-CNN | frame-level smoke training, device=cuda | 0.5217 | 1.0000 | 0.5385 | 0.7000 | 52.646 kbps |

## 3. 关键结论

1. 传统 `freq_median + connected components` 在当前小数据上仍是最可靠 baseline。
2. Tiny STFT-CNN 已完成 train/val/test 工程闭环，但小配置下尚未超过传统 baseline。
3. 这说明学习式方法后续需要更多训练样本、更合理的标签设计和 GPU 训练，而不是简单增加网络复杂度。
4. 当前 test 上传统语义候选框速率仍是 kbps 级，而 12+12 bit I/Q 是约 184.32 Mbps，容量瓶颈和语义压缩价值仍然成立。

## 4. 下一步建议

- 在 VSCode 的 GPU PyTorch 环境中重新运行 `train_frame_cnn.py --device cuda`，把 epoch 和 patches-per-frame 加大。
- 增加更多小规模真实/半真实 I/Q 数据，不能只依赖一条 5G SigMF 录制。
- 训练前先固定传统 baseline，后续学习式方法必须超过 tuned baseline 才作为主线结果。
- 下一阶段加入 payload 类型对比：硬判决、软概率、候选框语义包、局部 feature token。
