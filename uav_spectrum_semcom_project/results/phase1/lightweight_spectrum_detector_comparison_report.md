# Lightweight Spectrum Detector Comparison Report

## 1. 目的

本次实验尝试将当前频谱语义提取方法从：

```text
TinyOccupancyCNN mask segmentation
```

升级为更接近近两年时频目标检测论文思路的：

```text
YOLO-style lightweight grid detector
```

对比对象包括：

1. `TinyOccupancyCNN mask baseline`；
2. `TinyGridDetectorCNN 8×8 grid`；
3. `TinyGridDetector16CNN 16×16 grid`。

其中 16×16 grid 是本次新增的轻量化检测模型，相比旧 8×8 grid 保留了更多空间细节，适合细长或小尺度时频目标。

## 2. 检测性能对比

| 方法 | Precision | Recall | F1 | Mean IoU | Class Acc | Hard bits/frame |
|---|---:|---:|---:|---:|---:|---:|
| TinyOccupancyCNN mask | 0.4389 | 0.6260 | 0.5160 | 0.5033 | - | 262.0 |
| TinyGridDetectorCNN 8×8 | 0.4316 | 0.4399 | 0.4357 | 0.4651 | 0.2070 | 246.9 |
| TinyGridDetector16CNN 16×16 | 0.5128 | 0.5426 | 0.5273 | 0.4822 | 0.2071 | 248.3 |

结论：

- 旧 8×8 YOLO-style detector 明显弱于 mask baseline；
- 新增 16×16 detector 将 F1 提升到 0.5273，略高于 mask baseline 的 0.5160；
- 16×16 detector 的 precision 更高，说明误检减少；
- mask baseline 的 recall 更高，说明它更保守、更容易覆盖真实占用区域；
- 类别准确率仍然只有约 0.207，说明当前轻量 YOLO-style 仍不适合作为可靠类别语义方法；
- hard semantic bits/frame 仍保持在 250 bit 左右，与 mask baseline 同量级。

因此，从纯检测指标看：

```text
TinyGridDetector16CNN 已经可以作为比原 8×8 grid 更好的轻量 YOLO-style 频谱语义提取模型。
```

## 3. 下游资源优化对比

进一步将 mask baseline 和 16×16 YOLO-style detector 接入同一个多 UAV 资源优化场景：

- packet loss = 0.2；
- BER = 1e-4；
- 4 UAV observations / decision；
- 600 decision steps / trial；
- 3 trials。

| Detector | Scheme | Clean rate | Net utility | Regret | bits/decision |
|---|---|---:|---:|---:|---:|
| mask baseline | semantic_hard | 0.8911 | 0.9895 | 0.0035 | 1050.2 |
| mask baseline | semantic_hard_rep3 | 0.9300 | 0.9918 | 0.0012 | 3150.5 |
| mask baseline | semantic_soft_rep3 | 0.9300 | 0.9918 | 0.0012 | 3935.7 |
| grid16 YOLO-style | semantic_hard | 0.8361 | 0.9863 | 0.0066 | 990.7 |
| grid16 YOLO-style | semantic_hard_rep3 | 0.8739 | 0.9881 | 0.0049 | 2972.0 |
| grid16 YOLO-style | semantic_soft_rep3 | 0.8672 | 0.9884 | 0.0046 | 3539.2 |

这里出现一个关键现象：

```text
16×16 YOLO-style detector 的检测 F1 更高，
但下游资源优化 clean rate 低于 mask baseline。
```

这说明在本项目中，检测器的评价不能只看 box-level F1。

资源优化任务更关心：

```text
是否保守地覆盖真实占用频段，
是否避免选择受干扰的资源块。
```

mask baseline 虽然误检多一些，但 recall 更高，对频谱资源选择更安全；YOLO-style detector 更精确，但漏检会导致系统误判某些资源块为空闲，从而降低 clean rate。

## 4. 研究意义

这个结果很有价值，因为它说明：

> 面向频谱资源优化的语义提取方法，不能简单照搬通用目标检测的 F1/mAP 指标，而应该采用任务导向评价。

也就是说，本项目可以强调：

```text
频谱语义通信的目标不是还原完美检测框，
而是在低开销条件下支持正确的频谱资源决策。
```

因此，后续更合适的优化方向不是单纯提高检测 F1，而是设计：

```text
resource-aware spectrum semantic detector
```

也就是在训练或阈值选择时直接优化：

- clean channel rate；
- low-interference rate；
- regret；
- net utility；
- missed-occupancy penalty。

## 5. 当前推荐结论

短期内，频谱主系统仍建议使用：

```text
TinyOccupancyCNN mask baseline
```

作为主方法，因为它在资源优化闭环中表现更好。

同时，将新增的：

```text
TinyGridDetector16CNN YOLO-style detector
```

作为先进轻量检测结构的对比方法，说明：

- YOLO-style 可以提升 box-level 检测 F1；
- 但如果不加入资源感知损失，未必提升下游资源选择性能；
- 这为后续创新提供了方向。

## 6. 后续优化建议

下一步不建议简单继续堆模型，而应该做任务导向优化：

1. 调低 YOLO-style 阈值，提高 recall，牺牲部分 precision；
2. 在损失函数中增加漏检惩罚；
3. 引入 frequency-block occupancy loss；
4. 直接用资源块 clean / occupied 标签辅助训练；
5. 训练一个 hybrid 模型：

```text
YOLO-style box head + occupancy mask head
```

这样既保留 YOLO-style 的框语义和置信度，又保留 mask 方法对资源占用的保守覆盖能力。

## 7. 相关文件

- 新增模型：`src/spectrum_semcom/models.py` 中的 `TinyGridDetector16CNN`
- 训练脚本：`scripts/train_raddet_grid_detector.py`
- 16×16 detector 结果：`raddet_grid16_detector_result.json`
- 资源优化对比：`spectrum_detector_resource_comparison.md`
- 对比脚本：`scripts/compare_spectrum_detectors_resource.py`
