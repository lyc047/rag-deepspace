# Phase 3 Visual Semantic Start Report

## 1. 目标

在频谱系统闭环完成后，本阶段开始接入 UAV 图像语义，目标是形成如下链路：

```text
UAV 图像
→ 图像语义提取
→ 视觉语义 packet
→ 根据频谱资源状态决定传输粒度
→ 图像语义 / ROI / 整图之间自适应选择
```

当前阶段先完成最小可行图像语义验证，不直接追求完整多类别高精度检测。

## 2. 数据集

当前使用 VisDrone2019-DET validation set：

- 图像数量：548；
- 数据量：约 82 MB；
- 图像来源：无人机视角；
- 标注类型：目标框、类别、遮挡、截断；
- 主要类别：pedestrian、people、bicycle、car、van、truck、bus、motor 等。

前期 payload 统计显示：

| Payload | Mean bits/frame | Ratio |
|---|---:|---:|
| JPEG image | 1,185,557.3 | 223.5x vs ground-truth visual semantics |
| Raw RGB 8-bit image | 23,063,124.1 | 4346.9x vs ground-truth visual semantics |
| Ground-truth visual semantic boxes | 5,305.6 | 1.0x |

这说明 UAV 图像任务确实存在明显的数据量瓶颈，目标级语义具备显著压缩潜力。

## 3. 零样本 COCO 检测器 baseline

首先使用 torchvision 的 COCO 预训练轻量检测器：

```text
fasterrcnn_mobilenet_v3_large_320_fpn
```

它没有在 VisDrone 上微调，只作为“现成预训练模型能否直接用于 UAV 图像语义”的 baseline。

阈值 sweep 结果如下：

| Score threshold | Precision | Recall | F1 | pred boxes/frame | semantic bits/frame | JPEG/semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.1415 | 0.1465 | 0.1439 | 50.8 | 3866.6 | 291.9x |
| 0.05 | 0.1803 | 0.1398 | 0.1575 | 38.1 | 2949.5 | 382.7x |
| 0.10 | 0.2662 | 0.1276 | 0.1725 | 23.5 | 1902.8 | 593.2x |
| 0.15 | 0.3408 | 0.1167 | 0.1738 | 16.8 | 1418.6 | 795.7x |
| 0.25 | 0.4626 | 0.0930 | 0.1548 | 9.9 | 919.1 | 1228.1x |
| 0.35 | 0.5827 | 0.0736 | 0.1307 | 6.2 | 655.4 | 1722.2x |
| 0.50 | 0.7563 | 0.0537 | 0.1004 | 3.5 | 460.1 | 2453.2x |

最佳 F1 出现在 threshold = 0.15：

```text
Precision = 0.3408
Recall    = 0.1167
F1        = 0.1738
```

结论：

- COCO 预训练模型能生成低开销视觉语义；
- 但对无人机小目标召回很低；
- 不能作为最终视觉语义提取器；
- 后续需要 VisDrone 专用训练或微调。

## 4. 轻量视觉 ROI-mask 模型

为了快速形成图像语义闭环，本阶段训练了一个轻量 ROI/occupancy mask 模型：

```text
TinyOccupancyCNN visual ROI mask
```

输入：

```text
128×128 灰度 UAV 图像
```

输出：

```text
128×128 ROI/目标占用 mask
```

该模型不预测精细类别，而是回答：

```text
图像哪些区域存在目标，哪些区域值得补传 ROI 图像块？
```

训练设置：

| 项目 | 设置 |
|---|---:|
| Train frames | 360 |
| Val frames | 80 |
| Test frames | 108 |
| Epochs | 6 |
| Image size | 128×128 |
| Selected threshold | 0.75 |
| Box IoU threshold | 0.1 |

测试结果：

| Precision | Recall | F1 | Mask IoU | Mean matched IoU | bits/frame | JPEG/semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4341 | 0.1986 | 0.2725 | 0.3424 | 0.3725 | 2388.3 | 503.6x |

结论：

- 该模型性能仍然初步，尤其召回率仍需提升；
- 但相比零样本 COCO 检测器，F1 从约 0.174 提升到约 0.273；
- 输出语义开销仍非常低，约 2.39 kbit/frame；
- 相比 JPEG 图像，约小 503.6 倍；
- 当前更适合定位为“视觉 ROI 语义提取器”，而不是“完整目标检测器”。

## 5. 当前图像语义内容

目前图像语义分两类：

### 5.1 标注级目标语义

来自 VisDrone ground truth：

```json
{
  "class": "car",
  "bbox": [x, y, width, height],
  "truncation": 0,
  "occlusion": 1
}
```

适合做理论上限、payload 统计和策略验证。

### 5.2 模型预测 ROI 语义

来自 TinyOccupancyCNN：

```json
{
  "semantic_type": "visual_roi_occupancy",
  "roi_regions": [
    {
      "x_min": 0.12,
      "y_min": 0.35,
      "x_max": 0.28,
      "y_max": 0.47
    }
  ]
}
```

适合驱动：

- 是否需要补传 ROI 图像块；
- 哪些区域比整图更值得传；
- 在频谱资源紧张时是否只传 ROI/目标摘要。

## 6. 与频谱系统的连接方式

频谱系统输出：

```json
{
  "clean_channel_rate": 0.9308,
  "mean_net_utility": 0.9917,
  "recommended_resource": "lowest_occupancy_block"
}
```

图像语义系统输出：

```json
{
  "visual_roi_regions": [...],
  "semantic_bits": 2388.3,
  "jpeg_bits": 1202725.4
}
```

下一步多模态策略可以写成：

```text
频谱资源好：
    传视觉语义 + ROI 图像块

频谱资源一般：
    传视觉 ROI/目标语义

频谱资源差：
    只传高优先级视觉摘要
```

## 7. 当前判断

图像语义部分已经完成第一步：

```text
真实 UAV 图像数据接入
→ 视觉语义 payload 统计
→ 零样本检测 baseline
→ 轻量 ROI-mask 训练 baseline
→ 初步性能评估
```

但图像部分还没有达到频谱部分那样完整。当前最需要继续提升的是：

1. 提高视觉 ROI/目标召回率；
2. 从 ROI-mask 过渡到目标框 + 粗类别语义；
3. 将模型预测视觉语义接入频谱感知多模态传输策略；
4. 评估任务性能、图像语义开销和频谱资源状态之间的联合关系。

## 8. 相关文件

- 零样本检测脚本：`scripts/infer_visdrone_torchvision.py`
- 阈值 sweep 脚本：`scripts/sweep_visdrone_torchvision_thresholds.py`
- ROI-mask 训练脚本：`scripts/train_visdrone_roi_mask.py`
- 零样本阈值结果：`visdrone_torchvision_threshold_sweep.md`
- ROI-mask 结果：`visdrone_roi_mask_result.md`
- ROI-mask 权重：`visdrone_roi_mask.pt`
