# 面向低空 UAV 的频谱感知多模态语义通信系统后续改进计划

## 1. 当前项目状态

当前项目已经形成两个基础模块：

```text
频谱语义模块：
RadDet 时频图 → 频谱占用语义 → 语义上报 → 链路扰动 → 多 UAV 融合 → 频谱资源选择

图像语义模块：
VisDrone UAV 图像 → 视觉目标/ROI 语义 → 语义 payload 统计 → 初步传输策略仿真
```

其中频谱部分已经完成第一版闭环，并补充了公平大载荷传输基线、链路条件 sweep、多 UAV 频谱地图融合评估和轻量 YOLO-style 检测器对比。

图像部分已经完成 VisDrone 数据接入、视觉语义 payload 统计、零样本检测 baseline 和轻量 ROI-mask baseline，但当前视觉模型性能仍偏弱。

因此，后续目标不是简单增加模块，而是把系统推进为：

```text
频谱语义感知 + 图像语义提取 + 自适应传输决策 + 多模态任务性能评估
```

## 2. 总体改进路线

建议后续按照四条主线推进：

1. 频谱语义提取方法升级；
2. UAV 图像语义模型增强；
3. 频谱驱动的图像语义传输决策层；
4. 更真实的系统仿真与论文级实验整理。

整体路线为：

```text
阶段 A：完善频谱语义方法
阶段 B：提升图像语义模型
阶段 C：构建规则 + DQN 决策层
阶段 D：联合频谱和图像语义做多模态闭环
阶段 E：整理论文实验、消融和可视化
```

## 3. 阶段 A：频谱语义提取方法升级

### 3.1 当前问题

当前主频谱方法是：

```text
TinyOccupancyCNN mask baseline
```

其优点是：

- recall 较高；
- 资源选择 clean rate 好；
- 语义开销低；
- 适合频谱占用保守覆盖。

但它也有不足：

- 方法较简单；
- 不如 YOLO/Transformer 类方法新；
- 难以自然输出类别和置信度；
- 框语义来自连通区域后处理，不是端到端检测框。

本项目已经尝试了：

```text
TinyGridDetector16CNN YOLO-style detector
```

检测 F1 略高于 mask baseline，但资源选择 clean rate 反而下降。这说明频谱语义方法不能只看检测 F1，而要面向下游资源优化任务。

### 3.2 改进方向 A1：Hybrid mask + YOLO-style 双头模型

推荐设计：

```text
共享 CNN backbone
├── occupancy mask head
└── YOLO-style box/confidence/class head
```

目的：

- mask head 保证频谱占用覆盖和高 recall；
- YOLO head 输出规整框、置信度和可选类别；
- 两者共同生成更稳定的频谱语义。

损失函数：

```text
L = L_mask
  + λ1 L_objectness
  + λ2 L_box
  + λ3 L_class
  + λ4 L_resource
```

其中 `L_resource` 是后续重点，可以让模型直接关注资源块占用判断。

短期目标：

| 指标 | 当前 mask | 当前 grid16 | 目标 |
|---|---:|---:|---:|
| Detection F1 | 0.5160 | 0.5273 | ≥ 0.58 |
| Recall | 0.6260 | 0.5426 | ≥ 0.65 |
| Clean rate | 0.9308 左右 | 0.8739 左右 | ≥ 0.93 |
| bits/frame | 约 262 | 约 248 | 同量级 |

### 3.3 改进方向 A2：Resource-aware loss

当前训练只优化检测指标，后续应加入资源块占用损失：

```text
真实频谱框 → 真实资源块占用向量
预测频谱框/mask → 预测资源块占用向量
L_resource = BCE/MSE(predicted_occupancy, true_occupancy)
```

这样可以直接优化：

- 资源块是否空闲；
- 是否会误选受干扰频段；
- clean rate；
- regret；
- net utility。

这比单纯优化 mAP/F1 更符合本项目的语义通信目标。

### 3.4 改进方向 A3：近两年方法参考

可参考方向：

- YOLOv8/YOLOv9 用于时频目标检测；
- DFN-YOLO 类宽带窄带信号检测；
- 多分辨率 spectrogram YOLO；
- Transformer-based time-frequency localization；
- Wideband spectrum detector with attention。

但考虑当前硬件，优先采用：

```text
轻量 YOLO-style / Hybrid CNN
```

而不是直接上重 Transformer。

## 4. 阶段 B：UAV 图像语义模型增强

### 4.1 当前问题

当前图像部分已有：

1. VisDrone payload 统计；
2. COCO 预训练 Faster R-CNN zero-shot baseline；
3. TinyOccupancyCNN visual ROI-mask baseline。

当前性能：

| 方法 | Precision | Recall | F1 | 说明 |
|---|---:|---:|---:|---|
| COCO zero-shot detector | 0.3408 | 0.1167 | 0.1738 | 小目标召回太低 |
| Visual ROI-mask | 0.4341 | 0.1986 | 0.2725 | 可作为 ROI 语义 baseline |

当前图像模型不能作为最终方案，主要问题是：

- VisDrone 小目标密集；
- 128×128 下采样损失细节；
- TinyOccupancyCNN 表达能力不足；
- ROI-mask 不输出类别；
- 召回率偏低。

### 4.2 改进方向 B1：YOLOv8n / YOLOv5n 微调 VisDrone

推荐作为图像语义主线升级。

步骤：

```text
VisDrone 标注 → YOLO 格式转换
→ 小规模 train/val split
→ YOLOv8n 或 YOLOv5n 微调
→ 输出目标类别 + 框 + 置信度
→ 编码为视觉语义 packet
```

优点：

- 更适合目标检测；
- 输出语义天然适合通信；
- 可以评价 mAP、F1、召回；
- 论文可解释性更强。

硬件建议：

```text
imgsz = 320 或 416
batch = 2 或 4
epochs = 20~50
device = cuda
```

短期目标：

| 指标 | 当前 ROI-mask | 目标 |
|---|---:|---:|
| F1 | 0.2725 | ≥ 0.45 |
| Recall | 0.1986 | ≥ 0.45 |
| JPEG/semantic | 503.6x | ≥ 100x |

### 4.3 改进方向 B2：提高 ROI-mask 输入分辨率

如果暂时不安装 YOLO，可以继续优化 ROI-mask：

```text
128×128 → 256×256
```

同时改进：

- 使用 RGB 输入；
- 增加轻量 U-Net skip connection；
- 增加 focal loss；
- 对小目标加权；
- 增加数据增强。

适合目标：

```text
提升 ROI 区域召回率，
用于判断哪些区域值得补传。
```

### 4.4 改进方向 B3：网格级视觉语义

不用检测每个小目标，而是把图像分成网格：

```text
16×16 或 32×32 grid
```

每个格子预测：

- 是否有目标；
- 目标密度；
- 是否为高优先级区域；
- 是否需要 ROI 补传。

这种方法更接近语义通信，因为它直接服务传输决策，而不是追求标准目标检测指标。

## 5. 阶段 C：频谱驱动的图像语义传输决策层

### 5.1 当前策略

当前决策层是规则策略：

```text
频谱资源好：
    传视觉语义 + ROI 图像块

频谱资源一般：
    只传视觉语义

频谱资源差：
    只传高优先级摘要
```

它不是强化学习。

当前优点：

- 简单；
- 可解释；
- 适合作为 baseline。

当前不足：

- 阈值手工设定；
- 不能自适应复杂状态；
- 没有长期收益建模；
- 不能处理队列、延迟、历史频谱状态等序贯因素。

### 5.2 改进方向 C1：DQN 离散动作决策

该问题非常适合建模为 MDP。

状态：

```json
{
  "spectrum_clean_rate": 0.93,
  "spectrum_occupancy": [0.1, 0.3, 0.7, 0.2],
  "packet_loss": 0.1,
  "ber": 0.0001,
  "net_utility": 0.99,
  "visual_priority": 1,
  "num_objects": 12,
  "roi_area_ratio": 0.18,
  "queue_length": 3,
  "last_action": 1
}
```

动作：

```text
0: 不传图像，只传状态摘要
1: 只传视觉语义
2: 传视觉语义 + ROI
3: 传低分辨率图像 + 视觉语义
4: 传完整 JPEG
```

奖励：

```text
R = α × task_success
  + β × priority_detail_delivered
  - γ × bits_used
  - δ × latency
  - η × failed_transmission
  - μ × action_switch_cost
```

推荐方法：

```text
DQN / Double DQN / Dueling DQN
```

原因：

- 动作离散；
- 实现简单；
- 训练开销小；
- 适合当前电脑；
- 便于和规则策略对比。

### 5.3 改进方向 C2：规则策略作为 baseline

即使后续做 DQN，也必须保留规则策略作为对照：

```text
Rule-based policy
Random policy
Always semantic-only
Always ROI
Always JPEG
DQN policy
Oracle policy
```

评价指标：

- 任务成功率；
- 高优先级目标信息到达率；
- 平均 bits/frame；
- 平均延迟；
- packet loss 下鲁棒性；
- 单位 bit 任务收益；
- 频谱资源冲突率。

## 6. 阶段 D：多模态语义闭环

最终系统应形成：

```text
频谱语义：
    频谱占用框 / 资源块状态 / clean rate / net utility

图像语义：
    目标框 / 类别 / ROI 区域 / 任务优先级

决策层：
    根据频谱状态和视觉任务重要性选择传输粒度

通信层：
    语义 packet / ROI patch / 低分辨率图 / JPEG

评价层：
    任务性能 + 通信开销 + 链路鲁棒性
```

建议建立统一语义包格式：

```json
{
  "uav_id": 1,
  "timestamp": 12345,
  "spectrum_semantics": {
    "resource_block": 2,
    "clean_rate": 0.93,
    "occupancy": [0.1, 0.2, 0.8, 0.3]
  },
  "visual_semantics": {
    "priority": "high",
    "objects": [
      {"class": "car", "bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.86}
    ],
    "roi_regions": [...]
  },
  "decision": {
    "mode": "semantic_plus_roi",
    "estimated_bits": 12000
  }
}
```

## 7. 阶段 E：更真实的仿真与实验完善

### 7.1 频谱真实性增强

后续可补：

- SDR 小规模实测；
- GNU Radio 仿真；
- ns-3 通信链路；
- UAV/地面站链路模型；
- 更真实的 packet scheduling；
- ARQ/FEC 更具体实现。

### 7.2 图像真实性增强

后续可补：

- 完整 VisDrone train set；
- UAVDT 数据集；
- DroneVehicle 数据集；
- 视频帧时序；
- 目标跟踪语义；
- 图像语义过期指标。

### 7.3 指标体系完善

最终论文建议指标：

| 模块 | 指标 |
|---|---|
| 频谱检测 | F1、IoU、mAP、recall |
| 频谱资源优化 | clean rate、regret、net utility |
| 频谱地图 | occupancy MAE、clean-block accuracy |
| 图像检测 | precision、recall、F1、mAP |
| 视觉语义传输 | semantic bits、ROI bits、JPEG bits |
| 多模态决策 | task success、bits、delay、utility |
| 链路鲁棒性 | packet loss sweep、BER sweep |

## 8. 推荐近期任务顺序

建议近期按以下顺序执行：

### Task 1：频谱 hybrid detector

```text
实现 mask + YOLO-style 双头模型
加入 resource-aware occupancy loss
对比 mask baseline / grid16 detector
```

输出：

- 检测 F1；
- clean rate；
- net utility；
- bits/frame。

### Task 2：图像 YOLO 微调

```text
转换 VisDrone 到 YOLO 格式
训练 YOLOv8n 或 YOLOv5n
输出视觉目标级语义
```

输出：

- mAP / F1 / recall；
- semantic bits；
- JPEG/semantic ratio。

### Task 3：规则多模态决策正式化

```text
定义 state/action/reward-like utility
实现 rule-based policy
比较 semantic-only / ROI / JPEG / adaptive
```

输出：

- bits；
- priority detail rate；
- task utility。

### Task 4：DQN 决策层

```text
构建轻量仿真环境
训练 DQN
与规则策略对比
```

输出：

- DQN vs rule；
- 不同 packet loss / spectrum pressure 下性能。

### Task 5：论文图表整理

生成：

- 频谱检测对比图；
- 频谱资源 clean rate 图；
- 图像语义 payload 图；
- 多模态决策 trade-off 曲线；
- bits vs task performance 曲线。

## 9. 预期最终贡献点

最终项目可以形成以下贡献：

1. 提出面向低空 UAV 的频谱感知多模态语义通信框架；
2. 设计低开销频谱占用语义提取与资源优化闭环；
3. 构建视觉语义和频谱语义联合传输策略；
4. 引入规则/DQN 决策层，实现频谱状态驱动的视觉传输粒度选择；
5. 在公开频谱/图像数据集和仿真链路下验证任务性能与通信开销优势。

## 10. 风险与备选方案

| 风险 | 备选方案 |
|---|---|
| YOLOv8n 显存不足 | 使用 YOLOv5n、降低 imgsz/batch，或继续 ROI-grid 模型 |
| 图像检测性能仍低 | 改用网格级 ROI 语义，不追求每个目标框 |
| DQN 收敛慢 | 先用规则策略 + 贪心/动态规划 |
| 频谱 YOLO-style clean rate 不如 mask | 保留 mask 为主方法，YOLO 作为对比，发展 hybrid |
| 真实 UAV 数据不足 | 明确采用公开数据 + 仿真扩展，后续补 SDR 小实验 |
