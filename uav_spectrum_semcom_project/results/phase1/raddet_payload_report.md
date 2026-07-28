# RadDet 128 Payload and Annotation Report

## 1. 数据定位

- 数据集：RadDet40k128HW001Tv2
- 来源：https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025
- 许可证：CC-BY-NC-SA-4.0
- 数据形态：128×128 时频图 PNG、YOLO 目标框标签、每帧 metadata
- metadata 显示典型原始序列长度为 1,000,000 complex I/Q samples，采样率为 500 MHz。

## 2. Split 规模

| split | frames |
|---|---:|
| train | 14001 |
| val | 6001 |
| test | 20001 |

## 3. 类别分布

| class | boxes |
|---|---:|
| Rect | 1858 |
| Barker | 1850 |
| Frank | 1883 |
| P1 | 1846 |
| P2 | 1782 |
| P3 | 1825 |
| P4 | 1793 |
| Px | 1788 |
| ZadoffChu | 1789 |
| LFM | 1817 |
| FMCW | 1810 |

## 4. Payload 对比

| 指标 | 数值 |
|---|---:|
| 帧数 | 40003 |
| 有标签帧数 | 20041 |
| 平均目标框/帧 | 0.501 |
| PNG 压缩图像 bits/帧 | 116541.3 |
| 8-bit 128×128 谱图 bits/帧 | 131072.0 |
| 12+12 bit I/Q bits/帧 | 24000000.0 |
| hard box semantic bits/帧 | 245.1 |
| PNG/语义压缩倍数 | 475.54x |
| 8-bit 谱图/语义压缩倍数 | 534.83x |
| 12-bit I/Q/语义压缩倍数 | 97930.77x |

## 5. 对主项目的意义

1. RadDet 比 RadioML 更贴近本项目主线，因为任务是时频目标检测，而不是单窗口调制分类。
2. 该数据集可用于训练/验证从宽带观测中提取目标框、类别、置信度等任务语义。
3. 当前 payload 统计只证明“标签级语义开销极低”；下一步必须训练检测器并评估不同 SNR 下的 detection F1。

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\raddet_payload_summary.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\raddet_payload_summary.json`
