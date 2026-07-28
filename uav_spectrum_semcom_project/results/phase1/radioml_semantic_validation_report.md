# RadioML Medium Dataset Semantic Validation

## 1. 数据集

- 数据集：RadioML 2016.10A
- 大小：压缩包约 212.7 MB，解压后约 225.3 MB
- 样本数：220,000
- 类别数：11
- SNR：-20 dB 到 18 dB，步长 2 dB
- 样本形状：`(2, 128)` I/Q

## 2. 分类语义提取结果

| 实验 | 训练样本 | Val Acc | Test Acc | 说明 |
|---|---:|---:|---:|---|
| 全 SNR | 44000 | 0.4632 | 0.4647 | 包含 -20 到 18 dB，低 SNR 难度高 |
| SNR >= 0 dB | 33000 | 0.7593 | 0.7591 | 中高 SNR 下类别语义更稳定 |

## 3. Payload 开销

| Payload | bit/sample | 相比 12+12 bit I/Q 的压缩倍数 |
|---|---:|---:|
| 原始 I/Q | 3072 | 1.0x |
| hard class semantic | 4 | 768.0x |
| soft probability semantic | 88 | 34.9x |
| 16d/6bit feature token | 145 | 21.2x |

## 4. 阶段性结论

1. RadioML 验证了 class-level I/Q 语义在中高 SNR 下可以被轻量 CNN 提取。
2. 低 SNR 会显著降低语义提取准确率，因此后续必须做 SNR sweep 和鲁棒性建模。
3. 类别语义 payload 的 bit 数远低于原始 I/Q：hard class 约 768x，soft probability 约 34.9x，feature token 约 21.2x。
4. 该数据集验证的是调制类别语义，不验证宽带时频框定位；它与 SigMF/RadDet 类数据互补。
