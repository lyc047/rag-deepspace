# Fiesta Spectrum Crowdsensing Payload Report

## 1. 数据定位

- 数据集：Fiesta spectrum crowdsensing dataset
- 来源：https://www.kaggle.com/datasets/neutrinoliu/fiesta
- 文件数：48
- 设备数：8，设备：berserker, caster, gglPixel2, lancer, pikachu, rider, saber, saki
- 中心频点：525, 575, 625, 675, 725, 775 MHz
- 单帧格式：latitude、longitude、timestamp + 256 个 PSD bin

## 2. 当前语义化方法

本脚本把每个 PSD 快照中高于“该快照中位数 + margin”的连续频率 bin 合并为占用事件。该方法不是最终检测器，而是用于验证现实频谱测量中：如果任务只需要占用频段、峰值功率和位置时间元数据，完整上传 256-bin 频谱并非总是必要。

## 3. Payload 统计

| 指标 | 数值 |
|---|---:|
| 帧数 | 190587 |
| 事件总数 | 873334 |
| 平均事件/帧 | 4.582 |
| 非空帧比例 | 0.869 |
| 原始 float32 频谱 bits/帧 | 8288.0 |
| 原始 int16 频谱 bits/帧 | 4192.0 |
| 语义事件 bits/帧 | 380.0 |
| float32 原始/语义压缩倍数 | 21.81x |
| int16 原始/语义压缩倍数 | 11.03x |

## 4. 对主项目的意义

1. Fiesta 不是完整 I/Q 数据，但它证明了真实频谱众包/移动频谱测量存在大量连续频谱读数上传场景。
2. 当下游任务是频谱占用监测、异常频段告警或低空平台频谱地图更新时，事件级语义比完整频谱读数更贴近任务目标。
3. 该结果可作为论文中“现实数据容量瓶颈”的辅助证据；主线验证仍应以 SigMF/RadDet/真实 SDR I/Q 的时频目标检测为核心。

CSV: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\fiesta_payload_summary.csv`
JSON: `C:\Users\Lenovo\Desktop\项目\rag-deepspace\uav_spectrum_semcom_project\results\phase1\fiesta_payload_summary.json`
