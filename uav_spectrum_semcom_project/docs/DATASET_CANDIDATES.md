# 10GB 以下/优先小规模的数据集候选

## 已接入

### 1. Annotated 5G NR SigMF recording

- 类型：真实 I/Q，SigMF，带 5G 物理信道标注
- 大小：约 8.8 MB
- 用途：真实 I/Q 读取、STFT 时频框检测、候选框语义 payload 验证
- 来源：https://destevez.net/2023/12/an-annotated-5g-sigmf-recording/

### 2. RadioML 2016.10A

- 类型：I/Q 调制识别
- 大小：压缩包约 212.7 MB，解压后约 225.3 MB
- 样本：220,000 条，11 类，20 个 SNR
- 用途：class-level semantic payload 验证
- 来源：https://zenodo.org/records/18397070

### 3. Fiesta spectrum crowdsensing dataset

- 类型：真实移动/众包频谱测量数据，格式为地理位置、时间戳与 256-bin PSD 频谱读数
- 大小：Kaggle 压缩包约 84.2 MB，当前已完整下载
- 规模：48 个频谱文件，8 台设备，6 个中心频点，190,587 帧
- 用途：验证现实频谱测量中“完整频谱读数上传”与“占用事件语义上传”的开销差异；作为低空/移动频谱地图更新、频谱众包监测的数据容量瓶颈侧证
- 当前结论：简单事件语义化后，float32 频谱读数约 21.8x 压缩，int16 频谱读数约 11.0x 压缩
- 注意：Fiesta 不是完整 I/Q 波形，因此不替代 SigMF/RadDet 类 I/Q 或时频检测数据；它更适合作为现实应用动机和辅助实验
- 来源：https://www.kaggle.com/datasets/neutrinoliu/fiesta

### 4. RadDet ICASSP 2025 - RadDet40k128HW001Tv2

- 类型：宽带 RF 目标检测数据；由 500 MHz 频宽、1M I/Q sample 生成时频图并提供边界框/类别标注
- 大小：当前下载 `RadDet40k128HW001Tv2.tar.part-aa`，约 674.9 MB；解压后约 595 MB
- 数据形态：128×128 PNG 时频图、YOLO 标签、metadata；类别包括 Rect、Barker、Frank、P1/P2/P3/P4、Px、ZadoffChu、LFM、FMCW
- 用途：主线验证数据，比 RadioML 更贴近“从宽带观测中提取时频目标语义”的研究目标
- 当前快速统计：抽样 6000 帧，平均 0.507 个目标框/帧；PNG/语义约 474.4x，8-bit 谱图/语义约 533.8x，按 1M complex I/Q、12+12 bit 估算的 I/Q/语义约 97,748.5x
- 注意：上述是“标签级语义”的 payload 上限对比，下一步必须训练检测器并报告 detection F1/SNR 鲁棒性
- 来源：https://www.kaggle.com/datasets/abcxyzi/raddet-icassp-2025

## 后续候选

### 3. HisarMod2019.1

- 类型：调制识别
- 规模：文献中常见描述为 780,000 样本、26 类、SNR -20:2:18
- 用途：比 RadioML 2016.10A 类别更多，可作为调制类别语义泛化验证
- 注意：需要确认可靠下载源和实际大小后再下载

### 4. RadioML 2016.10B

- 类型：调制识别
- 用途：RadioML 2016.10A 的补充验证
- 注意：下载源和格式需要进一步核验

### 5. DeepSense spectrum sensing datasets

- 类型：真实 SDR/USRP 频谱感知数据，面向多 WiFi 信道占用检测
- 用途：非常贴近“低空平台/边缘设备频谱感知”任务，可作为真实占用检测验证
- 当前状态：GitHub 项目可访问，但其 Northeastern repository 数据下载入口当前返回 403 Forbidden，暂不作为自动下载对象
- 来源：https://github.com/wineslab/deepsense-spectrum-sensing-datasets

## 暂不优先

### RadioML 2018.01A

- 类型：大规模调制识别
- 规模：公开说明中常见为 2,000,000 条，每条 1024 samples，HDF5
- 问题：体积较大，当前磁盘和 MX450 训练条件下不优先
