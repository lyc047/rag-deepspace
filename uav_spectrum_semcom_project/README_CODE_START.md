# 多 UAV 宽带频谱语义通信项目

本目录用于低空 UAV 协同宽带频谱感知语义通信课题，和原始深空遥测项目代码分开维护。

## 硕士论文主线（阶段 0 已固定）

本工程当前的论文主线已经收敛为：

> 面向多无人机协同宽带频谱感知的任务导向数字语义通信与资源优化。

后续核心工作围绕“本地宽带感知—变长数字语义上报—公平数字链路—多 UAV 质量感知融合—频谱资源选择”展开。正式的研究问题、H1—H4、C1—C3、非研究内容、符号表、统一指标和实验映射规则见：

- `docs/THESIS_STAGE0_SCOPE.md`：论文主线的权威范围文档；
- `configs/research_scope.json`：供实验脚本和自动检查使用的机器可读范围配置；
- `configs/experiment_metadata.template.json`：正式实验的元数据模板。

阶段4的研究内容、算法、正负实验结果和证据边界已整理为一份可独立阅读的论文式报告：

- `docs/STAGE4_PAPER_STYLE_OVERVIEW.md`：面向不了解项目的读者，完整介绍研究问题、系统模型、C1—C3、数字链路、实验结果、创新性与局限性。
- `docs/STAGE4_AERPAW_PREFINAL_REPORT.md`：记录AERPAW功率扫频格式核验、永久排除pilot治理、端到端dry-run、完整软件链路复杂度及final适用边界。
- `docs/STAGE4_AERPAW_THREE_SITE_PREFINAL_PLAN.md`：当前有效的AERPAW三站点执行规则，包含2.4 GHz代理冻结、时间对齐、精简抽取、M1失败分支和一次性final前门槛。
- `docs/STAGE4_FINAL_RESULTS_REPORT.md`：已完成的单次AERPAW Final结果、C1负结论、选择性G2正结论及恢复审计。
- `docs/STAGE4_C1_V2_POSTFINAL_OPTIMIZATION.md`：针对Final暴露问题的C1-v2探索性代码优化、三次完整尝试和新独立验证要求。
- `docs/STAGE4_C1_V4_RESULTS.md`：C1-v4任务充分连续块语义与零比特状态回退的开发过程及两轮开发证据。
- `docs/STAGE4_C1_V4_FINAL_RESULTS.md`：使用未复用的2月14—15日CC1/CC2数据完成的单次跨日期Final；H1表示压缩与H2时效保护回退均通过。
- `docs/STAGE4_C1_V4_SUPPLEMENTARY_BASELINES.md`：Final后在已见开发集上补充硬/软表示、ARQ与状态保持基线；用于机制解释，不改变Final。
- `docs/STAGE4_C1_V4_SCALABILITY_EXPERIMENT.md`：在N=8—64和三种连续需求比例下验证任务充分块指示的通信开销扩展规律。
- `docs/STAGE4_THESIS_CHAPTER_FINAL.md`：阶段4正式论文章节，整合方法演进、正负结果、统计协议、创新点和局限性。
- `docs/STAGE4_CLOSEOUT_REPORT.md`：阶段4收尾状态、最终贡献结构和权威材料索引。

## 移动端代码分析交接

离开本机前可运行：

```powershell
python scripts/build_mobile_handoff.py
```

输出位于`mobile_handoff_20260717/`，包含代码文档ZIP、无需解压的源码Codebook、手机端分析提示词、文件清单和SHA256。交接包不包含第三方`data/`、模型checkpoint或实验缓存；范围和使用方法见`MOBILE_HANDOFF_README.md`。

任何主论文实验必须服务于 H1—H4 中至少一项，并同时报告下游资源选择性能和真实通信成本。图像语义模块继续保留，但仅作为系统扩展，不作为论文核心创新或 H1—H4 的主要证据。

正式实验生成结果前，可运行以下命令检查研究假设映射、数据划分、公平预算和信道字段是否齐全：

```powershell
python scripts/validate_experiment_metadata.py configs/experiment_metadata.template.json
```

## 阶段 1：可复现实验工程

阶段 1 已建立依赖锁定、数据角色与 SHA-256 注册表、防泄漏分组划分、一键基线入口和可审计运行记录。完整说明见：

- `docs/STAGE1_REPRODUCIBILITY.md`

快速验证：

```powershell
python scripts/freeze_dataset_registry.py
python scripts/run_stage1_baseline.py
python -m pytest
```

## 已有系统闭环与历史实验

当前项目已经形成“频谱语义感知—多 UAV 资源优化—图像语义传输决策”的软件仿真闭环。该闭环是后续研究的工程基础，其中视觉部分属于扩展验证。历史闭环说明见：

- `docs/SYSTEM_CLOSED_LOOP.md`

其中 DQN/contextual-bandit 决策层已经接入闭环，用于根据频谱 clean rate、packet loss、BER、视觉优先级和候选 payload 大小，自适应选择 `summary_only`、`semantic_only`、`semantic_plus_roi`、`lowres_plus_semantic` 或 `jpeg_full`。它属于已有扩展模块，不作为新论文主线的核心算法贡献。

## 第一阶段不要先写深度模型

第一阶段最重要的是先打通四件事：

1. I/Q 样本统一表示；
2. STFT/PSD 等基础预处理；
3. 原始 I/Q、时频特征、判决结果、语义 token 的 bit 账本；
4. 简单可解释基线，例如能量检测、硬判决、软信息上传。

只有这四件事稳定后，后面的语义编码器、量化器、多 UAV 融合器才有公平比较基础。

## 当前最小代码骨架

```text
uav_spectrum_semcom_project/
  src/spectrum_semcom/
    __init__.py
    types.py          # I/Q 样本、时频框、语义包的数据结构
    synthetic.py      # 小规模合成 I/Q 数据，先用于调通流程
    preprocessing.py  # STFT 与功率谱预处理
    bit_budget.py     # payload/bit 统计
    baselines.py      # 第一批非学习基线
    metrics.py        # IoU、检测指标等
    channels.py       # 上报链路扰动模型
  scripts/
    run_phase1_smoke.py
```

## 推荐实现顺序

### Step 1：先跑通 smoke 实验

```powershell
python uav_spectrum_semcom_project/scripts/run_phase1_smoke.py
```

这个脚本会生成一个小的合成 I/Q 帧，计算 STFT，跑简单能量检测，并输出 bit 账本。

### Step 2：把真实数据接入统一接口

真实数据不要一开始就全量下载。先选小子集，把每个样本转换为 `IQFrame`：

```python
IQFrame(
    iq=complex64_array,
    sample_rate_hz=...,
    center_freq_hz=...,
    boxes=[SignalBox(...)]
)
```

当前已经接入一个小规模真实 SigMF I/Q 样本：

```powershell
python uav_spectrum_semcom_project\scripts\run_real_sigmf_smoke.py
```

运行第一版真实数据能量检测 baseline：

```powershell
python uav_spectrum_semcom_project\scripts\run_real_sigmf_baseline.py
```

输出包括：

- `results/phase1/real_sigmf_energy_baseline_summary.csv`
- `results/phase1/real_sigmf_energy_baseline.json`
- `results/phase1/real_sigmf_energy_baseline_best.png`

生成阶段性 baseline 报告：

```powershell
python uav_spectrum_semcom_project\scripts\make_phase1_baseline_report.py
```

运行轻量 STFT-CNN smoke 训练：

```powershell
python uav_spectrum_semcom_project\scripts\train_tiny_cnn_smoke.py
```

注意：这个脚本只是小数据过拟合/流程验证，不代表最终泛化性能。

如果在 VSCode 中使用 GPU 版 PyTorch，可以显式指定：

```powershell
python uav_spectrum_semcom_project\scripts\train_tiny_cnn_smoke.py --device cuda
python uav_spectrum_semcom_project\scripts\infer_tiny_cnn_full_frame.py --device cuda
```

默认 `--device auto` 会自动检测 CUDA。

当前已确认可用的 GPU 环境：

```powershell
D:\anaconda\envs\pytorch\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

已验证输出：

```text
torch 2.7.1+cu118
cuda True
device NVIDIA GeForce MX450
```

由于 MX450 显存约 2GB，建议保持小模型、小 batch：

```powershell
D:\anaconda\envs\pytorch\python.exe uav_spectrum_semcom_project\scripts\train_frame_cnn.py --epochs 12 --patches-per-frame 64 --batch-size 16 --device cuda
```

运行整帧滑窗推理：

```powershell
python uav_spectrum_semcom_project\scripts\infer_tiny_cnn_full_frame.py
```

生成传统 baseline 与 CNN 的阶段性方法对比：

```powershell
python uav_spectrum_semcom_project\scripts\make_phase1_method_comparison.py
```

生成 frame-level 数据划分清单：

```powershell
python uav_spectrum_semcom_project\scripts\create_sigmf_frame_manifest.py
```

按 frame-level manifest 评估传统 baseline：

```powershell
python uav_spectrum_semcom_project\scripts\run_frame_manifest_baseline.py --split all
python uav_spectrum_semcom_project\scripts\run_frame_manifest_baseline.py --split test
```

在 val split 上调参并固定参数评估 test split：

```powershell
python uav_spectrum_semcom_project\scripts\tune_frame_baseline.py
```

按 frame-level split 训练并评估轻量 CNN：

```powershell
python uav_spectrum_semcom_project\scripts\train_frame_cnn.py --device auto
```

生成 frame-level 阶段报告：

```powershell
python uav_spectrum_semcom_project\scripts\make_frame_level_report.py
```

比较硬判决、软概率、候选框语义包和 feature token 的 payload 开销：

```powershell
python uav_spectrum_semcom_project\scripts\compare_payload_schemes.py
```

生成小规模半真实/可控合成 I/Q frame 数据：

```powershell
python uav_spectrum_semcom_project\scripts\create_synthetic_frame_dataset.py
```

该数据用于扩充训练流程和压力测试，不替代真实数据结论。

验证合成数据 baseline：

```powershell
python uav_spectrum_semcom_project\scripts\run_synthetic_frame_baseline.py --split test
```

训练 RadioML 中型 I/Q 调制分类验证：

```powershell
D:\anaconda\envs\pytorch\python.exe uav_spectrum_semcom_project\scripts\train_radioml_classifier.py --device cuda --epochs 5
```

RadioML 用于验证 class-level semantic payload，不用于时频框检测。

生成 RadioML 中型数据集验证报告：

```powershell
python uav_spectrum_semcom_project\scripts\make_radioml_report.py
```

生成第一阶段总体状态报告：

```powershell
python uav_spectrum_semcom_project\scripts\make_phase1_status_report.py
```

数据集候选记录：

```text
docs/DATASET_CANDIDATES.md
```

数据清单位于：

```text
data/manifest.csv
```

### Step 3：建立第一批 baseline

优先顺序：

1. 完整 I/Q 上传：性能上界、通信开销最大；
2. 能量检测 + 硬判决上传；
3. STFT 量化上传；
4. 简单语义包上传：候选框、类别概率、置信度；
5. 后续再加入神经网络语义 token。

### Step 4：所有方法必须报告 bit 数

不要只报告“压缩率”或“latent 维度”。每个方案至少报告：

- 每帧 payload bits；
- 每秒上报 bitrate；
- 检测/分类/定位指标；
- 上报信道噪声或丢包后的性能变化。

## 硕士论文阶段 2：公平数字链路

阶段 2 已将语义、谱图和理论 IQ 容量参考统一接入分包、CRC、FEC、调制、信道、有限重传和时延预算。详细方法、结论边界和结果说明见：

```text
docs/STAGE2_FAIR_DIGITAL_LINK.md
```

运行包级主实验与波形级一致性校验：

```powershell
python scripts/run_stage2_digital_link.py
```

将公平数字链路接入冻结检测器和资源压力任务：

```powershell
python scripts/evaluate_stage2_task_link.py
```

正式配置位于 `configs/stage2_digital_link.json`，结果位于 `results/stage2/`。阶段 2 的压力聚合不属于关联多 UAV 观测，不能用于证明 H3。

运行阶段 2 衰落鲁棒性与共享时延预算 Pareto 扫描：

```powershell
python scripts/sweep_stage2_robustness_pareto.py
```

增强结果位于 `results/stage2/enhanced/`。

运行量化 PSD 和独立 tile 谱图传统基线：

```powershell
python scripts/evaluate_stage2_classical_baselines.py
```

只在验证集选择 PSD 量化位数，再冻结到测试集：

```powershell
python scripts/select_stage2_psd_quantizer.py
```

结果位于 `results/stage2/classical_baselines/` 和 `results/stage2/psd_quantizer/`。

运行完整冻结本地测试目录（20,001 帧）主实验：

```powershell
python scripts/run_stage2_full_test.py --device auto
```

首次运行生成 detector/PSD 冻结缓存，后续重复链路实验会直接复用。结果位于 `results/stage2/full_test/`。

运行实际 LDPC/BP 波形和完整缓存任务验证：

```powershell
python -m pip install -e ".[research]"
python scripts/validate_stage2_ldpc.py
```

结果位于 `results/stage2/ldpc_validation/`。该实现是规则 LDPC 研究基线，不是 3GPP NR LDPC。
## C1-v3 时域确认（2026-07-21）

C1-v3 对逐场景资源块排序损失增加 CVaR 尾部约束，并在不复用原三站 Final 的条件下完成 60 景校准和 300 景一次性时域确认。结果为码率公平通过，但平均 regret 与 CVaR 均未通过；不得写成正向算法结论。完整方法、统计、负结果和 C1-v4 下一步见：

```text
docs/STAGE4_C1_V3_TEMPORAL_CONFIRMATION.md
```

## C1-v4 任务充分块语义（2026-07-22）

C1-v4 已完成连续频谱块代价语义、版本化编解码、数字链路配对评估器和不复用既有 holdout 的真实开发协议。原始ZIP所在 F 盘当前未挂载，因此真实缓存与性能结果尚未生成。方法、数据边界、运行命令和当前状态见：

```text
docs/STAGE4_C1_V4_DEVELOPMENT.md
```

C1-v4 两轮真实开发结果、分组统计、站点敏感性、创新点和局限见：

```text
docs/STAGE4_C1_V4_RESULTS.md
```
