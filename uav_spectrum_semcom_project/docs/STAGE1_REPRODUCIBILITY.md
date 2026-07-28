# 阶段 1：可复现实验环境、数据划分与基线入口

> 状态：已完成第一版工程固化  
> 上游范围：`docs/THESIS_STAGE0_SCOPE.md`

## 1. 阶段目标

阶段 1 不追求新增模型性能，而是保证后续论文实验能够回答“使用了什么代码、什么数据、什么划分、什么环境和什么公平预算”。本阶段完成：

1. Python 包和依赖声明；
2. 测试环境的精确版本快照；
3. 数据角色、划分规则、SHA-256 和样本数登记；
4. 源帧/采集序列先划分、派生信道样本后生成的防泄漏规则；
5. 正式实验元数据校验；
6. 一键可运行的最小基线与审计记录。

## 2. 环境安装

推荐使用 Python 3.12.7。核心测试环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

`requirements-lock.txt` 是当前已验证环境的精确快照；`pyproject.toml` 是项目的兼容版本范围和可编辑安装入口。视觉扩展需要时再安装：

```powershell
python -m pip install -e ".[vision]"
```

视觉依赖不属于频谱主线的核心测试前置条件。

当前共享的全局 Python 环境中，`pip check` 还会报告既有 `streamlit` 与 `protobuf/rich` 的版本冲突；它们不属于本项目核心依赖，也未影响本项目测试。正式论文实验必须使用上述独立 `.venv`，不要把共享全局环境作为最终复现环境。

## 3. 数据角色与划分结论

数据政策由 `configs/dataset_policy.json` 定义，冻结结果写入 `configs/dataset_registry.json`。

| 数据 | 角色 | 正式结论资格 | 当前处理 |
|---|---|---:|---|
| 短 SigMF 5G 录制 | 真实 I/Q 接口验证 | 否 | 14 帧来自同一录制，不能把帧级 train/val/test 当作独立泛化证据 |
| Synthetic controlled IQ | 管线诊断与可控 sweep | 否 | 可用于 SNR/信道单元测试，不能替代主数据证据 |
| RadDet 本地数据 | 主频谱基准 | 是，有限制 | 冻结本地 `data.yaml` 指定的目录划分；实测 test/train/val 为 20,001/14,001/6,001，未经来源核验不重命名；明确其为合成雷达基准而非 UAV 实测 |
| VisDrone | 可选视觉扩展 | 不证明频谱主假设 | 按采集序列前缀分组，禁止相邻帧跨 train/val/test |

冻结或复查数据登记表：

```powershell
python scripts/freeze_dataset_registry.py
```

该命令会：

- 计算 CSV manifest 的 SHA-256；
- 统计 split 样本数；
- 检查相同 frame ID 是否跨 split；
- 根据数据政策检查采集源/文件级 group 是否跨 split；
- 记录 RadDet 各目录帧数和文件名列表指纹；
- 对正式可用数据发现泄漏时返回失败。

## 4. 防泄漏原则

固定顺序为：

```text
原始录制/序列/场景分组
→ train/val/test 划分
→ 在每个 split 内生成 SNR、信道、丢包和码率派生样本
```

禁止顺序为：

```text
原始帧 × 多个信道条件
→ 随机打散
→ train/val/test
```

后者会使同一原始帧以不同信道版本同时进入训练集和测试集。已有视觉 contextual-bandit 脚本已经改为按 VisDrone 采集序列前缀分组切分，并在结果中保存每个 split 的帧 ID 和序列 ID。

## 5. 正式实验记录

复制模板：

```text
configs/experiment_metadata.template.json
```

至少填写：

- H1—H4 映射；
- split ID 和 manifest checksum；
- 配置与代码版本；
- 随机种子；
- 方法或基线；
- 公平预算类型和值；
- 感知信道和上报信道；
- 指标与重复次数。

校验命令：

```powershell
python scripts/validate_experiment_metadata.py path\to\metadata.json
```

## 6. 一键最小复现实验

运行：

```powershell
python scripts/run_stage1_baseline.py
```

该入口会：

1. 校验阶段 0 实验元数据；
2. 确认数据注册表存在；
3. 调用冻结的 synthetic energy-detector smoke baseline；
4. 记录 Python、平台和核心包版本；
5. 记录命令、运行时间、stdout/stderr；
6. 记录配置、数据注册表和结果文件 SHA-256；
7. 写入 `results/stage1/reproducibility_smoke/run_record.json`。

该 smoke 实验只证明环境和记录链路可复现，不作为 H1 的正式论文证据。

## 7. 阶段 1 完成条件

- [x] `pyproject.toml` 与精确依赖快照已建立；
- [x] 数据政策和冻结注册表已建立；
- [x] 短 SigMF 单录制的结论边界已明确；
- [x] 派生样本随机划分泄漏已修复；
- [x] 正式实验元数据模板和校验器已建立；
- [x] 一键最小复现实验入口已建立；
- [x] 自动测试覆盖数据分组和阶段 0/1 约束。

进入阶段 2 前仍需新增与主论文直接相关的公平数字链路配置，包括 packetization、CRC、FEC、重传和统一时延预算。
