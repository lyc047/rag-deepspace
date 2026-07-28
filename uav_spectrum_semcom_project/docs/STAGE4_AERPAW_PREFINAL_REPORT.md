# 阶段4 AERPAW 数据到位前准备报告（W15历史记录）

> 状态更新：LW1整包、20个永久排除pilot和后续三站点协议已经完成；CC1/CC2仍在下载。当前有效执行规则以[三站点final前计划](STAGE4_AERPAW_THREE_SITE_PREFINAL_PLAN.md)为准，本文件保留W15首次识别`rf32_le`语义和完整链路复杂度的历史记录。  
> 证据边界：没有注册、读取或运行 final holdout；`final_access_count=0`。

## 1. 工作目标

本批工作解决四个数据到位前问题：明确 AERPAW 数据的真实物理含义；建立严格且可测试的输入适配器；冻结永久排除 pilot 的治理流程；补齐从冻结检测器到连续资源选择的完整软件链路复杂度。工作不修改 C1、C2、C3 权重、阈值、预算或 final 统计协议。

## 2. 数据格式核验与重要修正

Dryad 数据集 DOI 为 `10.5061/dryad.hmgqnk9zn`。官方说明指出，每个文件对对应一次约 87 MHz--6 GHz 的扫频，包含 98,868 个频率 bin；`.sigmf-data` 保存 little-endian 32-bit 浮点功率值，而不是复数基带 I/Q。

在下载 ZIP 的已持久化前段中解析出的真实元数据进一步确认：

- `core:datatype = rf32_le`；
- `dataset:site = LW1`；
- `dataset:num_bins = 98868`；
- 频率轴为 87.1600--6019.1802 MHz，步长约 0.06 MHz；
- 首个核验 capture 使用带 UTC 偏移的 `core:datetime`；
- 三条已完整解压的功率向量均为有限 float32，实测范围约为 -141 至 -54 dBm。

因此，原有面向复数 I/Q 的 `load_sigmf_frame()` 不得用于本数据集。本批新增独立 `rf32_le` 功率扫频适配层，避免把相邻两个功率值错误组成 I/Q 样本。

## 3. 新增实现

### 3.1 功率扫频适配器

`src/spectrum_semcom/aerpaw_spectrum.py` 实现：

1. `rf32_le`、文件长度、bin 数、频率跨度、有限值和单调频率轴的严格校验；
2. 忽略 `__MACOSX` 资源分叉并检查 `.sigmf-meta/.sigmf-data` 成对完整性；
3. 将 98,868 bin 聚合为冻结任务所需的 8 个等宽资源频道；
4. 使用显式 `threshold_dbm` 生成频道占用比例，禁止隐式从 final 调阈值；
5. 计算最小平均功率的连续资源块；
6. 按时间戳等距、在查看模型输出前生成永久排除 pilot 清单。

### 3.2 数据治理工具

- `configs/aerpaw_lw1_pilot_protocol_v1.json`：记录 DOI、官方哈希、物理语义、pilot 允许/禁止用途和 final 兼容边界。
- `scripts/audit_aerpaw_spectrum_dataset.py`：对已解压目录进行配对、schema和抽样数值审计。
- `scripts/prepare_aerpaw_lw1_after_download.py`：下载完成后执行官方 SHA-256、ZIP 路径穿越检查、安全解压、数据审计和20-scene永久排除 pilot 冻结；不会生成 final catalog。

### 3.3 下载器可靠性修复

原 curl 多分片在网络重试时会截断该分片的既有内容。本批替换为 `download_lw1_resumable.py`：每条 Range 请求从分片当前持久化长度继续，连接失败不会回退已下载字节；完成后合并并核验 Dryad SHA-256 `595e...1951`。

## 4. 端到端接口 dry-run

`scripts/run_aerpaw_stage4_dry_run.py` 使用确定性、与真实 AERPAW 尺寸一致的 98,868-bin 合成功率扫频，依次运行：

```text
rf32功率扫频 -> 8频道占用 -> 冻结C1 -> G2语义包 -> Hamming数字链路 -> 解码 -> 连续资源块选择
```

100次单线程运行的中位/p95时延为：

| 环节 | 中位时延 | p95时延 |
|---|---:|---:|
| 功率扫频到8频道占用 | 0.647 ms | 1.193 ms |
| detection-only与resource C1两次推理 | 2.844 ms | 4.700 ms |
| 两个G2 codec与数字链路 | 2.516 ms | 3.650 ms |
| 连续资源选择 | 0.158 ms | 0.285 ms |

该结果只证明接口、维度、bit账本和解码路径一致，不是算法性能结果。两个冻结 C1 在该域外合成输入上都选择1-bit并产生相同硬占用，不能解释为外部泛化成功或失败。

## 5. 完整 development 软件链路复杂度

`scripts/profile_stage4_full_chain.py` 使用注册表中的首个 validation scene、8张 RadDet 谱图、冻结检测器和冻结 C1，在单线程 CPU 上预热20次、测量100次：

| 环节 | 中位时延 | p95时延 |
|---|---:|---:|
| 读取8张图并max-hold | 11.258 ms | 13.767 ms |
| 归一化与tensor构造 | 0.399 ms | 0.765 ms |
| 冻结occupancy detector | 24.149 ms | 29.148 ms |
| 8频道pool与C1 | 1.842 ms | 2.649 ms |
| selective-G2的5条消息codec与链路 | 4.194 ms | 5.752 ms |
| 融合与连续资源选择 | 0.172 ms | 0.248 ms |
| **完整软件链路** | **42.157 ms** | **51.678 ms** |

该次dry-run每个决策产生519 application bit和1918实际信道bit。检测器/C1参数量分别为21,201/1,740。结果不包含SDR采集、射频前端、设备间传输和嵌入式板端能耗。

## 6. AERPAW 对阶段4 final 的适用边界

| 要求 | LW1数据现状 | 判断 |
|---|---|---|
| 真实宽带频谱资源代价 | 真实单站点功率扫频 | 可用于pilot和资源代价代理 |
| 原始复数I/Q外部验证 | 只有扫频功率 | 不支持 |
| 独立人工occupancy标签 | deposit未提供 | 必须在pilot冻结代理标签规则并降低声明强度 |
| C1单节点外部资源任务 | 可映射为8频道，但存在域差异 | 有条件支持，需先完成pilot适配审计 |
| selective G2冻结四节点链路 | LW1只有一个接收站 | LW1单包不支持 |
| H3真实空间协作 | 单站点无同scene多接收机视图 | 不支持 |

这意味着“有超过200个扫频文件”不等于“自动满足当前 final 协议”。在 pilot 完成前，不得把剩余 LW1 文件批量注册为 final；尤其不能用同一功率向量一边选择阈值、一边报告独立泛化。

## 7. 下载完成后的冻结执行顺序

1. 验证整包大小与官方 SHA-256；
2. 安全解压并完成所有文件对审计；
3. 按预先确定的等距规则生成20个永久排除 pilot scene；
4. 只在 pilot 上确定关注频段、功率到occupancy映射和异常文件规则；
5. 重新审计 C1 输入分布、任务非饱和性及声明范围；
6. 只有协议兼容性成立时才另行生成候选 final catalog；
7. 在此之前保持 final access count 为0。

## 8. 自动验证

新增 `tests/test_aerpaw_spectrum.py`，覆盖正常读取、非I/Q语义、错误datatype、截断文件、缺失配对、频道映射和pilot永久排除状态。定向测试4项全部通过。完整回归结果在本批最终验收后补记。

## 9. 原始资料

- Dryad 数据集主页：<https://doi.org/10.5061/dryad.hmgqnk9zn>
- SigMF 规范：<https://github.com/sigmf/SigMF>
