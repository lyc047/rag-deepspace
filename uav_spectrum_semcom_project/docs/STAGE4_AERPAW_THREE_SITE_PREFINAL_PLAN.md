# 阶段4 AERPAW 三站点 final 前执行计划与冻结记录

> 版本：v1.2（2026-07-21）  
> 状态：Final前门槛和单次Final均已完成；C1未通过、选择性G2通过；final access count = 1。  
> 适用范围：后续AERPAW数据处理、C1条件性外部验证和三站点选择性G2迁移检查。

## 1. 数据事实与研究边界

Dryad数据集`10.5061/dryad.hmgqnk9zn`版本3包含CC1、CC2、LW1三个固定监测站。官方说明给出的载荷是USRP B205mini对约87 MHz--6 GHz的重复功率扫频；每次扫频含98,868个频点，数据为little-endian float32 dBm功率，不是复数I/Q。三个压缩包及官方SHA-256已写入`configs/aerpaw_three_site_prefinal_protocol_v1.json`。

该数据可以提供真实时间、频率和地点变化，但没有独立人工occupancy标签，也没有无人机移动接收机或真实语义上报链路。因此论文只能主张：

1. C1在真实接收功率扫频资源代理上的条件性外部验证；
2. 若三站点满足时间对齐门槛，检验冻结选择性G2在三固定站点上的迁移性；
3. 上报信道仍为冻结数字链路仿真，不能称实测空口；
4. 三固定站点不等于原四节点条件，不证明H3无人机空间协作，也不构成MIMO。

## 2. 已冻结的资源代理

20个LW1 scene已按文件时间轴等距选取，并永久排除出任何final family。pilot manifest SHA-256为`289f90d5...ab7045`。本轮没有导入或执行学习模型，仅使用pilot的功率统计完成以下冻结：

- 频段：2400.0 MHz（含）至2483.5 MHz（不含）的2.4 GHz ISM频段；
- 频道化：8个等宽频道，每频道10.4375 MHz；
- 门限：每次扫频独立计算`median + 3 × 1.4826 × MAD`，鲁棒标准差下限0.25 dB；
- C1输入：每频道超过门限的频点比例；
- 资源代价：每频道平均接收功率dBm；
- 业务需求：选择4个连续频道；
- oracle：平均功率最低的连续4频道。

选择2.4 GHz的理由是其与无人机/WLAN非授权共存和动态选频直接相关，且所有扫频均完整覆盖；并非根据模型效果挑选。per-sweep鲁棒门限用于抵消不同站点的噪声底和增益偏移，不使用其他scene、标签或模型输出。

pilot审计得到：选中频段每扫频1391个频点，门限中位数约-132.809 dBm，总体频道占用中位数约0.03169。所有20个scene均满足有限值、维度和映射范围检查。上述数值仅说明代理没有明显退化，不是模型性能结果。

## 3. 三站点时间对齐与独立scene规则

文件名时间按官方说明解释为`America/New_York`本地时间并转换为UTC。LW1作为anchor，CC1和CC2各选择距离最近且未被使用的扫频：

- 每站相对LW1绝对偏差不超过10秒；
- 三站点最大时间跨度不超过20秒；
- 一个扫频不得重复进入两个scene；
- 不插值、不复制、不用缺失站点填充；
- 任一站缺失则丢弃该scene；
- 三站点频率轴必须在`1e-5 MHz`绝对误差内一致。

对齐后先删除所有与LW1 pilot同stem的scene，再按UTC贪心保留至少间隔30分钟的scene，最后从整个可用时间跨度确定性等距选取200个。30分钟间隔用于降低短时自相关和伪重复风险，但不宣称数学独立性；所有统计仍以配对scene为唯一统计单位，链路重复嵌套在scene内。

## 4. 存储与数据治理方案

F盘当前完整LW1解压目录约55.4 GB。CC1、CC2压缩包合计约39.7 GB，若完整解压会超出剩余空间。冻结方案因此改为：

```text
三个官方ZIP
  -> 只读取ZIP中央目录和文件名时间戳
  -> 三站点对齐、pilot排除、30分钟稀疏化、等距选200个scene
  -> 仅抽取200 × 3个SigMF文件对
  -> schema、功率、频率网格和SHA-256审计
  -> pre-final inventory
```

未入选测量值不会被解压或送入模型。精简副本预计约1--2 GB。三站点精简副本、官方ZIP、pilot manifest和审计结果验证完成前，不删除LW1完整解压目录；之后该目录可作为可选清理项，但不得删除官方ZIP、`final_compact`、`work`或项目内冻结结果。

## 5. 三节点算法兼容与数字链路

冻结C1 head本身按scene逐行推理，不绑定节点数。三节点合成dry-run已验证两种C1检查点均可接受`3 × 8`输入。选择性G2使用三站点全G1加一个阶段3先验节点G2，对照为三站点all-G2；经典均值融合和无报告0.5回退保持不变。

AERPAW没有上报链路测量。final仍使用AWGN、Rayleigh、Rician及全局$E_b/N_0=0/3/6/9$ dB。为避免任意把有利链路条件绑定某个站点，相对偏置`(-1,0,+1)` dB的全部六种站点排列均嵌套在每个scene内，每种排列重复5次。三节点合成演练已完成全部六种排列，证明代码可执行；演练结果不用于算法选型或性能声明。

## 6. 进入final前的硬门槛

只有以下条件全部满足，才能生成候选catalog：

1. 三个ZIP大小和Dryad SHA-256全部匹配；
2. 每个ZIP的SigMF meta/data配对无缺失和路径穿越；
3. 至少得到200个通过30分钟稀疏化的三站点对齐scene；
4. 三站点site字段、`rf32_le`、98,868 bins、频率轴和功率范围全部通过；
5. pilot overlap为0；
6. 精简源文件与代理标签sidecar哈希全部冻结；
7. catalog与development registry不存在scene、event、provenance或源哈希重叠；
8. 代码快照和唯一final统计入口通过完整回归；
9. `configs/stage4_final_access_state.json`仍为`access_count=0`。

catalog注册只允许验证文件、provenance和代理标签派生，不得运行学习模型或查看方法差异。正式推理开始前才消费唯一访问；消费后无论结果正负均不得调参或重跑。

## 7. 预注册失败分支

下载完成后最关键的不确定性不是文件数量，而是三个站点是否有足够时间重叠。执行以下预注册分支：

- **M1通过**：若存在不少于200个满足时间和独立性规则的三站点scene，则C1和三站点选择性G2在同一catalog上一次性执行；选择性G2结论限定为三固定站点迁移检查。
- **M1失败**：若时间不重叠或不足200个，禁止把异步测量拼成协作scene。仅从LW1按同一30分钟与等距规则构建C1单站点final；选择性G2保持development证据，不给出真实多节点final结论。
- **C1代理门失败**：若频率网格、功率范围或代理映射大量异常，则AERPAW不进入final；论文如实报告外部数据不兼容，不用validation冒充final。

## 8. 已实现文件与待执行命令

核心新增文件：

- `configs/aerpaw_three_site_prefinal_protocol_v1.json`：全部参数、哈希、门槛和声明边界；
- `scripts/freeze_aerpaw_pilot_resource_proxy.py`：仅pilot的代理冻结审计；
- `scripts/prepare_aerpaw_three_site_prefinal.py`：ZIP级发现、对齐、稀疏化、精简抽取和质控；
- `scripts/build_aerpaw_final_catalog_from_prefinal.py`：通过pre-final门后生成代理label sidecar和候选catalog；
- `scripts/run_aerpaw_three_site_prefinal_dry_run.py`：三节点合成接口演练；
- `src/spectrum_semcom/aerpaw_final_statistics.py`与`scripts/run_aerpaw_c1_only_final_statistics.py`：M1失败时的预注册单family统计分支；
- `tests/test_aerpaw_spectrum.py`与`tests/test_aerpaw_prefinal_protocol.py`：数据、对齐、ZIP安全、协议和防误触回归。

下载完成后依次执行：

```powershell
python scripts/prepare_aerpaw_three_site_prefinal.py
python scripts/build_aerpaw_final_catalog_from_prefinal.py
python scripts/register_stage4_final_catalog.py --catalog results/stage4/aerpaw_three_site_prefinal_v1/aerpaw_candidate_final_catalog.json
python scripts/freeze_stage4_final_snapshot.py
```

上述前三步均不消费final。只有catalog、完整性、快照和回归全部通过后，才按`STAGE4_FINAL_HOLDOUT_RUNBOOK.md`调用单次访问入口并立即运行冻结推理与统计。

## 9. 当前禁止事项

- 不修改C1/C2/C3权重、checkpoint、损失或预算；
- 不用非pilot真实数据选择频段、门限或scene；
- 不查看非pilot方法输出后排除不利scene；
- 不把功率扫频写成原始IQ；
- 不把三固定站点写成四无人机、MIMO或H3已验证；
- 不完整解压CC1/CC2；
- Final完成后不得再次调用`consume_stage4_final_access.py`或把访问次数改回0。

## 10. 原始资料

- Dryad数据集主页：<https://doi.org/10.5061/dryad.hmgqnk9zn>
- Dryad v2 API（版本与文件SHA-256）：<https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.hmgqnk9zn>
- SigMF规范：<https://github.com/sigmf/SigMF>

## 11. 本批验收结果

- 当前三站点协议SHA-256：`5c6c3ebd33cd5ba3978c1644bc1461d66823c00739e04f03e709c9e6c77fe838`；
- pilot冻结结果与三节点合成dry-run均绑定上述协议哈希；
- LW1、CC1、CC2官方ZIP的文件大小和SHA-256全部复核通过，分别发现21,617、32,529、34,865个完整SigMF文件对；
- 三站点对齐得到13,620个scene；排除pilot重叠后按30分钟稀疏化保留227个，M1门槛通过；
- 按预注册等距规则选取200个scene，抽取1,200个SigMF文件（约1.65 GB），逐scene频率网格、功率范围和代理映射质控通过；
- 已生成200份确定性代理label sidecar和候选catalog，并注册为`configs/stage4_final_registry_v1.json`；真实多接收站scene数为200；
- AERPAW定向数据/协议/统计测试16项通过；
- 全项目168项测试通过；
- `development_acceptance=true`，`failed_item_ids=[]`；
- F01数据物化和F02单次执行均已完成；主检验结果为C1失败、选择性G2通过；
- 正式推理入口为`scripts/run_aerpaw_three_site_final_inference.py`；访问前只读preflight已经通过；
- C1资源后悔固定为所选连续4频道相对最优连续4频道的平均接收功率差，单位dB；选择性G2固定升级LW1，不读取scene真值或模型输出选择节点；
- 可执行快照包含124个文件，SHA-256为`43e6162d2d267246977da8f3ec35fa881fc2db77de8aeab8e705137afbae379f`；
- `stage4_final_access_state.json`为`access_consumed / access_count=1`；原访问回执保持不变。

因此，Final前准备和单次Final均已闭环。后续只允许结果复核、论文制图和明确标记的探索性分析；不再修改算法后使用同一数据声称新的独立Final结果。
