# 阶段 4 Batch 3 进度：Gate A开发清单冻结

日期：2026-07-16  
状态：Part 1—2完成，Batch 3继续进行  
final holdout：未建立、未访问

## 已完成

依据Batch 2在validation上冻结的8源、8资源块、连续需求4配置，生成`configs/stage4_development_registry_v1.json`：

| Split | 原始目录 | Frame offset | Scene数 | 每scene源数 |
|---|---|---:|---:|---:|
| train | train | 0 | 300 | 8 |
| calibration | train | 2400 | 50 | 8 |
| validation | val | 0 | 150 | 8 |

清单保存每个scene的固定源frame ID、scene hash和split级hash。校验器检查scene重复、frame重复、数量、hash、原始split命名空间及final holdout状态。

## 发现并修复的问题

RadDet的train与val目录都从相同数字stem开始编号。首次生成时若只把stem作为全局ID，会产生大量假性“跨split泄漏”。现改用`source_split:stem`作为全局源ID，例如`train:000000000000`与`val:000000000000`被正确视为不同原始文件；校验器同时强制ID前缀必须匹配声明的原始split。

该修复没有放松真实泄漏检查。train与calibration仍使用不相交offset，validation仍来自独立原始目录。

## 验证

- 开发清单专项测试：2项通过；
- 阶段4协议相关定向测试：通过；
- 全项目：97项通过，用时8.09 s；
- final holdout：不包含、未访问。

## 下一步

### Part 2：冻结缓存

已从清单生成train/calibration/validation三个独立NPZ缓存，保存scene ID、冻结detector的8维基础occupancy、truth occupancy和label-free detector confidence。元数据记录protocol、registry、checkpoint、模型结果和各缓存SHA-256。

| Split | Scenes | 基础平均离散regret | CVaR0.9 | Brier |
|---|---:|---:|---:|---:|
| train | 300 | 0.025184 | 0.144092 | 0.009671 |
| calibration | 50 | 0.018807 | 0.087625 | 0.008779 |
| validation | 150 | 0.022882 | 0.111778 | 0.007611 |

validation的非零平均regret和较高尾部regret说明任务没有重新饱和。缓存位于`results/stage4/gate_a_cache_v1/`。

### Part 2：统一变位宽模型

已实现所有Gate A方法共用的轻量残差occupancy head。初始残差严格为零，因此训练起点保持冻结detector输出。语义精度候选固定为1/2/4/8 bit，训练使用STE离散量化和soft precision选择，推理使用单一argmax精度及整数应用层bit数；G2应用开销与Batch 1 codec一致。

新增缓存和模型测试7项，全项目当前104项通过。

## 下一步

实现四方法同初始化、同seed和同训练预算脚本；仅在train拟合，calibration用于校准，validation用于模型选择。推理输出必须再次通过Batch 1 codec和阶段2链路换算实际发送bit。
