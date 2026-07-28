# 阶段4 Final scene级指标接口

状态：访问前冻结  
schema：`stage4_final_scene_metrics_v1`  
统计引擎：`stage4_final_statistics_v1`

## 1. 统计输入边界

统计引擎不读取原始RF、标签或模型checkpoint，只接收冻结推理入口输出的scene级聚合指标。每个独立scene在每个主检验族、每个方法中只能出现一次。两个主检验族必须使用完全相同的scene集合，至少200个。

C1 scene级指标须先在该scene内对五个预注册训练seed和全部冻结链路重复求均值；选择性G2须先在scene内对预注册信道条件和链路重复求均值。不得只选有利seed、Eb/N0、信道或重复。`actual_bits`必须包含协议头、CRC、FEC、填充、反馈和重传。

统计引擎在独立scene层面执行：

- 10,000次配对scene bootstrap生成95% CI；
- C1的CVaR0.9在scene级regret分布上计算；
- 单侧配对随机化检验生成component p值；
- 每个intersection-union主检验族取component p值最大值；
- 两个family p值使用Holm校正；
- family只有在全部CI门槛和Holm调整p值不高于0.05时通过。

## 2. JSON结构

```json
{
  "schema_version": "stage4_final_scene_metrics_v1",
  "registry_sha256": "64-hex",
  "access_receipt_sha256": "64-hex",
  "families": {
    "C1_resource_loss": {
      "rows": [
        {"scene_id": "scene-001", "method": "detection_only", "regret": 0.01, "actual_bits": 1024.0},
        {"scene_id": "scene-001", "method": "detection_plus_resource", "regret": 0.008, "actual_bits": 1025.0}
      ]
    },
    "selective_G2_digital_reporting": {
      "rows": [
        {"scene_id": "scene-001", "method": "all_G2", "regret": 0.01, "actual_bits": 2300.0},
        {"scene_id": "scene-001", "method": "selective_G2", "regret": 0.011, "actual_bits": 1800.0}
      ]
    }
  }
}
```

以上数值仅展示schema，不能作为实验数据。模板不得放入正式结果目录。

## 3. 完整性拒绝条件

任一条件满足即拒绝运行且不得悄悄修补：

- scene少于200；
- 方法名称不是四个冻结名称之一；
- 同一scene-method重复；
- 同一family两个方法的scene集合不同；
- 两个family的scene集合不同；
- regret或actual_bits缺失、非有限或为负；
- registry/access receipt哈希与单次访问状态不一致；
- access count不是1；
- schema或统计引擎版本不匹配。

## 4. 正式命令

只有冻结推理入口已生成scene级JSON且访问状态已经原子变为1后才能运行：

```powershell
python scripts/run_stage4_final_statistics.py --input <final_scene_metrics.json>
```

正式统计已经完成，输出为`results/stage4/final_holdout_v1/final_statistics_result.json`；access count为1，不得再次运行确认性Final。冻结推理入口及其访问前`--preflight`记录继续用于审计。合成自检结果位于`results/stage4/final_statistics_selftest_v1/`，仅验证统计代码分支，不能替代正式结果。
