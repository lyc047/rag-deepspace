# 阶段 S6R.3h：确认锁箱协议冻结与无信号 Dry-run 报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-07-30
- Verification Status：VERIFIED
- Version Label：S6R-LOCKBOX-PREFLIGHT-v1

## 1. 结论

S6R-FH10-v1 的六站点支持性确认协议已经注册、实现、测试并冻结。无信号值 dry-run 正式通过，候选具备一次性显式执行资格。

当前确认锁箱访问次数仍为0，储备访问次数仍为0。本阶段没有打开或加载任何锁箱 NPY 数组，也没有运行算法指标。

## 2. 确认角色与站点

确认角色固定为 `confirmation_lockbox`，包含以下六个在原始元数据拆分时就已确定的站点：

1. Oreland；
2. PiSDR1；
3. Skap_French_Riviera；
4. Geneva；
5. URJC1；
6. Princeton1。

六个站点的注册形状均满足至少200个时间场景和64个频率采样点。任何实际访问后发现的无效站点都必须保留失败记录，不得从储备站点替换。

## 3. 冻结实验条件

相对已失败的24站点主 Final，唯一允许变化的字段是固定静默心跳：

| N | 原主 Final 心跳 | 锁箱候选心跳 |
|---:|---:|---:|
| 8 | 10景 | 10景 |
| 16 | 16景 | 10景 |
| 32 | 16景 | 10景 |
| 64 | 10景 | 10景 |

以下内容保持冻结：

- 任务容差 ε = 0.2 dB；
- K = 4任务码本；
- 三种需求比例0.25、0.50、0.75；
- 时间一致性上下文门控；
- 64 bit版本化激活协议；
- 最多两次激活尝试；
- 累计ACK与信念状态恢复；
- 硬regret校验、escape和fail-closed；
- 主故障概率与精确动作基线；
- 所有前向、反馈、心跳、安装、恢复、重复和回退开销；
- 60条配对公共随机轨迹；
- 5000次站点—轨迹分层自举。

## 4. 支持性确认门槛

支持性确认必须同时满足：

1. 6个注册站点全部可评价；若不足6个则判为数据不足，不得替换；
2. 四种 N 的场景加权 clean 95%下界均不低于90%；
3. 四种 N 的场景加权配对总 bit节省95%下界均大于0；
4. 四种 N 的等站点宏平均节省95%下界均大于0；
5. 错误码本动作执行次数为0；
6. 所有 bit账本恒等式成立；
7. 不读取或替换储备站点。

该门槛比“至少三种 N 的宏平均节省为正”更严格，要求四种 N 全部通过。

## 5. 访问与执行保护

执行器新增了独立的治理层，执行前必须验证：

- Pilot访问次数为1；
- 主 Final访问次数为1；
- 确认锁箱访问次数为0；
- 储备访问次数为0；
- 协议哈希与冻结值一致；
- 代码快照与冻结值逐文件一致；
- 六站点名称和顺序与注册表一致；
- 输出结果不存在；
- 命令显式携带 `--consume`。

任一条件不满足，执行器都会在加载锁箱数组前终止。访问声明写入账本后才允许提取六个指定成员，自动重试和覆盖结果均被禁止。

## 6. Dry-run 与测试结果

- 定向治理测试：13项通过；
- 全量回归测试：400项通过；
- 元数据门槛：6/6站点通过；
- 心跳映射：四种 N 均为10景；
- 代码快照校验：通过；
- 确认锁箱访问次数：0；
- 储备访问次数：0；
- 锁箱信号值加载：否；
- 正式结果文件：不存在；
- Dry-run判定：`PASSED`。

关键哈希：

- 代码快照 SHA-256：`34e386037301af910302daafc647b4055ab001c878d74b7398f3587984d832e3`
- 冻结结果 SHA-256：`b9263842e11b45ca497237e436b7cafa7bf39014fa62c56daabe6eadd9578fa8`
- Dry-run结果 SHA-256：`fe5c79a6ee93d1b4dd567b6d2254ea43e308adf75546074fa44ac644cd0a3645`

## 7. 证据边界

- 这六个站点只用于验证主 Final 失败后提出的固定心跳修复；
- 即使支持性确认通过，原24站点主 Final仍保持 `FAILED`；
- 六站点样本量较小，结果只能作为独立支持证据，不能等同于新的大规模主 Final；
- 若确认失败，不得返回4个Pilot站点继续追正；
- 若失败原因仍是静默上下文恢复，应另行预注册风险感知看门狗，并寻找新的独立数据确认。

## 8. 复现入口

- 协议：`configs/stage6r_confirmation_lockbox_protocol_v1.json`
- 治理模块：`src/spectrum_semcom/stage6r_confirmation_governance.py`
- 冻结脚本：`scripts/prepare_stage6r_confirmation_lockbox.py`
- Dry-run脚本：`scripts/dry_run_stage6r_confirmation_lockbox.py`
- 单次执行入口：`scripts/run_stage6r_confirmation_lockbox.py`
- 冻结结果：`results/stage6r/confirmation_lockbox_freeze_v1/result.json`
- Dry-run结果：`results/stage6r/confirmation_lockbox_dry_run_v1/result.json`

下一步只有一个合法执行命令：

`python scripts/run_stage6r_confirmation_lockbox.py --consume`

该命令尚未执行。
