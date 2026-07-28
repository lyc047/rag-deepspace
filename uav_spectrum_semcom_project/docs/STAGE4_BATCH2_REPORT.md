# 阶段 4 Batch 2 报告：Gate A资源损失与任务非饱和审计

日期：2026-07-16  
状态：完成  
对应假设：H2  
final holdout：未建立、未访问

## 1. 本批结论

本批完成Gate A的数学核心和统一对照接口，并在训练前执行truth-only任务难度审计。首次审计无候选通过冻结门槛；保留该负结果后，仅增加每scene源密度而不修改门槛，第二次审计选出8源、8资源块、连续需求4的开发任务。新增12项测试，全项目95项通过。

本批证明的是“损失实现正确、梯度可用、任务具备非零决策空间”，尚未证明resource-aware训练优于detection-only。

## 2. 资源任务损失

对每scene的$C$个子信道和连续需求$d$，先计算所有连续资源块的平均占用。离散评价仍采用预测最小占用块，并以其真实占用减去oracle真实占用作为regret。

训练时采用：

$$
\pi(a)=\operatorname{softmax}(-\tau\hat O_a),\qquad
\mathcal L_{res}=\sum_a\pi(a)O_a-\min_jO_j.
$$

同时实现：

- missed-occupancy：仅对真实占用项施加正类对数损失，显式提高漏占用代价；
- empirical CVaR：取每批最差$1-\alpha$比例scene的平均regret；
- Brier：约束占用概率校准；
- rate项：$\bar B/B_{max}-1$；
- 投影对偶更新：$\lambda\leftarrow[\lambda+\eta(\bar B/B_{max}-1)]_+$。

所有损失均保持逐scene分量，可同时报告平均性能与尾部风险。测试验证soft regret在高逆温度下逼近离散选择，且梯度有限、非零。

## 3. Gate A统一接口

四种设置共享输入、前端、初始化、训练轮数、预算和评价，仅改变以下损失项：

| 方法 | detection | rate | resource | miss/CVaR/Brier |
|---|---:|---:|---:|---:|
| detection-only | 是 | 否 | 否 | 否 |
| detection+rate | 是 | 是 | 否 | 否 |
| detection+resource | 是 | 否 | 是 | 否 |
| full joint | 是 | 是 | 是 | 是 |

默认逆温度20、CVaR水平0.9；后续权重只能在validation上选择。

## 4. 任务难度审计

审计只读取RadDet train/val真值几何，不读取detector预测或所提方法结果，因此不会按算法表现挑任务。无关帧组合只用于H2压力诊断，不能证明H3。

### v1负结果

候选为1/2/4源、4/8/16资源块、1/2/4连续需求，共54组train/validation组合。没有验证组合同时满足：

- mean random-oracle gap至少0.02；
- ambiguous oracle rate不高于0.8；
- oracle clean rate位于0.2—0.98。

结果保存在`results/stage4/task_difficulty_audit/`。

### v2修正与选择

根据预先定义的“任务饱和”失败分支，仅增加6、8源候选，所有门槛不变。v2共90组，validation选中：

| Sources/scene | 资源块 | 连续需求 | Oracle clean | Random-oracle gap | Oracle margin | Ambiguous |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 4 | 0.8200 | 0.026813 | 0.013327 | 0.2133 |

结果保存在`results/stage4/task_difficulty_audit_v2/`。该配置现已冻结用于下一批Gate A开发训练。

## 5. 验证与边界

- Batch 2新增测试：12项；
- 全项目：95项通过，用时6.31 s；
- final holdout：未建立、未访问；
- 任务选择使用truth-only几何，不构成算法效果调参；
- 当前rate为可微期望bit约束，正式评价仍必须经过Batch 1 codec与阶段2链路统计实际发送bit；
- 当前没有H2优越性结论。

## 6. 下一批唯一主线

物化选定开发任务的train/validation清单，在冻结前端和完全一致的训练控制下运行Gate A四种损失设置。主判断顺序为离散occupancy regret、CVaR、实际bit，再报告检测和校准指标。
