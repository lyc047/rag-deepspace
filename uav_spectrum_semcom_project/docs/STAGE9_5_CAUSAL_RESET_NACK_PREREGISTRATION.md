# 阶段9.5显式因果反馈与RESET-NACK预注册

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-01
- Verification Status: ANALYZED
- Version Label: stage9_5_preregistration_v1

## 1. 研究问题

阶段9.5检验：在发送端完全不能读取接收端内部状态的条件下，当失活接收端收到普通更新时，返回一个明确计费的32 bit `RESET-NACK`，能否缩短 boot ID未知期，并恢复阶段9.4丢失的clean性能。

## 2. 因果闭包

发送端只能根据成功送达的四类控制帧改变信念：

1. 64 bit启动状态；
2. 24 bit ACK；
3. 32 bit RESET-NACK；
4. 32 bit看门狗响应。

前向包未送达时，接收端不产生响应；反馈包未送达时，发送端状态不得变化。实验评估器可以读取接收端动作计算clean，但这些信息不得回流到发送端。

## 3. RESET-NACK结构

32 bit帧包括：4 bit魔数、2 bit版本、2 bit原因、16 bit boot ID和8 bit目录版本。原因字段区分目录存在、冷启动目录缺失和目录损坏。

本阶段把“异步启动公告不可用”与“普通反馈链路不可用”分开。公告发送器故障时，接收端仍可响应已收到的普通更新；该假设不适用于整个反馈发射机失效。

## 4. 公平比较

所有策略均采用阶段9.3标签恢复和相同因果状态机：

- 30景看门狗加RESET-NACK；
- 30景看门狗但无RESET-NACK；
- fixed10加RESET-NACK；
- boot-event加RESET-NACK。

四类策略共享完全相同的任务轨迹、重启和双向丢包序列。阶段9.1旧的内部oracle结果不再作为主基线。

## 5. 门槛

- 所有发送端内部状态读取、未计费反馈转移、错误上下文执行和旧恢复误接收均为0；
- 正常两类中，RESET-NACK相对无NACK的clean置信下界不低于−0.5个百分点，平均bit增长不超过5%；
- 启动公告故障两类中，RESET-NACK相对无NACK的clean改善置信下界至少3个百分点，平均bit增长不超过20%；
- 正常两类相对因果fixed10保持至少5%的bit节省置信下界；
- 启动公告故障两类相对因果boot-event的clean改善置信下界至少1个百分点，平均bit增长不超过20%。

开发使用每类80条轨迹、种子20261501；开发通过并冻结后，确认使用每类120条轨迹、种子20261601。任何失败都按预注册停止或记录，不根据结果修改故障率和门槛。

