# RAG增强深空遥测信号传输系统

基于检索增强生成（RAG）的深空通信信号优化系统。通过"物理参数粗筛 + 信号特征精排"的分层检索策略，在不增加带宽的前提下提升深空遥测信号的重建质量。

## 安装
```bash
pip install -r requirements.txt
```

## 运行实验
```bash
python experiments/exp01_mvp_baselines.py
```

## 运行测试
```bash
pytest tests/ -v
```

## 项目结构
```
src/              # 核心模块
  telemetry.py    # 合成遥测生成器
  channel.py      # 深空信道仿真器
  knowledge.py     # 知识库（SQLite + FAISS）
  retrieval.py     # 检索策略（粗筛+精排+基线）
  reconstruction.py # 信号重建
  metrics.py       # 评估指标
  visualization.py  # 论文图表
experiments/       # 实验脚本
tests/             # 单元测试
results/           # 实验结果输出
```
