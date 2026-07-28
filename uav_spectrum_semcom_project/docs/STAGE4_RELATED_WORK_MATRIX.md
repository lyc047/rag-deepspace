# 阶段4相关工作核验与定位矩阵

核验日期：2026-07-16  
用途：硕士论文相关工作章节；只保留可由出版社、作者机构、CVF或arXiv原始页面确认的文献。  
注意：本矩阵比较研究问题与证据边界，不以网络名称相似作为“被借鉴”或“优于”的依据。

## 1. 核验文献

| 编号 | 文献与原始入口 | 核心问题 | 与本项目的关系 | 不能据此声称的内容 |
|---|---|---|---|---|
| R1 | P. Yi, Y. Cao, X. Kang, Y.-C. Liang, “Integrated Distributed Semantic Communication and Over-the-Air Computation for Cooperative Spectrum Sensing,” *IEEE TCOM*, 73(4):2416–2430, 2025, [DOI](https://doi.org/10.1109/TCOMM.2024.3468215), [IEEE](https://ieeexplore.ieee.org/document/10693603/) | 多传感器协作频谱感知，结合分布式语义通信和AirComp降低上报占用 | 与本项目最接近的频谱感知语义通信对照；说明检测性能—上报开销联合设计的重要性 | 本项目没有AirComp波形、同步叠加或MIMO物理层，不能把R1的物理层结论移植过来 |
| R2 | X. Li, S. Bi, S. Wang, X. Li, Y.-J. A. Zhang, “Digital Semantic Device-Edge Co-Inference With Task-Oriented ARQ,” *IEEE TVT*, 73(9):13986–13990, 2024, [DOI](https://doi.org/10.1109/TVT.2024.3390213), [机构页](https://research.cuhk.edu.hk/en/publications/digital-semantic-device-edge-co-inference-with-task-oriented-arq-2/) | 图像分类device-edge co-inference，依据SNR和推理结果做任务导向ARQ | 支持“CRC成功不等于任务充分”和任务反馈应计入时延/bit；对应本项目阶段3 ACK与阶段4G3机制 | 任务是图像分类而非频谱资源选择；不能把其准确率/时延数字作为本项目基线 |
| R3 | Y. Hu et al., “Pragmatic Communication in Multi-Agent Collaborative Perception,” arXiv:2401.12694, 2024, [arXiv](https://arxiv.org/abs/2401.12694) | 多智能体3D检测/跟踪中的消息、表示和协作者选择 | 为任务相关消息选择与协作者剪枝提供概念对照 | 属于视觉协同感知，通信量口径和任务指标不同；不能把视觉特征压缩倍数作为频谱语义收益 |
| R4 | C. Cai, X. Yuan, Y.-J. A. Zhang, “End-to-End Learning for Task-Oriented Semantic Communications Over MIMO Channels: An Information-Theoretic Framework,” *IEEE JSAC*, 43(4):1292–1307, 2025, [DOI](https://doi.org/10.1109/JSAC.2025.3531575), [机构页](https://research.cuhk.edu.hk/en/publications/end-to-end-learning-for-task-oriented-semantic-communications-ove/) | 多设备分类任务、MIMO多址信道、特征编码和预编码联合设计 | 说明真正的MIMO task-oriented SemCom需要显式信道矩阵、预编码和分类目标 | 本项目是多节点数字上报，不具备MIMO信道矩阵或联合预编码，不能称MIMO算法 |
| R5 | B. Liu et al., “mmCooper: A Multi-agent Multi-stage Communication-efficient and Collaboration-robust Cooperative Perception Framework,” *ICCV*, 28396–28406, 2025, [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_mmCooper_A_Multi-agent_Multi-stage_Communication-efficient_and_Collaboration-robust_Cooperative_Perception_Framework_ICCV_2025_paper.html) | 视觉协同感知中间/后期多阶段融合和标定误差鲁棒性 | 与C3的“低质量协作者可能传播错误”问题相似，可用于讨论鲁棒协同 | 其检测框/特征融合不等于频谱占用融合；本项目C3失败，不能借其结论宣称鲁棒协同成立 |
| R6 | W. Chen et al., “Entropy-and-Channel-Aware Adaptive-Rate Semantic Communication with MLLM-Aided Feature Compensation,” arXiv:2501.15414v4, 2026（v1提交于2025）, [arXiv](https://arxiv.org/abs/2501.15414) | MIMO衰落下基于CSI、SNR、熵和特征选择的图像自适应码率 | 与G0—G3和链路自适应有概念联系；强调内容与信道共同决定码率 | 当前版本研究图像与MLLM补偿，不是频谱任务；题名自v1后已变化，引用时必须使用当前题名/版本 |
| R7 | B. Li et al., “Toward Reliable Semantic Communication: Beyond Average Performance,” arXiv:2606.01284, 2026, [arXiv](https://arxiv.org/abs/2606.01284) | 讨论平均性能之外的尾部可靠性、适应、鲁棒codec和HARQ | 支持本项目报告CVaR、失效场景和反馈成本的研究动机 | 属于综述/观点性工作，不是本项目方法的性能基线，也不能替代本项目统计证据 |

## 2. 研究空隙

现有工作分别覆盖协作频谱检测、任务导向ARQ、视觉协作者选择、MIMO预编码、自适应图像码率和语义可靠性，但它们没有同时处理以下组合：

1. 以连续频谱块选择的occupancy regret而非检测准确率或重建质量作为主任务损失；
2. 采用可审计数字包，统一计入header、CRC、FEC、填充、反馈和重传；
3. 在发送候选高精度报告前估计其相对当前融合状态的反事实边际价值；
4. 区分感知SNR、前端削顶/泄漏、预测校准、信息年龄和上报可靠性；
5. 以独立scene为统计单位报告平均regret与CVaR，并预注册final非劣和多重校正。

本项目最有证据支持的差异是第1、2和5点。第3点的网络/解析实现未通过Gate B，第4点的质量诊断完成但C3融合增益未建立。因此论文应把“研究了上述组合”与“每个组合都取得正收益”严格区分。

## 3. 可直接用于论文的相关工作段落

协作频谱感知中的语义通信已经开始从硬判决或完整软数据上报转向任务相关表示。例如，Yi等将分布式语义通信与AirComp结合，以协作主用户检测为目标降低上报资源占用。与此不同，本文不研究模拟空口叠加，而是在可审计数字链路上优化连续频谱块选择后悔值。任务导向ARQ工作进一步表明，传统bit错误判决并不完全反映推理任务是否需要重传；本文据此将反馈、补传bit和等待时延纳入闭环，但任务从图像分类改为频谱资源选择。

在多智能体协同感知领域，PragComm通过任务相关消息和协作者选择减少视觉特征通信，mmCooper则通过多阶段融合缓解标定误差和低质量协作者传播。它们说明“选择谁、发送什么以及如何抵抗错误协作者”是共同问题，但视觉检测特征、通信量口径和空间协同数据与本文不同，不能直接作为H3证据。Cai等的工作显式联合MIMO预编码、特征编码和分类，进一步说明多设备不自动等于MIMO；本文没有信道矩阵或联合预编码，只称为多节点数字上报系统。

自适应码率语义通信通常依据内容熵、CSI和SNR改变图像特征发送量，而近期可靠语义通信观点工作开始强调平均质量之外的尾部性能和HARQ开销。本文将这些思想落实到G0—G3数字频谱语义、CVaR和实际bit审计，但没有采用图像重建、MLLM补偿或模拟JSCC。因此，本文的核心定位不是提出更大的通用语义网络，而是验证任务损失、数字包成本和频谱资源决策之间的闭环关系。
