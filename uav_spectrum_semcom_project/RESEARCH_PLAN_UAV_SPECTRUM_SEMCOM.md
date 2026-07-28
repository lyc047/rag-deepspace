# 面向低空无人机协同宽带频谱感知的任务导向 I/Q 语义通信研究计划

> 状态说明（2026-07-27）：本文为早期总体研究设想。当前频谱资源算法的正式执行主线已转移至 `docs/MASTER_RESEARCH_MAINLINE_PLAN.md`，图像、多模态、多无人机和MIMO不属于近期默认研究范围。

> 版本：v1.2  
> 日期：2026-07-07  
> 项目目录：`rag-deepspace/uav_spectrum_semcom_project`  
> 研究主线：宽带频谱感知、任务导向语义通信、多无人机协同、感知—计算—通信联合优化

> **硕士论文范围更新（阶段 0）**：后续论文主线以 `docs/THESIS_STAGE0_SCOPE.md` 为权威定义，聚焦“频谱语义数字上报—多 UAV 质量感知融合—频谱资源选择”。本文档中图像语义、多模态决策、DQN、轨迹及其他扩展内容均视为候选扩展，不再与阶段 0 的 H1—H4 和 C1—C3 具有同等优先级。机器可读约束见 `configs/research_scope.json`。

## 1. 课题定位

### 1.1 建议题目

**面向低空无人机协同宽带频谱感知的任务导向 I/Q 语义通信与感知—计算—通信联合优化**

英文暂定题目：

> Task-Oriented Semantic Communication for Collaborative Wideband Spectrum Sensing in Low-Altitude UAV Networks

### 1.2 核心问题

宽带 SDR 会持续生成高采样率复数 I/Q 数据。若观测带宽为 $B$，I、Q 每个分量使用 $q$ bit 量化，则未压缩数据率近似为：

$$
R_{IQ}=2Bq.
$$

例如，$B=100\,\mathrm{MHz}$、$q=12$ bit 时，原始数据率约为 $2.4\,\mathrm{Gbps}$。多架无人机不可能长期把完整 I/Q 数据实时上传到地面融合中心。

但下游任务通常并不要求逐样本恢复 I/Q，而只关心：

- 哪些频段被占用；
- 信号何时出现和消失；
- 信号的时频范围；
- 信号或辐射源类别；
- 是否属于干扰、欺骗或未知信号；
- 发射源的大致位置；
- 判断结果的置信度和可复核证据。

因此，本课题研究如何在有限空地带宽、机载算力和能量约束下，只传输完成上述任务所需的最小信息。

### 1.3 不作为主线的内容

- 不以完整 I/Q 波形重建 MSE 为主要目标；
- 不把普通 RGB 图像压缩作为主要贡献；
- 不在第一阶段同时引入 AirComp、强化学习、轨迹优化和真实飞行；
- 不把连续潜在向量未经数字化就称为“传输比特”；
- 不使用生成模型伪造关键原始证据；
- 不在无严格数据划分的情况下报告模型性能。

## 2. 研究背景、实验必要性与研究现状

### 2.1 背景知识与问题来源

本课题处在“宽带频谱感知—低空无人机网络—任务导向语义通信—感知/计算/通信融合”四条研究线的交叉位置。它不是简单把语义通信模型套到图像压缩上，而是面向一种更贴近无线系统实际瓶颈的对象：宽带射频 I/Q 数据。

从通信系统角度看，传统通信关注比特可靠传输，核心指标是误码率、吞吐率和频谱效率。Shannon 信息论为可靠通信奠定了基本边界，但并不区分“每个比特是否对下游任务有用”[1]。在智能感知与边缘推理场景中，接收端往往并不需要恢复完整源数据，而只需要完成检测、分类、定位、告警或决策。语义通信和任务导向通信正是沿着这一逻辑发展：通信系统不再只优化源重建失真，而是优化任务性能、通信开销、时延和鲁棒性的联合指标[13-17]。

从频谱感知角度看，认知无线电和动态频谱接入长期关注“频谱是否被占用、主用户是否存在、是否存在频谱空洞”。但早期大量工作集中在窄带检测、能量检测、循环平稳特征检测或简单协作判决上。随着频谱监测带宽扩大，系统需要同时判断信号的时间边界、频率边界、类别、置信度以及跨节点一致性，问题已经从二元检测演化为宽带时频目标检测与多节点融合问题。

从低空无人机场景看，UAV 具备快速部署、三维机动、视距链路概率高和空间多样性等优势，非常适合临时频谱监测、干扰排查、灾害应急、低空空域监管和电子态势感知。但 UAV 的机载算力、电池、存储和空地链路带宽均受限，且低空环境中的遮挡、多径、姿态变化和多普勒效应会使观测质量与上报链路质量动态变化。也就是说，UAV 可以“看见”大量频谱数据，却未必能把这些数据完整传回。

因此，本项目的核心矛盾可以表述为：

> 低空 UAV 集群能够产生高吞吐的宽带 I/Q 观测，但空地链路、机载计算与能耗限制要求系统只上传对频谱感知任务真正有用的信息。

这使得“任务导向 I/Q 语义通信”比普通图像压缩更适合作为研究切入点：一方面，原始 I/Q 数据存在明确的容量瓶颈；另一方面，任务输出通常远比原始波形稀疏，具备语义压缩的空间；再者，错误代价并不等同于像素 MSE 或波形 MSE，而是体现为漏检、误检、时频定位偏差、类别混淆和错误告警。

### 2.2 项目实验必要性

本项目必须通过实验推进，而不能只停留在理论描述，原因主要有五点。

第一，容量瓶颈需要被定量验证。宽带 I/Q 的理论数据率可以用 $R_{IQ}=2Bq$ 粗略估算，但真实系统还涉及采样率、帧长、元数据、缓存格式、STFT 特征膨胀、压缩格式和上报协议开销。若不做数据率审计，很容易把“模型压缩率”误认为“链路节省率”。第一阶段实验必须给出原始 I/Q、STFT、传统压缩、硬判决、软判决和语义 token 的统一 bit 账本。

第二，任务性能与压缩率之间的关系无法仅靠公式判断。频谱感知任务对不同信息的敏感性不同：频率边界误差、时间边界误差、类别错误和低置信度告警具有不同后果。实验需要回答“保留哪些特征最值得”“传多少 bit 后性能收益趋于饱和”“什么信道条件下语义传输优于硬判决或软数据融合”等问题。

第三，宽带 RF 数据与普通图像数据的统计结构不同。I/Q 数据是复值、相干、受信道影响强、对同步和频偏敏感的物理层观测；STFT 图虽然可以借用目标检测或语义分割方法，但其像素强度与颜色纹理不同，背后对应功率谱、瞬态、调制结构和噪声统计。直接迁移图像压缩或图像语义通信结论，缺少说服力。

第四，多 UAV 协同会引入单节点实验看不到的问题。不同 UAV 的观测存在空间多样性，也存在异步、遮挡、局部强干扰和链路质量差异。简单汇聚全部软数据会带来上报开销，简单投票又可能损失关键信息。因此必须实验比较硬判决融合、软数据融合、特征融合、语义 token 融合和按不确定性触发的原始证据回传。

第五，真实或半真实数据稀缺，必须建立可复现实验协议。RF 数据常见问题包括同源样本泄漏、按窗口随机切分导致训练集和测试集高度相关、仿真参数过于单一、只在固定 SNR 上评估、未区分感知信道与上报信道。若第一阶段不建立数据卡、划分清单和容量审计，后续模型即便指标较高，也难以支撑论文结论。

据此，本项目第一阶段实验应优先验证以下假设：

- H1：在宽带频谱感知任务中，完整 I/Q 上传相对于任务输出存在显著冗余，真实瓶颈是 I/Q 或高分辨率时频表示，而不是少量遥测元数据。
- H2：在相同上报 bit 预算下，任务导向语义 token 能在检测、时频定位和分类任务上优于传统硬判决，并接近软数据融合性能。
- H3：当上报信道存在噪声、丢包或 SNR 失配时，端到端训练的语义编码需要显式数字化、量化和纠错建模，否则难以与可部署通信链路对接。
- H4：多 UAV 协同的收益主要来自空间多样性和不确定性互补，而不是简单增加样本数；因此融合模型必须利用 UAV 位置、观测质量和置信度。
- H5：小规模代表性数据子集足以完成管线验证、容量审计和初始模型筛选，但最终泛化结论必须在更完整的数据覆盖上复验。

### 2.3 国内外研究现状

#### 2.3.1 频谱感知与协同频谱感知

认知无线电研究最早将频谱资源视为可动态感知和机会接入的对象。Haykin 的认知无线电框架强调无线系统应具备环境感知、学习和自适应能力[2]。随后，Yucek 和 Arslan 系统总结了能量检测、匹配滤波、循环平稳检测、协同感知等典型频谱感知算法，奠定了频谱感知研究的基本分类[3]。Akyildiz 等进一步讨论了协同频谱感知中的隐藏节点、报告信道、融合规则和协作开销问题[4]。

这些工作的重要性在于，它们证明了频谱感知不是单纯信号分类问题，而是受感知信道、报告信道、融合中心和判决策略共同影响的系统问题。但其局限也很明显：很多经典算法面向窄带或固定信道，任务输出多为“是否占用”的二元判决；即使是协同感知，也常在硬判决融合或软能量融合框架下讨论，较少考虑宽带时频定位、信号类别、语义级特征压缩和端到端任务性能。

#### 2.3.2 宽带 I/Q 数据、压缩感知与宽带信号识别

宽带频谱感知的关键难点是采样率与数据率随带宽线性增长。Tian 和 Giannakis 将压缩感知用于宽带认知无线电，利用频谱稀疏性降低采样和检测开销[5]。Mishali 和 Eldar 的 sub-Nyquist 宽带感知工作进一步说明，若信号在频域具有多带稀疏结构，可以通过混合模拟/数字结构低于 Nyquist 速率完成频谱支撑恢复[6]。

压缩感知方向给本项目提供了重要基线：它回答的是“能否用更少观测恢复频谱支撑”。但本项目更关注“在已经获得本地 I/Q 观测后，UAV 应该向融合中心传什么”。二者相关但不等价。压缩感知通常依赖稀疏性、测量矩阵和恢复算法假设；而任务导向语义通信需要直接面向检测、定位、分类和融合性能优化，并把上报信道、量化 bit 数和误码/丢包纳入实验。

近年 RF 机器学习从调制识别扩展到宽带信号识别。O'Shea 等关于 over-the-air 深度无线信号分类的工作推动了 I/Q 深度学习建模[10]；West、O'Shea 和 Roy 提出的宽带信号识别数据集明确把任务定义为检测、时频定位和分类的联合问题[11]；RadDet 则进一步给出宽带雷达频谱检测数据与时频标注[12]。这些数据集和任务定义非常贴近本项目，因为它们不再只问“这是什么调制”，而是问“信号在哪里、是什么、在多宽频带和多长时间内出现”。

#### 2.3.3 UAV/低空无线网络与上报链路约束

UAV 通信研究表明，低空平台在覆盖增强、应急通信、临时基站、移动中继和空中终端方面具备独特优势。Zeng、Zhang 和 Lim 讨论了 UAV 无线通信的机会和挑战，指出 UAV 信道、轨迹、覆盖和资源分配需要联合设计[7]。Mozaffari 等在 IEEE Communications Surveys & Tutorials 中系统总结了 UAV 无线网络的应用、关键挑战和开放问题，强调了移动性、能量、部署、干扰和链路可靠性等因素[8]。此外，蜂窝连接 UAV 的智能连接、安全与机器学习研究也说明，低空平台需要把通信可靠性、环境感知和学习式决策结合起来考虑[9]。

对本项目而言，UAV 不是简单的数据采集平台，而是“移动感知节点 + 边缘计算节点 + 受限通信节点”。这意味着实验中必须区分两类信道：发射源到 UAV 的感知信道，以及 UAV 到地面站的上报信道。前者决定本地观测质量，后者决定语义 bit 流能否可靠到达。若两类信道混为一谈，模型性能提升无法解释，也难以形成工程上可部署的结论。

#### 2.3.4 语义通信、JSCC 与任务导向通信

语义通信研究从“可靠传输比特”转向“可靠传输意义或任务相关信息”。Xie 等提出 DeepSC，将深度学习用于文本语义通信，显示了在低 SNR 下以语义相似度为目标的端到端通信潜力[13]。Yang 等在 IEEE Communications Surveys & Tutorials 中系统梳理了语义通信的基本概念、应用和挑战[14]；Qin 等从原则与挑战角度强调，语义通信需要重新定义性能指标，不能只沿用 BER/BLER[15]。

另一方面，深度 JSCC 工作证明了端到端学习式源信道联合编码可在图像传输中获得优雅退化特性。Bourtsoulatze、Kurka 和 Gündüz 的 DeepJSCC 是该方向代表性工作，但其主要对象是自然图像重建，评价指标以 PSNR/MS-SSIM 等重建质量为主[16]。Shao、Mao 和 Zhang 的多设备协同边缘推理研究则更接近本项目的任务导向思想：边缘设备上传与推理任务相关的特征，而不是完整原始数据[17]。

这些研究为本项目提供了方法基础，但也留下研究空白：现有语义通信大量集中在文本、图像、语音和视频；RF I/Q 场景下的语义表示、bit 账本、数字化传输、上报信道鲁棒性、多节点协同和可解释证据回传仍不充分。尤其是频谱感知任务中的错误并不是“图像变模糊”，而是漏检干扰源、误判占用频段或错误估计时频边界，因此评价指标必须从重建失真转为任务效用。

#### 2.3.5 ISAC/ISCC 与“感知—计算—通信”融合

ISAC/JCR 研究强调通信与感知共享频谱、硬件、波形和信号处理链路。Liu 等在 IEEE JSAC 的综述指出，感知功能正在成为 6G RAN 的关键能力[18]；Zhang 等从信号处理角度总结了联合通信与雷达感知系统的设计范式[19]。它们说明“通信系统本身承担感知功能”已经是 6G 重要趋势。

但本项目与典型 ISAC/JCR 仍有差异：本项目不首先设计共享发射波形，而是研究多个 UAV 已经获得宽带被动/半主动频谱观测后，如何在受限上报链路中完成协同感知。换言之，它更接近 ISCC：感知数据在边缘被计算压缩，再经通信链路传输到融合中心。Yi、Cao、Kang 和 Liang 关于“分布式语义通信 + 空中计算”的协同频谱感知工作已经明确触及这一方向，表明语义通信与协同频谱感知结合具有现实研究价值[20]。但该方向仍处于较早阶段，尤其缺少面向宽带时频目标检测、多类别信号、真实/半真实数据集、数字 bit 预算和 UAV 低空链路的系统实验。

### 2.4 现有研究不足与本项目切入点

综合上述文献，可以得到以下判断。

1. 频谱感知已有坚实基础，但从“二元占用检测”扩展到“宽带时频检测 + 分类 + 多 UAV 融合 + bit 受限上报”的研究仍不充分。
2. 压缩感知可降低采样或恢复成本，但它主要面向信号支撑恢复，不直接等价于面向任务效用的语义压缩与协同融合。
3. RF 深度学习已有调制识别和宽带识别数据集，但大量工作默认完整数据在推理端可用，较少把上报链路容量作为核心约束。
4. 语义通信和 DeepJSCC 已在文本/图像/视频上快速发展，但 RF I/Q 数据的物理含义、信道失配、数字化部署和任务指标尚未形成成熟范式。
5. UAV 网络研究充分说明低空平台适合移动感知，但多数工作关注覆盖、轨迹、接入和资源分配，对“UAV 采集到的宽带 I/Q 如何低开销回传并融合”关注不足。
6. 公开宽带 RF 数据仍然稀缺，且容易出现数据泄漏和评估不可复现问题。因此第一阶段必须先完成数据卡、划分清单、容量审计和基线体系，而不是急于堆复杂模型。

本项目的可发表切入点是：

> 在低空 UAV 协同宽带频谱感知场景中，构建一个严格 bit 预算下的任务导向 I/Q 语义通信框架，比较完整 I/Q、传统压缩、硬判决、软数据融合和语义 token 融合在检测、时频定位、分类、鲁棒性和通信开销上的系统差异。

这一切入点的优势是问题真实、瓶颈清晰、实验可落地、与语义通信/ISAC/UAV/RFML 多个热点相连，同时避开了普通图像压缩研究过度拥挤的问题。

### 2.5 权威文献矩阵

下表列出第一阶段建议优先阅读和引用的核心文献。文献数量不少于 15 篇，覆盖理论基础、频谱感知、宽带压缩、UAV 网络、语义通信、ISAC/ISCC 与数据集。正式出处和原文链接见文末“参考文献”。

| 序号 | 文献 | 类型/来源 | 对本项目的作用 |
|---:|---|---|---|
| 1 | C. E. Shannon, “A Mathematical Theory of Communication,” Bell System Technical Journal, 1948. | 信息论基础 | 提供传统可靠通信的理论起点，用于说明语义通信并非否定 Shannon，而是在任务层重新定义目标。 |
| 2 | S. Haykin, “Cognitive Radio: Brain-Empowered Wireless Communications,” IEEE JSAC, 2005. DOI: 10.1109/JSAC.2004.839380. | 认知无线电奠基 | 说明无线系统需要环境感知、学习与自适应，是频谱感知研究的源头之一。 |
| 3 | T. Yucek and H. Arslan, “A Survey of Spectrum Sensing Algorithms for Cognitive Radio Applications,” IEEE Communications Surveys & Tutorials, 2009. DOI: 10.1109/SURV.2009.090109. | 频谱感知综述 | 建立能量检测、匹配滤波、循环平稳检测、协同感知等基线分类。 |
| 4 | I. F. Akyildiz, B. F. Lo, and R. Balakrishnan, “Cooperative Spectrum Sensing in Cognitive Radio Networks: A Survey,” Physical Communication, 2011. DOI: 10.1016/j.phycom.2010.12.003. | 协同感知综述 | 支撑多节点融合、报告信道、硬/软判决融合等背景。 |
| 5 | Z. Tian and G. B. Giannakis, “Compressed Sensing for Wideband Cognitive Radios,” IEEE ICASSP, 2007. DOI: 10.1109/ICASSP.2007.367330. | 宽带压缩感知 | 作为宽带稀疏频谱检测与传统压缩类基线。 |
| 6 | M. Mishali and Y. C. Eldar, “Wideband Spectrum Sensing at Sub-Nyquist Rates,” IEEE Signal Processing Magazine, 2011. DOI: 10.1109/MSP.2011.941094. | sub-Nyquist 宽带感知 | 说明宽带频谱感知中的采样率瓶颈和频谱支撑恢复思想。 |
| 7 | Y. Zeng, R. Zhang, and T. J. Lim, “Wireless Communications with Unmanned Aerial Vehicles: Opportunities and Challenges,” IEEE Communications Magazine, 2016. DOI: 10.1109/MCOM.2016.7470933. | UAV 通信经典综述 | 支撑低空 UAV 场景、链路与机动性约束。 |
| 8 | M. Mozaffari et al., “A Tutorial on UAVs for Wireless Networks: Applications, Challenges, and Open Problems,” IEEE Communications Surveys & Tutorials, 2019. DOI: 10.1109/COMST.2019.2902862. | UAV 网络权威教程 | 支撑 UAV 网络中的能耗、部署、移动性、资源分配和开放问题。 |
| 9 | U. Challita et al., “Machine Learning for Wireless Connectivity and Security of Cellular-Connected UAVs,” IEEE Wireless Communications, 2019. DOI: 10.1109/MWC.2018.1800155. | UAV + ML | 支撑低空/蜂窝连接 UAV 中学习式资源与安全管理背景。 |
| 10 | T. J. O’Shea, T. Roy, and T. C. Clancy, “Over-the-Air Deep Learning Based Radio Signal Classification,” IEEE JSTSP, 2018. DOI: 10.1109/JSTSP.2018.2797022. | RF 深度学习 | 提供 I/Q 深度学习和 over-the-air 信号分类基线。 |
| 11 | N. West, T. O’Shea, and T. Roy, “A Wideband Signal Recognition Dataset,” IEEE SPAWC, 2021. IEEE Xplore: 9593265; arXiv:2110.00518. | 宽带信号识别数据集 | 明确检测、时频定位和分类联合任务，适合作为第一阶段数据/任务参考。 |
| 12 | Z. Huang et al., “RadDet: A Wideband Dataset for Real-Time Radar Spectrum Detection,” ICASSP, 2025. DOI: 10.1109/ICASSP49660.2025.10887772; arXiv:2501.10407. | 宽带雷达检测数据集 | 提供宽带 I/Q、时频标注、多 SNR、多密度雷达场景，适合验证容量瓶颈。 |
| 13 | H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, “Deep Learning Enabled Semantic Communication Systems,” IEEE Transactions on Signal Processing, 2021. DOI: 10.1109/TSP.2021.3071210. | DeepSC/语义通信 | 语义通信代表性端到端模型，用于说明从 bit 正确转向语义正确。 |
| 14 | W. Yang et al., “Semantic Communications for Future Internet: Fundamentals, Applications, and Challenges,” IEEE Communications Surveys & Tutorials, 2023. DOI: 10.1109/COMST.2022.3223224. | 语义通信权威综述 | 提供语义通信分类、指标、应用与挑战的系统背景。 |
| 15 | Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, “Semantic Communications: Principles and Challenges,” arXiv, 2021. arXiv:2201.01389. | 语义通信原则 | 支撑语义指标、系统框架和开放问题讨论。 |
| 16 | E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, “Deep Joint Source-Channel Coding for Wireless Image Transmission,” IEEE TCCN, 2019. DOI: 10.1109/TCCN.2019.2919300. | DeepJSCC | 作为源信道联合编码方法论参考，同时说明普通图像 JSCC 与 RF 任务的差异。 |
| 17 | J. Shao, Y. Mao, and J. Zhang, “Task-Oriented Communication for Multi-Device Cooperative Edge Inference,” IEEE Transactions on Wireless Communications, 2022/2023. DOI: 10.1109/TWC.2022.3191118. | 任务导向通信 | 支撑多设备协同推理和任务相关特征上传思想。 |
| 18 | F. Liu et al., “Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond,” IEEE JSAC, 2022. DOI: 10.1109/JSAC.2022.3156632. | ISAC 权威综述 | 支撑感知与通信融合的 6G 背景。 |
| 19 | J. A. Zhang et al., “An Overview of Signal Processing Techniques for Joint Communication and Radar Sensing,” IEEE JSTSP, 2021. DOI: 10.1109/JSTSP.2021.3113120. | JCR/ISAC 信号处理综述 | 支撑联合通信雷达感知中的信号处理与系统设计背景。 |
| 20 | P. Yi, Y. Cao, X. Kang, and Y.-C. Liang, “Integrated Distributed Semantic Communication and Over-the-Air Computation for Cooperative Spectrum Sensing,” IEEE Transactions on Communications, 2025. DOI: 10.1109/TCOMM.2024.3468215. | 语义通信 + 协同频谱感知 | 直接相关前沿工作，证明“语义通信 + CSS + AirComp”具有研究价值，同时暴露宽带时频/UAV/数据集方向的进一步空间。 |

### 2.6 第一阶段应形成的文献综述结论

第一阶段报告不应只是罗列文献，而要形成以下结论链：

1. 频谱资源动态利用和低空电磁监管需要宽带频谱感知，经典认知无线电与协同频谱感知提供理论基础。
2. 宽带 I/Q 数据量巨大，完整上传在 UAV 空地链路中不可持续，因此存在真实容量瓶颈。
3. 下游任务关注检测、时频定位、分类和告警，而非逐样本恢复 I/Q，因此具备任务导向语义压缩空间。
4. 传统压缩感知和硬/软判决融合是必要基线，但不能完整解决“bit 受限、信道受损、多 UAV 协同、任务指标驱动”的联合问题。
5. 语义通信、DeepJSCC 和任务导向边缘推理提供方法启发，但 RF I/Q 宽带感知场景仍缺少统一 bit 账本、无泄漏数据协议和系统级对比。
6. 因此，本项目的第一阶段实验必须先完成数据容量审计、代表性数据子集、基线复现和评价协议，随后再进入语义编码模型设计。

## 3. 研究场景与系统模型

### 3.1 场景描述

考虑 $K$ 架搭载 SDR 的无人机，对同一片低空区域进行宽带无线频谱监测。各 UAV 从不同空间位置接收通信、雷达或干扰信号，在机载端进行轻量语义提取，通过受限空地链路上报地面融合中心。

```text
通信/雷达/干扰发射源
          │
          │ 感知信道
          ▼
┌─────────────────────────────┐
│ UAV 1：SDR → 本地语义编码器  │──┐
│ UAV 2：SDR → 本地语义编码器  │──┼→ 空地上报信道 → 地面融合中心
│ ...                         │  │                    │
│ UAV K：SDR → 本地语义编码器  │──┘                    ▼
└─────────────────────────────┘              检测/时频定位/分类
                                               /发射源定位/置信度
```

### 3.2 两类信道必须分开建模

1. **感知信道**：发射源到 UAV 的无线传播，决定本地 I/Q 观测质量。
2. **上报信道**：UAV 到融合中心的通信链路，决定语义比特流能否可靠传输。

感知信道考虑：

- 发射源—UAV 距离；
- 路径损耗；
- Rician/多径衰落；
- 阴影遮挡；
- 热噪声与局部干扰；
- 载波频偏；
- UAV 运动引起的多普勒。

上报信道考虑：

- UAV—地面站距离和仰角；
- LoS/NLoS 状态；
- Rician 衰落；
- 可用上报带宽；
- 调制编码方式；
- 包错误和突发丢包；
- 排队与传输时延。

### 3.3 任务输出定义

第 $m$ 个信号实例定义为：

$$
o_m=(c_m,t_m^{start},t_m^{end},f_m^{low},f_m^{high},p_m),
$$

其中 $c_m$ 是类别，$t_m$ 和 $f_m$ 表示时频边界，$p_m$ 是置信度。系统最终输出目标集合：

$$
\hat{\mathcal O}=\{\hat{o}_1,\hat{o}_2,\ldots,\hat{o}_{\hat M}\}.
$$

## 4. 总体技术路线

### 4.1 UAV 本地处理

```text
宽带 I/Q
  → 同步、去直流、增益归一化
  → 复数网络或 I/Q 双通道网络
  → 多尺度局部时频特征
  → 语义 Token 与重要度
  → Token 选择
  → 量化、熵编码、分包
  → 数字通信链路
```

第一版同时保留两条输入路线：

- STFT 时频图：作为稳定、易复现的基线；
- 原始 I/Q 双通道或复数 1D 网络：作为后续主要研究路线。

### 4.2 地面融合处理

```text
各 UAV 的语义 Token
  + UAV 位置
  + 本地感知 SNR
  + 置信度
  + 上报链路质量
        ↓
集合 Transformer / 图注意力网络
        ↓
全局检测、分类、时频定位与发射源定位
```

### 4.3 三流传输机制

#### A. 可靠元数据流

无损、强保护传输：

- UAV 编号、位置、姿态；
- 时间戳；
- 中心频率、采样率、增益；
- 设备标定参数；
- 数据段编号和 CRC。

#### B. 实时语义流

传输：

- 频谱占用；
- 信号类别；
- 时频目标框；
- 干扰等级；
- 发射源位置估计；
- 模型置信度和不确定性。

#### C. 原始证据流

原始 I/Q 在 UAV 本地环形缓存，仅在以下条件触发上报：

- 高风险信号；
- 未知类别；
- 多 UAV 结论冲突；
- 模型不确定性过高；
- 地面主动请求；
- 链路空闲且缓存临近覆盖。

## 5. 优化目标

整体训练目标可写为：

$$
\mathcal L =
\mathcal L_{det}
+\alpha\mathcal L_{cls}
+\beta\mathcal L_{tf-loc}
+\gamma\mathcal L_{src-loc}
+\lambda R_{bits}
+\mu T_{latency}
+\nu E_{energy}
+\xi\mathcal L_{calibration}.
$$

其中：

- $\mathcal L_{det}$：目标存在性和目标框检测损失；
- $\mathcal L_{cls}$：信号类别损失；
- $\mathcal L_{tf-loc}$：时频边界定位损失；
- $\mathcal L_{src-loc}$：发射源位置估计损失；
- $R_{bits}$：真实数字码流长度；
- $T_{latency}$：感知、计算、排队和传输总时延；
- $E_{energy}$：采样、计算和上报总能耗；
- $\mathcal L_{calibration}$：置信度校准损失。

系统级约束可包括：

$$
P_D \ge P_D^{min},\qquad
P_{FA}\le P_{FA}^{max},\qquad
T_{end-to-end}\le T_{max},\qquad
E_k\le E_k^{budget}.
$$

## 6. 研究阶段总览

| 阶段 | 时间建议 | 主要目标 | 核心交付物 |
|---|---:|---|---|
| 第一阶段 | 第 1—2 月 | 问题定义、数据审计、无泄漏基准 | 数据管线、划分清单、基线复现报告 |
| 第二阶段 | 第 2—3 月 | 单 UAV 无压缩上界与传统压缩基线 | 任务性能—数据量基准曲线 |
| 第三阶段 | 第 3—5 月 | 单 UAV 数字任务导向语义编码 | 固定/可变码率语义编码器 |
| 第四阶段 | 第 5—7 月 | 多 UAV 语义融合 | 多节点融合模型和掉线鲁棒性实验 |
| 第五阶段 | 第 7—9 月 | 感知—计算—通信联合优化 | 节点、带宽、码率和能量调度器 |
| 第六阶段 | 第 9—10 月 | SDR 与嵌入式验证 | 受控 SDR 原型、时延和功耗报告 |
| 第七阶段 | 第 10—12 月 | 完整实验与论文 | 论文、代码、配置与可复现结果 |

## 7. 第一阶段详细任务分解

### 7.1 第一阶段目标

第一阶段不开发最终语义通信模型，而是回答四个前置问题：

1. 数据容量瓶颈是否真实且可量化？
2. 数据集是否支持检测、时频定位和分类任务？
3. 如何建立无数据泄漏、可复现的划分和指标？
4. 全数据检测上界和简单传统方法的性能分别是多少？

第一阶段完成标志：任何后续模型都能够在同一数据、同一划分、同一指标和同一通信开销计算规则下公平比较。

### 7.2 第一阶段周期

建议周期为 6 周。若硬件、磁盘或数据下载受限，可扩展至 8 周。

### 7.3 WP0：工作环境与存储准备

**建议时间：第 1 周前 1—2 天**

#### 任务

- 确认 PyTorch、CUDA、NumPy、SciPy、HDF5/Zarr 等环境；
- 确认至少 500GB 可用数据空间，推荐 1—2TB SSD；
- 建立数据、配置、日志和结果目录；
- 设置 Git 忽略规则，禁止原始数据和大模型进入版本库；
- 记录 CPU、GPU、内存、磁盘、CUDA 和依赖版本；
- 编写 100 个样本的 smoke test 配置。

#### 建议目录

```text
data/spectrum/
  raw/                 # 不加入 Git
  interim/             # 中间表示
  processed/           # 划分后的索引和轻量缓存
configs/spectrum/
src/spectrum/
experiments/spectrum/
results/spectrum/
docs/research/
```

#### 交付物

- `docs/research/environment.md`
- `configs/spectrum/smoke.yaml`
- 数据目录说明和 `.gitignore` 规则

#### 验收标准

- 能在当前机器上完成一次 100 样本读取和前向计算；
- 运行时不会一次性把整个数据集加载到内存；
- 原始数据路径与代码路径分离；
- 环境版本可被另一台机器复现。

### 7.4 WP1：研究边界与文献矩阵

**建议时间：第 1 周**

#### 任务 1：限定任务

第一篇工作只保留：

- 宽带信号检测；
- 时频目标定位；
- 信号分类；
- 通信开销统计。

暂缓：

- 发射源地理定位；
- UAV 轨迹优化；
- AirComp；
- 联邦学习；
- 强化学习；
- 真实飞行。

#### 任务 2：建立文献矩阵

文献表至少包含以下字段：

| 字段 | 内容 |
|---|---|
| 论文 | 标题、作者、年份、期刊/会议 |
| 数据 | 原始 I/Q、STFT、PSD 或人工特征 |
| 任务 | 检测、定位、分类、重建 |
| 节点 | 单节点或多节点 |
| 信道 | 是否区分感知和上报信道 |
| 编码 | 压缩感知、AutoEncoder、JSCC、数字语义编码 |
| 输出 | 连续潜在向量或实际数字码流 |
| 指标 | $P_D$、$P_{FA}$、mAP、F1、bit、时延、能耗 |
| 数据划分 | 是否防止同源样本泄漏 |
| 实验 | 纯仿真、SDR、真实飞行 |
| 局限 | 可直接用于本课题的研究空白 |

#### 必读起点

1. RadDet: A Wideband Dataset for Real-Time Radar Spectrum Detection  
   <https://arxiv.org/abs/2501.10407>
2. A Wideband Signal Recognition Dataset  
   <https://arxiv.org/abs/2110.00518>
3. Collaborative Wideband Spectrum Sensing and Scheduling for Networked UAVs  
   <https://arxiv.org/abs/2308.05036>
4. Integrated Distributed Semantic Communication and Over-the-Air Computation for Cooperative Spectrum Sensing  
   <https://ieeexplore.ieee.org/document/10693603/>
5. Wideband Spectrum Sensing: A Bayesian Compressive Sensing Approach  
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC6022006/>

#### 交付物

- `docs/research/literature_matrix.csv`
- `docs/research/problem_statement.md`
- 一页“已有工作—研究空白—拟解决问题”说明

#### 验收标准

- 至少精读 15 篇，其中近三年论文不少于 8 篇；
- 能明确说明本课题相对“UAV 本地分类后融合”的区别；
- 能明确说明本课题相对“二元协同频谱感知 + AirComp”的区别；
- 研究问题不依赖模糊的“语义相似度”。

### 7.5 WP2：数据获取、校验与容量审计

**建议时间：第 2 周**

#### 任务 1：获取数据与代码

- 获取 RadDet 数据或官方生成脚本；
- 获取 Wideband Signal Recognition 数据或生成代码；
- 保存许可证、版本、下载地址和校验值；
- 明确数据是原始 I/Q、时频图还是二者都有；
- 不在第一阶段同时引入过多辅助数据集。

#### 任务 2：容量审计

统计：

- 样本数；
- 每帧 I/Q 点数；
- 量化格式；
- 单帧和全数据集字节数；
- 每秒帧数假设；
- 原始实时数据率；
- STFT 后数据量；
- 标签数据量；
- 不同压缩方案的理论上报数据量。

至少比较：

- float32 I/Q；
- int16 I/Q；
- 8 bit I/Q；
- STFT float32；
- STFT 8 bit；
- 仅时频框和类别。

#### 任务 3：数据完整性检查

- 文件数量和校验值；
- NaN/Inf；
- 长度异常；
- 标签越界；
- 空标注；
- 类别不平衡；
- SNR 分布；
- 稀疏/密集场景比例；
- 重复样本和疑似同源样本。

#### 交付物

- `docs/research/data_card_raddet.md`
- `results/spectrum/phase1/data_audit.json`
- `results/spectrum/phase1/data_rate_table.csv`
- 随机样本 I/Q、PSD、STFT 和标注可视化

#### 验收标准

- 任意样本可由唯一 ID 定位；
- 可以从样本 ID 追溯类别、SNR、密度和基础信号实例；
- 给出不少于三种数据表示的实际容量；
- 明确说明为什么完整 I/Q 上报不现实。

### 7.6 WP3：统一数据接口与预处理

**建议时间：第 3 周**

#### 任务 1：统一样本对象

建议每个样本返回：

```python
{
    "iq": complex_or_two_channel_tensor,
    "sample_id": str,
    "source_group_id": str,
    "snr_db": float,
    "scene_density": str,
    "sample_rate_hz": float,
    "center_frequency_hz": float,
    "objects": [
        {
            "class_id": int,
            "t_start": float,
            "t_end": float,
            "f_low": float,
            "f_high": float,
        }
    ],
}
```

#### 任务 2：实现两类输入

1. 原始 I/Q：`[2, N]`，I/Q 分别作为两个通道；
2. STFT：`[C, F, T]`，至少提供对数功率谱和可选相位/实虚部。

#### 任务 3：预处理规则

- 去直流；
- 可选 IQ 不平衡校正；
- 训练集统计量归一化；
- 频率和时间坐标标准化；
- 长序列分块，但保留跨块目标标记；
- 支持变长输入或固定窗口；
- 所有参数写入配置文件；
- 不能使用测试样本自身统计量在接收端恢复尺度，除非计入旁信息。

#### 任务 4：缓存策略

- 原始数据优先保存为 int16 或原格式；
- 使用 memory map、HDF5 或 Zarr 分块读取；
- 第一阶段仅缓存一种主要 STFT 分辨率；
- 不复制多 SNR、多 UAV 数据，后续在线生成；
- 不生成不可追溯的匿名 `.npy` 窗口集合。

#### 交付物

- `src/spectrum/iq_dataset.py`
- `src/spectrum/iq_preprocessing.py`
- `tests/test_iq_dataset.py`
- `tests/test_iq_preprocessing.py`
- `configs/spectrum/data_raddet.yaml`

#### 验收标准

- 同一配置和种子产生完全一致的样本；
- 可在 16GB 内存机器上流式遍历数据；
- 1000 个样本不存在 NaN/Inf；
- 时频标注映射经过可视化人工核对；
- 数据加载器单元测试通过。

### 7.7 WP4：无泄漏数据划分与实验协议

**建议时间：第 4 周前半周**

#### 划分原则

不能随机打散所有窗口。需要先定义 `source_group_id`，同一基础信号实例派生出的：

- 不同 SNR；
- 不同噪声种子；
- 不同 STFT 分辨率；
- 不同窗口；
- 未来的不同 UAV 观测；

必须进入同一个数据划分。

#### 建议测试集

| 测试集 | 目的 |
|---|---|
| Test-A | 同分布基础性能 |
| Test-B | 未见 SNR 和信号密度组合 |
| Test-C | 未见发射参数或波形参数 |
| Test-D | 开放集未知类别 |
| Test-E | 后续真实 SDR 数据 |

#### 固定内容

- 训练/验证/测试 manifest；
- 随机种子；
- 类别映射；
- STFT 参数；
- 目标框 IoU 计算方式；
- mAP 阈值；
- 检测阈值选择规则；
- bit 数计算规则；
- 运行次数和置信区间规则。

#### 交付物

- `data/spectrum/processed/manifests/*.jsonl`
- `docs/research/evaluation_protocol.md`
- `tests/test_dataset_split.py`

#### 验收标准

- 不同划分的 `source_group_id` 交集为空；
- 所有划分统计量可自动生成；
- 类别和 SNR 分布差异有记录；
- 任何实验不得自行重新随机划分数据。

### 7.8 WP5：第一批基线与全数据性能上界

**建议时间：第 4 周后半周至第 5 周**

#### 基线 A：非学习方法

- 能量检测；
- Welch PSD；
- 固定阈值频谱占用；
- 连通域或峰值方法产生时频框；
- 简单规则或传统分类器。

#### 基线 B：完整信息学习上界

- 使用完整 STFT 输入的监督检测器；
- 不加入通信压缩；
- 输出检测框和信号类别；
- 该模型作为后续语义通信的任务上界，而不是最终方案。

#### 基线 C：低开销决策传输

- UAV 本地硬决策：类别 + 时频框；
- UAV 本地软决策：类别概率 + 时频框 + 置信度；
- 精确统计实际字节数，包含字段、量化、分包和元数据。

#### 基线 D：简单通用压缩

- I/Q 降位量化；
- I/Q + 通用无损压缩；
- STFT 量化；
- PCA 或线性降维；
- 小型普通 AutoEncoder，仅优化重建误差。

#### 第一阶段指标

- $P_D$ 与 $P_{FA}$；
- Precision、Recall、Macro-F1；
- 时频目标框 mAP；
- 类别混淆矩阵；
- bit/帧；
- 编码时间；
- 峰值内存；
- 模型参数量和 MACs。

#### 交付物

- `experiments/spectrum/exp01_data_baselines.py`
- `experiments/spectrum/exp02_full_information_detector.py`
- `experiments/spectrum/exp03_payload_accounting.py`
- `results/spectrum/phase1/baseline_summary.csv`
- `results/spectrum/phase1/figures/`

#### 验收标准

- 至少有一个传统方法和一个监督学习上界；
- 所有方法使用完全相同的测试 manifest；
- 所有通信方案统计真实 payload，而不是只报潜在维数；
- 结果可由单条命令和固定配置复现；
- 基线失败时能够定位到数据、标注、模型或指标环节。

### 7.9 WP6：阶段报告与 Go/No-Go 决策

**建议时间：第 6 周**

#### 必答问题

1. 原始 I/Q、STFT、软决策和硬决策各自需要多少 bit？
2. 完整信息监督检测器的任务上界是多少？
3. 简单能量/PSD 方法在低 SNR 和多信号重叠下损失多少？
4. 本地硬决策是否已经足够？
5. 任务潜在特征是否存在比硬/软决策更合理的空间？
6. 数据是否过度合成，是否必须提前增加真实 SDR 数据？
7. 当前硬件能否支持第三阶段训练？

#### 决策规则

**继续任务导向语义编码，若：**

- 完整信息模型显著优于硬/软规则基线；
- 原始或 STFT 数据量明显超过上报预算；
- 多信号重叠和低 SNR 下，本地独立判决存在明显信息损失；
- 中间特征具有压缩空间且能够数字化传输。

**转向语义调度而非复杂编码器，若：**

- 简单 FFT 峰值和软统计量已接近完整信息上界；
- 复杂模型收益小于其计算和能耗代价；
- 主要瓶颈是何时采样、由谁上报，而不是如何编码。

**提前增加真实数据，若：**

- 模型主要学习合成数据生成器特征；
- 在信道、频偏和硬件非理想变化下性能骤降；
- 公开数据无法覆盖目标低空场景。

#### 交付物

- `docs/research/phase1_report.md`
- `docs/research/phase2_decision.md`
- 第一阶段汇报 PPT 或 5—8 页技术报告

#### 验收标准

- 报告包含失败结果和局限，而非只保留最好结果；
- 明确第二阶段继续、缩减或调整的技术路线；
- 固定后续所有实验必须沿用的基准协议。

### 7.10 第一阶段逐周清单

| 周次 | 重点任务 | 周末应完成 |
|---|---|---|
| 第 1 周 | 环境、研究边界、文献矩阵 | 环境文档、问题定义、15 篇文献表 |
| 第 2 周 | 数据获取、容量与质量审计 | Data Card、容量表、样本可视化 |
| 第 3 周 | 数据接口与预处理 | 可测试的数据加载器和 STFT 管线 |
| 第 4 周 | 无泄漏划分、指标、传统基线 | 固定 manifest、评测协议、传统结果 |
| 第 5 周 | 完整信息上界、低开销传输基线 | 检测器、通信 payload 对比表 |
| 第 6 周 | 误差分析和路线决策 | 第一阶段报告、第二阶段 Go/No-Go |

## 8. 第二至第七阶段任务摘要

### 8.1 第二阶段：单 UAV 基线体系

- 完善压缩感知、PCA、AutoEncoder 和完整数字通信基线；
- 建立任务精度—数据量—计算量曲线；
- 给出本地硬决策、软决策和中间特征三种上报层级；
- 确定第一篇论文的最小问题范围。

### 8.2 第三阶段：单 UAV 数字语义编码

- I/Q 双通道或复数编码器；
- 多尺度时频语义 Token；
- Token 重要度和可变长度选择；
- 向量量化和熵编码；
- 分包、CRC 和实际数字链路；
- 信道状态条件编码；
- 未知类别和不确定性检测。

### 8.3 第四阶段：多 UAV 协同融合

- 生成同源多视角感知数据；
- 实现硬、软和语义特征融合；
- 对齐 UAV 时间、频率和位置；
- 处理 UAV 掉线、丢包和观测冲突；
- 学习共享语义和节点私有语义；
- 比较节点数量与性能、通信量关系。

### 8.4 第五阶段：联合资源优化

决策变量包括：

- UAV 选择；
- 感知频段；
- 感知驻留时间；
- 采样率；
- Token 数量；
- 量化位数；
- 上报功率；
- 调制编码；
- 原始证据触发策略。

方法顺序：规则 → 可微门控 → 监督调度 → Lyapunov/受约束 MDP → 必要时强化学习。

### 8.5 第六阶段：SDR 和嵌入式验证

三级测试：

1. 文件回放和信道仿真；
2. 同轴线 + 衰减器 + 多 SDR；
3. 合法频段被动接收或受控 ISM 信号，后续再考虑挂载 UAV。

测量：

- 实际采样吞吐；
- 编码时延；
- 上报时延；
- Jetson/树莓派功耗；
- 时钟偏差和频偏；
- 仿真—真实性能差距。

### 8.6 第七阶段：论文和可复现性

- 固定配置、随机种子和依赖；
- 每项关键实验重复 3—5 次；
- 报告均值、标准差或置信区间；
- 提供完整消融；
- 保存失败配置；
- 清理训练/测试泄漏；
- 开源能够复现实验图表的脚本。

## 9. 评价指标体系

### 9.1 任务指标

- 检测概率 $P_D$；
- 固定虚警率下的检测率；
- Precision、Recall、Macro-F1；
- 时频框 mAP；
- 时频边界误差；
- 类别准确率和混淆矩阵；
- 发射源定位误差；
- 开放集 AUROC/FPR95；
- 置信度校准 ECE。

### 9.2 通信指标

- bit/帧；
- 平均和峰值码率；
- 压缩比；
- 包错误率；
- 端到端时延；
- 固定带宽可支持 UAV 数量；
- 原始证据回传比例。

### 9.3 计算与能耗指标

- 参数量；
- MACs/FLOPs；
- 峰值 RAM/VRAM；
- 编码和解码时间；
- 单帧推理能量；
- 采样、计算、通信总能耗。

### 9.4 系统级指标

$$
\mathrm{Semantic\ Efficiency}
=\frac{\mathrm{Task\ Utility}}{\mathrm{Transmitted\ Bits\ or\ Energy}}.
$$

最终应报告任务性能—码率—时延—能耗的 Pareto 前沿，而不是单一压缩比。

## 10. 预期论文拆分

### 论文一：单 UAV 数字语义编码

> Digital Task-Oriented Semantic Communication for Wideband I/Q Spectrum Perception

内容：宽带多目标检测、数字语义码流、固定/可变码率、信道鲁棒性。

### 论文二：多 UAV 协同 ISCC

> Uncertainty-Aware Collaborative Semantic Spectrum Sensing for Multi-UAV Networks

内容：多 UAV 语义融合、节点选择、感知时间、码率和能量联合优化。

### 论文三：真实系统验证（可选）

> A Real-World SDR Testbed for Task-Oriented Aerial Spectrum Semantic Communication

内容：SDR 数据、嵌入式部署、仿真—现实迁移、真实吞吐与功耗。

## 11. 项目代码迁移建议

建议新增独立模块，不直接覆盖现有深空遥测实现：

```text
src/spectrum/
  iq_dataset.py
  iq_preprocessing.py
  sensing_channel.py
  reporting_channel.py
  spectrum_detector.py
  semantic_codec.py
  quantizer.py
  packetizer.py
  multi_uav_fusion.py
  scheduler.py
  spectrum_metrics.py
```

可复用：

- `src/jscc_telemetry.py` 的训练框架思想；
- `src/channel.py` 的信道接口；
- `src/metrics.py` 和现有实验脚本组织方式；
- 测试、结果输出和绘图结构。

需要替换：

- 600 点实数遥测窗口 → 长复数 I/Q；
- MSE 主指标 → 检测、分类和时频定位；
- 深空信道 → 感知信道 + UAV 上报信道；
- 固定 200 bit → 可变长度数字码流；
- 手工 16 维语义 → 任务学习 Token；
- CCSDS → 实际数字分包和 LDPC/5G 风格链路。

DDPM 暂不进入主线，仅可在后期用于非关键可视化或作为对照，不能用于生成关键频谱证据。

## 12. 硬件与数据策略

推荐开发配置：

- GPU 显存：12—16GB；
- 系统内存：64GB 推荐，32GB 可起步；
- SSD：2TB NVMe 推荐；
- 当前机器可用于代码、小样本和传统基线，完整训练使用云端 16/24GB GPU。

降低资源需求的原则：

- 不将 100 万点整帧直接送入大型 Transformer；
- 先分块局部编码，再做 Token 级融合；
- 使用混合精度、梯度累积和梯度检查点；
- 多 UAV 信道在线生成，不复制数据集；
- 本地编码器预训练后可冻结，再训练融合器；
- 数据用分块格式流式读取。

## 13. 主要风险与止损策略

| 风险 | 表现 | 止损/替代方案 |
|---|---|---|
| 数据过度合成 | 模型学习生成器特征 | 提前采集少量 SDR 数据，做跨域测试 |
| 语义模型不优于简单统计量 | 收益不足以覆盖计算开销 | 转向语义调度和节点选择 |
| 多 UAV 规模过大 | 显存和训练时间失控 | 先独立编码，再做 Token 融合 |
| AirComp 同步过难 | 仿真假设不现实 | 第一、二篇均采用数字 OFDMA/TDMA |
| 强化学习不稳定 | 难复现、难解释 | 优先可微优化、监督调度和 Lyapunov |
| 真实发射受法规限制 | 无法开展外场实验 | 同轴闭环、屏蔽环境、合法 ISM 或被动接收 |
| 未知类别被错误高置信分类 | 安全风险 | 开放集检测、不确定性、原始证据回传 |

## 14. 最小可发表版本

若时间和硬件有限，最小版本只包含：

1. RadDet 数据；
2. 单 UAV 与 2—4 UAV 仿真；
3. 宽带检测、时频定位和分类；
4. 数字语义 Token；
5. 感知和上报信道分离；
6. 严格 payload 统计；
7. 与硬决策、软决策、压缩感知、AutoEncoder 和完整信息检测器比较；
8. 无泄漏数据划分和完整消融。

轨迹优化、AirComp、联邦学习、真实飞行和生成式重建均为增强项，不应阻塞第一篇论文。

## 15. 参考文献

[1] C. E. Shannon, “A Mathematical Theory of Communication,” *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, 1948.  
原文链接：https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

[2] S. Haykin, “Cognitive Radio: Brain-Empowered Wireless Communications,” *IEEE Journal on Selected Areas in Communications*, vol. 23, no. 2, pp. 201–220, 2005.  
原文链接：https://doi.org/10.1109/JSAC.2004.839380

[3] T. Yucek and H. Arslan, “A Survey of Spectrum Sensing Algorithms for Cognitive Radio Applications,” *IEEE Communications Surveys & Tutorials*, vol. 11, no. 1, pp. 116–130, 2009.  
原文链接：https://doi.org/10.1109/SURV.2009.090109

[4] I. F. Akyildiz, B. F. Lo, and R. Balakrishnan, “Cooperative Spectrum Sensing in Cognitive Radio Networks: A Survey,” *Physical Communication*, vol. 4, no. 1, pp. 40–62, 2011.  
原文链接：https://doi.org/10.1016/j.phycom.2010.12.003

[5] Z. Tian and G. B. Giannakis, “Compressed Sensing for Wideband Cognitive Radios,” in *Proceedings of IEEE ICASSP*, 2007.  
原文链接：https://doi.org/10.1109/ICASSP.2007.367330

[6] M. Mishali and Y. C. Eldar, “Wideband Spectrum Sensing at Sub-Nyquist Rates,” *IEEE Signal Processing Magazine*, vol. 28, no. 4, pp. 102–135, 2011.  
原文链接：https://doi.org/10.1109/MSP.2011.941094

[7] Y. Zeng, R. Zhang, and T. J. Lim, “Wireless Communications with Unmanned Aerial Vehicles: Opportunities and Challenges,” *IEEE Communications Magazine*, vol. 54, no. 5, pp. 36–42, 2016.  
原文链接：https://doi.org/10.1109/MCOM.2016.7470933

[8] M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A Tutorial on UAVs for Wireless Networks: Applications, Challenges, and Open Problems,” *IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2334–2360, 2019.  
原文链接：https://doi.org/10.1109/COMST.2019.2902862

[9] U. Challita, W. Saad, and C. Bettstetter, “Machine Learning for Wireless Connectivity and Security of Cellular-Connected UAVs,” *IEEE Wireless Communications*, vol. 26, no. 1, pp. 28–35, 2019.  
原文链接：https://doi.org/10.1109/MWC.2018.1800155

[10] T. J. O’Shea, T. Roy, and T. C. Clancy, “Over-the-Air Deep Learning Based Radio Signal Classification,” *IEEE Journal of Selected Topics in Signal Processing*, vol. 12, no. 1, pp. 168–179, 2018.  
原文链接：https://doi.org/10.1109/JSTSP.2018.2797022

[11] N. West, T. O’Shea, and T. Roy, “A Wideband Signal Recognition Dataset,” in *Proceedings of IEEE SPAWC*, 2021.  
原文链接：https://ieeexplore.ieee.org/document/9593265  
arXiv 链接：https://arxiv.org/abs/2110.00518

[12] Z. Huang et al., “RadDet: A Wideband Dataset for Real-Time Radar Spectrum Detection,” in *Proceedings of IEEE ICASSP*, 2025.  
原文链接：https://doi.org/10.1109/ICASSP49660.2025.10887772  
arXiv 链接：https://arxiv.org/abs/2501.10407

[13] H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, “Deep Learning Enabled Semantic Communication Systems,” *IEEE Transactions on Signal Processing*, vol. 69, pp. 2663–2675, 2021.  
原文链接：https://doi.org/10.1109/TSP.2021.3071210

[14] W. Yang et al., “Semantic Communications for Future Internet: Fundamentals, Applications, and Challenges,” *IEEE Communications Surveys & Tutorials*, vol. 25, no. 1, pp. 213–250, 2023.  
原文链接：https://doi.org/10.1109/COMST.2022.3223224

[15] Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, “Semantic Communications: Principles and Challenges,” arXiv preprint, 2021/2022.  
原文链接：https://arxiv.org/abs/2201.01389

[16] E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, “Deep Joint Source-Channel Coding for Wireless Image Transmission,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 5, no. 3, pp. 567–579, 2019.  
原文链接：https://doi.org/10.1109/TCCN.2019.2919300

[17] J. Shao, Y. Mao, and J. Zhang, “Task-Oriented Communication for Multi-Device Cooperative Edge Inference,” *IEEE Transactions on Wireless Communications*, vol. 22, no. 1, pp. 73–87, 2023.  
原文链接：https://doi.org/10.1109/TWC.2022.3191118

[18] F. Liu et al., “Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond,” *IEEE Journal on Selected Areas in Communications*, vol. 40, no. 6, pp. 1728–1767, 2022.  
原文链接：https://doi.org/10.1109/JSAC.2022.3156632

[19] J. A. Zhang et al., “An Overview of Signal Processing Techniques for Joint Communication and Radar Sensing,” *IEEE Journal of Selected Topics in Signal Processing*, vol. 15, no. 6, pp. 1295–1315, 2021.  
原文链接：https://doi.org/10.1109/JSTSP.2021.3113120

[20] P. Yi, Y. Cao, X. Kang, and Y.-C. Liang, “Integrated Distributed Semantic Communication and Over-the-Air Computation for Cooperative Spectrum Sensing,” *IEEE Transactions on Communications*, 2025.  
原文链接：https://doi.org/10.1109/TCOMM.2024.3468215
