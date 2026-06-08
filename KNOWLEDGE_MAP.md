# RAG增强深空遥测语义传输系统 — 知识点清单

> 覆盖本系统所涉及的全部知识点，按学科领域分类，详细到每个概念、公式、工具。
> 适用于硕士论文"理论基础"章节撰写和答辩准备。

---

## 第一类：深空通信基础

### 1.1 深空通信环境
- 自由空间路径损耗 (Free-Space Path Loss, FSPL):
  `FSPL(dB) = 20*log10(4*pi*d/lambda)`，其中 d 为通信距离，lambda 为载波波长
- 通信距离：0.5-3.0 AU（天文单位），1 AU = 1.496e8 km
- X-band 载波频率：~8 GHz，波长 lambda ≈ 0.0375 m
- 典型接收信噪比：-170 dB 至 -130 dB
- 信号传播延迟：火星-地球单向 4-24 分钟
- 真空信道特性：无多径效应（无大气层反射）、无雨衰

### 1.2 加性高斯白噪声 (AWGN)
- 定义：w[n] ~ N(0, sigma_w^2)，独立同分布
- 噪声功率与SNR关系：sigma_w^2 = P_signal / 10^(SNR/10)
- 对MSE的影响：直接传输时 MSE = sigma_w^2
- 高斯分布的概率密度函数：f(x) = (1/(sigma*sqrt(2*pi)))*exp(-x^2/(2*sigma^2))

### 1.3 多普勒频移效应
- 物理机制：探测器-地球相对运动引起的频率偏移
- 取值范围：±50000 Hz（深空探测器典型径向速度）
- 对基带信号的影响（菲涅尔旋转）：
  正交频移公式 `y[n] = x[n]*cos(2*pi*fd*n/fs) - H{x}[n]*sin(2*pi*fd*n/fs)`
- Hilbert变换 H{·}：实信号的解析表示，`x_a(t) = x(t) + j*H{x}(t)`
- 多普勒补偿：施加逆频移 `DopplerShift(-fd)` 恢复原始频率

### 1.4 太阳闪烁 (Solar Scintillation)
- 物理机制：太阳风等离子体引起的信号幅度和相位随机起伏
- 闪烁指数 m4 ∈ [0, 0.5]，m4 < 0.3 为弱闪烁，> 0.3 为强闪烁
- 闪烁指数与太阳角的负相关关系
- 有色噪声的生成：白噪声通过Butterworth低通滤波器
- CCSDS 401.0-B 标准中的闪烁模型

### 1.5 CCSDS 推荐标准
- CCSDS 121.0-B-3：无损数据压缩（Rice编码）
  - 预处理器：单位延迟预测器 delta[n] = x[n] - x[n-1]
  - 块自适应熵编码器：J样本/块（默认J=16），选最优Rice参数k
  - Rice编码：quotient = floor(|delta|/2^k)，unary编码商，k-bit编码余数，1-bit符号
  - 参考样本间隔(RSI)：每J个样本插入一个原始参考样本用于错误恢复
  - 典型压缩比：1.5:1 至 2.0:1
- CCSDS 401.0-B：射频与调制系统标准

---

## 第二类：信号处理

### 2.1 统计矩 (Statistical Moments)
- 一阶矩（均值）：mu = E[y] = (1/N)*sum(y_i)
- 二阶矩（方差/标准差）：sigma^2 = E[(y-mu)^2], sigma = sqrt(sigma^2)
- 三阶标准化矩（偏度 Skewness）：gamma_1 = E[(y-mu)^3] / sigma^3
  - 物理含义：分布的不对称程度，gamma_1=0为对称分布
- 四阶标准化矩（峰度 Kurtosis）：gamma_2 = E[(y-mu)^4] / sigma^4 - 3
  - 物理含义：分布的尾部厚度，gamma_2=0为正态分布
  - 超值峰度(Excess Kurtosis) = gamma_2 - 3，减去3便于与正态分布比较
- 数值稳定性：scipy.stats.skew() 和 kurtosis() 的精度损失问题

### 2.2 互相关 (Cross-Correlation)
- 定义：R_xy[tau] = sum(x[n]*y[n+tau])
- 离散互相关的三种模式：'full'(2N-1点), 'same'(N点), 'valid'(N-M+1点)
- FFT加速实现：R_xy = IFFT(FFT(x)*conj(FFT(y)))，复杂度O(N log N)
- 归一化互相关：R_norm = R_xy / (N*sigma_x*sigma_y)，值域[-1, 1]
- 峰值检测：argmax(R_norm) 确定最优滞后量
- 循环移位对齐：np.roll(template, lag)

### 2.3 欧氏距离 (Euclidean Distance)
- 定义：d(q, t) = sqrt(sum((q_i - t_i)^2))
- 均方误差(MSE)：d^2/N = (1/N)*sum((q_i - t_i)^2)
- 几何意义：L2范数在信号空间中的距离
- 对相位敏感的局限性：相同波形错位 → 大距离 → 需要相位对齐

### 2.4 动态时间规整 (Dynamic Time Warping, DTW)
- 目的：度量两个时间序列在时间轴弯曲后的相似度
- 累积距离矩阵：D[i,j] = d(q_i, t_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
- Sakoe-Chiba窗口约束：限制弯曲路径在|i-j| <= window 范围内
  - 作用：防止病态匹配 + 降低计算复杂度（O(N^2) → O(N*W)）
  - 本系统默认窗口=信号长度的10%
- dtaidistance库：C扩展实现，window参数传递

### 2.5 频谱分析
- 快速傅里叶变换 (FFT)：时域 → 频域，O(N log N)
- 实信号的RFFT：numpy.fft.rfft() 仅计算正频率分量
- 频谱幅值：abs(FFT(y))，用于频谱包络距离
- 频谱归一化：频谱/总能量，消除幅值对频谱距离的影响
- 频谱包络距离：d_spec(q,t) = mean((|FFT(q)|_norm - |FFT(t)|_norm)^2)

### 2.6 希尔伯特变换 (Hilbert Transform)
- 定义：H{x}(t) = (1/pi)*PV(integral(x(tau)/(t-tau), dtau))
- 频率响应：H(omega) = -j*sign(omega)
- 解析信号：x_a(t) = x(t) + j*H{x}(t)
- 在多普勒频移中的应用：正交调制/解调的虚部分量
- scipy.signal.hilbert() 实现

---

## 第三类：机器学习与信息检索

### 3.1 检索增强生成 (Retrieval-Augmented Generation, RAG)
- 经典RAG流程：查询 → 检索(Retrieve) → 增强(Augment) → 生成(Generate)
- 本系统的RAG映射：
  - 查询 = 接收到的含噪信号 y_received
  - 知识库 = 信号模板库 (SQLite + FAISS)
  - 检索 = 统计特征粗筛 + 信号精排
  - 增强 = 相位对齐 + 残差计算
  - 生成 = y_recon = y_template + residual

### 3.2 向量检索与近似最近邻 (ANN)
- FAISS (Facebook AI Similarity Search)：倒排文件索引 (IVF)
- IVF原理：用K-means聚类将向量空间划分为Voronoi单元，查询时只搜索最近单元
- 本系统限制：统计特征仅4维 → 暴力扫描 5000×4=20000 次乘加 → 不需要ANN
- 数据库规模与检索策略：N<10000时暴力扫描更优（无索引开销）

### 3.3 距离度量
- L2距离（欧氏距离）：sqrt(sum((q_i-t_i)^2))
- DTW距离（动态时间规整）：允许非线性时间对齐
- 多维度加权融合：
  d_multi = w_euc*d_euc + w_spec*d_spec + w_stat*d_stat
  (w_euc=0.5, w_spec=0.3, w_stat=0.2)
- 距离度量的选择原则：不同信号类型的最优度量可能不同

### 3.4 特征工程
- 统计特征的设计：为何选 [mean, std, skew, kurt]？
  - mean/std捕获信号的整体位置和尺度
  - skew捕获信号的不对称性（脉冲尖峰）
  - kurt捕获信号的尾部行为（噪声vs周期信号）
- 统计特征 vs 物理参数的信息含量对比
- 特征向量的归一化：不需要（信号已同分布）

### 3.5 评估指标
- 精确率/召回率：
  - Recall@K = (最优模板在Top-K候选中的概率)
  - Recall@1 是最严格的标准
- MSE (Mean Squared Error)：重建质量
- 修剪均值 (Trimmed Mean)：trim_mean(arr, 0.05)
  - 去除首尾各5% → 对抗离群值
  - vs 中位数：中位数完全忽略分布形状，修剪均值保留了分布信息
- 残差方差：var(y_original - y_template)，可压缩性指标

---

## 第四类：信息论与编码

### 4.1 均匀量化 (Uniform Quantization)
- 量化步长：step = 2*x_max / (2^bits - 1)
- 量化噪声方差：sigma_q^2 ≈ step^2 / 12（高斯近似）
- 量化信噪比：SQNR ≈ 6.02*bits + 1.76 dB
- 过载/限幅：值超出[-x_max, x_max]时的clip误差

### 4.2 率失真理论 (Rate-Distortion Theory)
- 率失真函数 R(D)：在给定失真D下所需的最小比特率
- 高斯信源的率失真函数：R(D) = 0.5*log2(sigma^2/D)，当 D <= sigma^2
- 本系统的应用：由R(D)推导最优量化比特数 b* = 0.5*log2(12*sigma_r^2/epsilon) + 1

### 4.3 熵编码 (Entropy Coding)
- Rice编码（Golomb编码的特例，m=2^k）：
  - 商 q = floor(x / 2^k)，unary编码(q个0 + 1个1)
  - 余数 r = x mod 2^k，k-bit二进制编码
  - 符号位：1-bit（正=0, 负=1）
- Unary编码：数字n编码为n个0 + 1个1
- 最优Rice参数k的选择：遍历k=0..14，选择总比特最小的k
- CCSDS 121.0的分块策略：每J=16个样本一块，参考样本+预测残差

### 4.4 带宽效率
- bits per sample：B_total / N
- 压缩比：B_PCM / B_ours
- PCM 8-bit基准：600样本 × 8 bit = 4800 bit
- 模板ID开销：ceil(log2(N_templates)) bit
- 残差比特：N_samples × n_bits

### 4.5 信道容量
- Shannon信道容量：C = B*log2(1 + SNR)
- 深空信道SNR极低 (-170至-130 dB)，C极低
- 语义通信的优势：传输"含义"而非"原始比特"，低于Shannon极限也能通信
- 信息年龄 (Age of Information, AoI)：衡量信息时效性的指标

---

## 第五类：概率论与统计学

### 5.1 期望与方差
- 期望（一阶矩）：E[X] = integral(x*f(x), dx)，离散：sum(x_i*p_i)
- 方差（二阶中心矩）：Var[X] = E[(X-E[X])^2] = E[X^2] - (E[X])^2
- 独立随机变量和的方差：Var[X+Y] = Var[X] + Var[Y]（协方差=0）
- 全期望公式：E[X] = E[E[X|Y]] = sum(E[X|Y=y]*P[Y=y])

### 5.2 不等式工具
- Minkowski不等式：||x+y||_p <= ||x||_p + ||y||_p（三角不等式的推广）
  - 在本系统中用于推导残差功率上界
- Cauchy-Schwarz不等式：|<x,y>| <= ||x||*||y||
- Jensen不等式：f(E[X]) <= E[f(X)]（凸函数）

### 5.3 蒙特卡洛方法
- 蒙特卡洛积分：E[f(X)] ≈ (1/N)*sum(f(X_i))（大数定律）
- 随机种子的重要性：复现性
- 样本量的选择：200样本 → 标准误差 ≈ sigma/sqrt(200) ≈ 0.07*sigma

### 5.4 噪声模型
- 高斯噪声：n ~ N(0, sigma^2)，最常见的自然噪声模型
- 均匀分布：U(a, b)，用于模拟刻度误差等非高斯因素
- 1/f 噪声（粉红噪声）：功率谱密度与频率成反比，自然界广泛存在
  - 生成方法：频域法——频域生成随机频谱，乘以1/sqrt(f)，逆FFT
- 有色噪声 vs 白噪声：有色噪声具有频率相关性

---

## 第六类：线性代数

### 6.1 向量范数
- L2范数（欧氏范数）：||x||_2 = sqrt(sum(x_i^2))
- L2范数的平方：||x||^2 = sum(x_i^2) = N * MSE
- 向量内积：<x, y> = sum(x_i*y_i)，与L2范数的关系：||x||^2 = <x, x>
- 信号功率：P = (1/N)*||x||^2

### 6.2 矩阵运算
- 向量减法：残差 = y - t（逐元素运算）
- 逐元素乘法：Hadamard积（频谱归一化）
- 广播机制：NumPy自动对齐不同形状的数组

### 6.3 FFT与卷积
- 卷积定理：FFT(x * y) = FFT(x) · FFT(y)
- 互相关与卷积的关系：R_xy[tau] = x[-n] * y[n]（翻转后卷积）
- 循环卷积 vs 线性卷积：FFT默认循环卷积，补零可实现线性卷积

---

## 第七类：软件工程

### 7.1 Python生态
- NumPy：多维数组操作（向量化、广播、np.mean/np.std/np.var）
- SciPy：
  - scipy.stats：skew(), kurtosis()
  - scipy.signal：hilbert(), butter(), filtfilt()
  - scipy.stats.trim_mean：修剪均值
- FAISS-CPU：近似最近邻搜索
- dtaidistance：动态时间规整
- Matplotlib：论文级图表（semilogy对数坐标、多子图布局、颜色映射）
- tqdm：进度条
- pytest：单元测试框架

### 7.2 数据库
- SQLite：嵌入式关系型数据库
  - 内存模式 (':memory:')：无磁盘IO
  - SQL范围查询：WHERE x BETWEEN a AND b
  - 游标与事务管理
- 物理参数索引：在distance_au, sun_earth_probe_angle, snr_db上建联合查询

### 7.3 设计模式
- 策略模式：CoarseRetriever/FineRetriever/HierarchicalRetriever
- 工厂模式：generate_telemetry_segment()按signal_type分发生成器
- 观察者模式：FineRetriever存储last_aligned_templates供reconstruction使用
- 单例模式（隐式）：KnowledgeBase在实验中通常只构建一次

### 7.4 实验管理
- Git版本控制：18次提交，原子化粒度
- 实验结果持久化：np.save()中间数据 + PNG图表
- 参数化实验：argparse命令行参数
- 模型失配参数化：eta ∈ [0, 1]控制失配强度
- 可复现性：seed参数贯穿所有随机过程

---

## 第八类：深空探测器与任务知识

### 8.1 典型深空任务
- 火星科学实验室 (MSL/Curiosity)：2011年发射，火星表面漫游车
- MAVEN：火星大气与挥发物演化探测器
- SMAP：土壤水分主被动探测卫星（地球轨道）
- 深空网络 (DSN)：NASA的三个地面站（Goldstone, Madrid, Canberra）

### 8.2 探测器约束
- 功耗预算：MSL计算机系统 < 100W
- 数据率：X-band下行 0.5-32 kbps（根据距离变化）
- 存储：MSL有2GB闪存用于数据缓存
- 抗辐射要求：总剂量 > 100 krad

### 8.3 遥测数据类型
- 工程遥测 (Housekeeping)：温度、电压、电流、压力
- 姿控遥测 (AOCS)：陀螺仪、星敏感器、反作用轮
- 科学数据：相机图像、光谱仪、粒子探测器
- 根据NASA telemanom数据集的通道命名规范：A-x, B-x, C-x等

---

## 第九类：数学分析

### 9.1 优化理论
- 穷举搜索：遍历所有候选，取最小值
- 网格搜索：在物理参数空间均匀采样
- 贪心策略：粗筛取Top-K → 精排取Top-3
- 最优量化参数：遍历k=0..14取最小比特的k值（全局最优，穷举即可）

### 9.2 渐近分析
- O(N) vs O(N log N) vs O(N^2)：算法复杂度对实时性的影响
- CCSDS 121.0编码复杂度：O(N)，每个样本常数时间
- DTW复杂度：O(N*W)，W为Sakoe-Chiba窗口宽度

### 9.3 数值分析
- 浮点精度：float32 vs float64的取舍（本系统用float32存信号，float64算统计量）
- 灾难性抵消 (Catastrophic Cancellation)：两个接近的大数相减导致精度损失
  - 在本系统中出现于：scipy.stats.skew/kurtosis 对近恒定信号的计算
- 避免除零：分母加epsilon (1e-8 ~ 1e-12)

---

## 第十类：论文写作与方法论

### 10.1 实验设计方法论
- 控制变量法：每次只改变一个因素（信号类型、SNR、失配强度）
- 基线对比：7种检索策略横评，No Retrieval作为绝对底线
- 消融实验：逐个去掉系统组件，验证每个组件的贡献
- 鲁棒性测试：3级模型失配 + 真实数据验证

### 10.2 论文结构
- 摘要：背景+方法+结果+关键词
- 绪论：背景→问题→相关工作→本文方案→贡献
- 系统架构：模块划分+数据流+技术栈
- 理论分析：命题→证明→实验验证
- 实验：环境→指标→设计→结果→分析
- 讨论：对比+局限+启示
- 结论：总结+展望

### 10.3 诚实报告
- 负结果的价值：多维度加权无增益、DTW不改变排序、periodic相位偏移瓶颈
- 局限性的披露：合成数据的理想化、真实数据归一化、无飞行验证
- 与CCSDS的关系：增强式而非替代式

---

## 知识点交叉矩阵

| 知识点 | telemetry | channel | knowledge | retrieval | reconstruction | preprocessing | ccsds121 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FSPL | | ✓ | | | | | |
| AWGN | | ✓ | | | ✓ | | |
| Doppler | | ✓ | | | | ✓ | |
| Scintillation | | ✓ | | | | | |
| 统计矩 | | | ✓ | | | | |
| 互相关 | | | | | | ✓ | |
| DTW | | | | ✓ | | | |
| RAG框架 | | | ✓ | ✓ | ✓ | | |
| FAISS | | | ✓ | | | | |
| 均匀量化 | | | | | ✓ | | |
| Rice编码 | | | | | | | ✓ |
| Minkowski不等式 | | | | | | | |
| SQLite | | | ✓ | ✓ | | | |
