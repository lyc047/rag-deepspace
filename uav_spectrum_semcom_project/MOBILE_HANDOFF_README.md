# UAV频谱语义通信项目：移动端交接说明

交接日期：2026-07-17  
项目：`uav_spectrum_semcom_project`  
用途：在离开电脑期间，使用同一账号的ChatGPT手机端继续做代码阅读、实验审计和论文分析。

## 1. 当前研究状态

- 论文主线：面向多无人机协同宽带频谱感知的任务导向数字语义通信与频谱资源选择；
- 核心正结果：C1资源选择后悔感知语义生成通过开发集Gate A；
- 负结果：C2反事实价值调度未通过Gate B，C3可靠少数/补传未通过Gate C；
- 经典保留方案：选择性G2在12个开发信道点均节省实际发送bit，任务非劣性等待final；
- 工程状态：149项测试通过，开发验收通过，0项未解释失败；
- final状态：0/200个新增独立scene，访问计数0，算法、阈值和统计协议保持冻结。

## 2. 交接文件

打包脚本会在`mobile_handoff_20260717/`中生成：

1. `UAV_MOBILE_SOURCE_CODEBOOK.md`  
   将源码、脚本、配置、测试和关键研究文档合并为一个可搜索文本文件。手机端只上传一个文件即可开始代码分析。

2. `uav_spectrum_semcom_code_docs_20260717.zip`  
   包含源码、脚本、配置、测试、研究文档和关键论文图表。

3. `HANDOFF_MANIFEST.json`、`HANDOFF_MANIFEST.csv`和`HANDOFF_SHA256SUMS.txt`  
   记录每个文件的相对路径、大小、SHA256以及压缩包自身的SHA256。

## 3. 为什么不包含`data/`

`data/`约2.23 GB、超过10万文件，主要是第三方原始数据和缓存，不属于本次代码与文档交接范围。移动端代码分析所需的数据身份、split、来源边界和泄漏规则已经保存在`configs/`和`docs/`中。

不包含原始数据还可避免：

- 超过ChatGPT账号或移动网络上传限额；
- 把开发数据误认为独立final；
- 重复分发第三方数据；
- 在手机端分析时被大量样本文件淹没。

如后续需要复现实验，应回到本机数据目录或使用原始公开数据入口，不应从移动交接包重建数据。

## 4. 手机端推荐使用顺序

### 最轻量方式

1. 在ChatGPT中新建项目，例如“UAV频谱语义通信代码分析”；
2. 上传`UAV_MOBILE_SOURCE_CODEBOOK.md`；
3. 再上传`docs/STAGE4_PAPER_STYLE_OVERVIEW.md`或完整包中的同名文件；
4. 粘贴`MOBILE_GPT_ANALYSIS_PROMPT.md`中的提示词；
5. 先要求建立文件—算法—实验映射，再逐模块分析。

### 完整方式

若手机端支持解压ZIP，可上传代码文档包；若ZIP无法直接解析，直接使用Codebook即可。研究数值和结论优先读取`docs/STAGE4_PAPER_STYLE_OVERVIEW.md`、各Batch报告和`docs/STAGE4_DEVELOPMENT_SYNTHESIS.md`。

## 5. 分析时必须遵守的证据边界

- validation正结果不能写成final证明；
- C1是当前唯一通过性能门槛的核心算法候选；
- C2和C3是完整负结果，不能包装成正性能创新；
- report delivery ratio不能替代解码后regret、clean rate和Brier；
- 受控多节点视图不能证明真实H3空间协同；
- 理论IQ容量不能写成真实codec压缩结果；
- 本系统不是MIMO或AirComp系统；
- 图像语义不是论文主线；
- 在新增独立final之前，不得根据任何拟似final输出修改算法。

## 6. 桌面端重新构建

在项目根目录运行：

```powershell
python scripts/build_mobile_handoff.py
```

脚本会重建代码文档ZIP、Codebook、清单和哈希，并执行ZIP CRC检查。输出包不会包含`data/`、实验缓存、模型checkpoint、临时文件、私钥或环境变量文件。
