# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**语言 / Language**：[简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md) | [English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md)

## 影像生境分析的标准化瓶颈与开源工具HABIT

影像组学（Radiomics）是指从医学影像（如CT、MRI、PET-CT等）中高通量地提取大量定量特征，并将这些特征与临床结局、基因表型或病理信息进行关联分析的一门方法学，从2012年提出至今已发表数万篇论文。肿瘤内部细胞密度、灌注、坏死、乏氧、免疫浸润及基质反应等并非空间均质的。Radiomics的缺陷在于它将肿瘤作为整体，将异质性极高的肿瘤压缩为一组全局特征，丢失了空间信息。近年来，学界提出了影像生境分析。其范式突破在于：在体素水平上将肿瘤划分为多个子区域（称为生境或Habitat），每个生境具有相似的生理病理特征。这种方法能够在术前、活体条件下，无创地量化肿瘤的空间异质性。该思路源于生态学，核心假设是影像特征相似的生境共享相近的肿瘤生物学行为。不同生境对应不同的肿瘤微环境选择压力与适应策略，部分承载更具侵袭性、更难治的种群，且生境间交互亦可改变肿瘤生物学行为。该方法被称为Radiomics++，即影像组学的高级或补充形式。尽管已有数百篇论文发表且近年呈指数增长，但其仍处早期阶段，流程复杂，缺乏标准化协议和分析工具，已成为阻碍影像生境分析走向可重复、可验证及临床转化的瓶颈。基于此，我们历时2年多开发了HABIT (Habitat Analysis: Biomedical Imaging Toolkit)，一个开源影像生境分析工具，旨在为影像生境分析提供一个标准化、可重复、可扩展、可嵌入生态的分析框架。

## HABIT的特色与功能

据我们检索公开仓库与文献，目前尚无专注于影像生境分析的完整开源工具。零星有代码是针对单篇研究的脚本，绑定该研究的特征与划分规则，不能通用，也缺乏可独立使用的文档。国内闭源商业平台在影像组学教培领域取得较大成功，已被数百篇论文使用，其支持部分生境分析，但其核心细节不开源，不可审查、不可扩展，也不易作为可引用模块嵌入既有影像组学生态。

HABIT从设计之初即以可审查、可重复、可扩展、可嵌入生态为约束。源码与计算条件可供核对，同一协议可在不同队列上复现，环节可替换，并可作为模块接入已有影像组学流程。软件全面开源并附详细文档，即将进入公测和宣讲阶段。历经2年多开发，即将进入全面公测与宣传。源码：https://github.com/lichao312214129/HABIT 。在线文档：https://lichao312214129.github.io/HABIT/ 。用户可以通过 `pip install habitat-analysis` 安装，也可以通过源码安装。

```bash
pip install habitat-analysis
```

HABIT的功能可归为七项。

1. 体素纹理提取与学术界标准对齐，并提供极致的时间效率。研究专用脚本与商业平台的体素纹理特征提取未声明与IBSI对齐，且计算速度慢。HABIT的特征公式严格遵循IBSI标准，并与PyRadiomics对齐。HABIT内置CUDA加速的体素纹理提取方法。同机对照（NVIDIA GeForce RTX 3070 Laptop，binWidth=25，kernelRadius=1，80084个体素）中，PyRadiomics体素路径耗时414.58秒，HABIT经CUDA加速后为7.59秒。

2. 体素特征预处理与生境划分。研究专用脚本与商业平台一般不支持体素特征的预处理与筛选。生境划分通常仅提供一或两种规则。HABIT内置学术界常用的三种划分，即超体素后聚类（two_step_habitat）、例内体素聚类（one_step_habitat）与队列体素池化聚类（direct_pooling_habitat）。各步骤以原子操作提供，使用方可组合策略。训练集fit得到共享的预处理参数、质心与分配规则；测试集predict在同一特征空间生成生境图。

3. 体素特征稳定性分析。已有研究表明，体素纹理特征中相当一部分在重测或扰动下不稳定。HABIT内置稳定特征筛选模块。

4. 生境图谱定量。HABIT支持全面的生境图谱定量特征提取，其中包括自主开发的全量图论特征。HABIT严格依据学术界已发表的标准定义各种特征，并在文档中公开全部计算公式和API。

5. 扩展接口。HABIT的所有核心模块均可替换和扩展，软件整体可作为模块接入既有影像组学流程。研究专用脚本更换步骤时，需改写该研究的流程代码；商业平台将流程封装于界面或作业系统，难以嵌入既有预处理或统计流程。HABIT以协议与注册表规定可替换步骤，上层接口为Study、HabitatSpec与YAML，下层接受SimpleITK影像、numpy数组、文件路径与路径列表。

6. 队列执行。HABIT支持大队列的连续运行。体素提取与按例计算可使用进程池；任务中断后从已完成病例继续运行；单例失败被隔离并留下记录，不中断整个队列。

7. 文档。HABIT附有覆盖全流程的在线文档。HABIT在线文档覆盖安装、Quickstart、Habitat Analysis Guide、特征定义（体素纹理、MSI、ITH、图论）并附可运行示例。

## 引用与许可

- 问题：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- 引用：[CITATION.cff](CITATION.cff) · [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- 许可：HABIT 本体为 [Apache-2.0](LICENSE)。共识聚类模块 `habit/third_party/inmoose/` 为 GPL-3.0-or-later（InMoose / Sajovic），不随 Apache-2.0 重新许可。再分发时保留版权、许可声明与 [NOTICE](NOTICE)。用于科研时请引用 HABIT。
