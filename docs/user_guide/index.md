# HABIT 用户指南

欢迎使用 HABIT (Habitat Analysis Tool for Medical Images)，这是一个用于医学图像生境分析的综合工具包。本指南将介绍如何安装和使用 HABIT 的各种功能。

## 目录

1. [安装指南](installation.md)
2. [快速入门](quickstart.md)
3. [生境映射](habitat_mapping.md)
4. [特征提取](feature_extraction.md)
5. [测试-重测分析](test_retest.md)
6. [ICC分析](icc_analysis.md)
7. [机器学习](machine_learning.md)
8. [配置文件](configuration.md)
9. [常见问题解答](faq.md)

## 简介

HABIT是一个为医学图像分析设计的工具包，专注于生境分析方法。生境分析是一种基于医学图像内部异质性的分析方法，可以发现和量化组织内部不同区域的特征，这些区域被称为"生境"。

本工具包完整实现了从图像处理、聚类分析、特征提取到机器学习的端到端流程。

## 基本概念

生境分析通常包含以下几个主要步骤：

1. **生境地图生成**：使用聚类算法将组织划分为多个生境区域
2. **特征提取**：从各个生境区域提取放射组学特征
3. **测试-重测分析**：评估生境分析的可重复性
4. **ICC分析**：计算特征的组内相关系数，筛选稳定特征
5. **机器学习分析**：使用生境特征构建预测模型

## 工作流程

典型的HABIT工作流程如下：

1. 首先使用`generate_habitat_map.py`生成生境地图
2. 然后使用`extract_features.py`提取生境特征
3. 如果有测试-重测数据，使用`habitat_test_retest_mapper.py`匹配生境标签
4. 使用`icc_analysis.py`计算ICC值，评估特征稳定性
5. 最后使用`run_machine_learning.py`构建机器学习模型

请参考各个章节了解每个步骤的详细说明。 