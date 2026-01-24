架构设计
========

HABIT 采用模块化设计，主要包含以下核心模块：

核心模块
--------

.. image:: ../static/images/architecture.png
   :alt: HABIT Architecture
   :align: center

Habitat Analysis
^^^^^^^^^^^^^^^

负责肿瘤生境分析的核心模块，包括：

* **FeatureManager**: 特征提取和管理
* **ClusteringManager**: 聚类算法管理
* **ResultManager**: 结果存储和导出
* **Strategies**: 不同的聚类策略（一步法、二步法、直接拼接法）

Machine Learning
^^^^^^^^^^^^^^

机器学习模块，提供：

* **ModelComparison**: 多模型比较和评估
* **MultifileEvaluator**: 多文件评估工具
* **Metrics**: 指标计算和验证

Preprocessing
^^^^^^^^^^^^

影像预处理模块，包括：

* **BatchProcessor**: 批量图像处理
* **ImageProcessor**: 单图像处理工具

Common
^^^^^^

通用工具模块，包括：

* **ServiceConfigurator**: 服务配置和依赖注入
* **ConfigAccessor**: 配置访问器
* **DataFrameUtils**: DataFrame 工具函数

设计模式
--------

HABIT 使用了多种设计模式：

策略模式 (Strategy Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^

用于实现不同的聚类策略：

* TwoStepStrategy
* OneStepStrategy
* DirectPoolingStrategy

工厂模式 (Factory Pattern)
^^^^^^^^^^^^^^^^^^^^^^

用于创建不同的特征提取器和聚类算法。

依赖注入 (Dependency Injection)
^^^^^^^^^^^^^^^^^^^^^^^^^^

通过 ServiceConfigurator 管理依赖关系，提高可测试性。

模块依赖关系
------------

.. graphviz::

   digraph architecture {
       rankdir=TB;
       node [shape=box, style=rounded];
       
       HabitatAnalysis -> FeatureManager;
       HabitatAnalysis -> ClusteringManager;
       HabitatAnalysis -> ResultManager;
       
       FeatureManager -> BaseExtractor;
       ClusteringManager -> BaseClustering;
       
       TwoStepStrategy -> HabitatAnalysis;
       OneStepStrategy -> HabitatAnalysis;
       DirectPoolingStrategy -> HabitatAnalysis;
   }
