基本概念
========

本节介绍 HABIT 的核心概念和术语，帮助您更好地理解和使用这个工具包。

生境（Habitat）
--------------

生境是指肿瘤内部具有相似生物学特征的区域。通过将肿瘤分割为多个生境，可以更好地理解肿瘤的异质性，这对于诊断、预后评估和治疗反应预测具有重要意义。

**生境分析的核心思想：**

1. **肿瘤异质性**: 肿瘤不是均匀的组织，而是由多个具有不同特征的亚区域组成
2. **功能分区**: 不同的生境可能对应不同的生物学过程（如血管生成、坏死、炎症等）
3. **临床意义**: 生境特征可以提供比整体肿瘤特征更丰富的诊断信息

**生境分析流程：**

.. image:: ../user_guide/images/habitat_concept_workflow.png
   :alt: 生境分析工作流程
   :align: center

聚类策略
--------

HABIT 提供三种聚类策略用于生境分割：

**1. One-Step 策略**

- **描述**: 对每个肿瘤独立进行体素到生境的聚类
- **适用场景**: 样本量较小，或者希望保留每个肿瘤的独特特征
- **优点**: 简单直接，每个肿瘤独立分析
- **缺点**: 可能难以发现跨患者的共同模式

**2. Two-Step 策略（推荐）**

- **描述**: 两步聚类
  - 第一步：体素到超像素（个体级别）
  - 第二步：超像素到生境（群体级别）
- **适用场景**: 样本量较大，希望发现跨患者的共同生境模式
- **优点**: 可以发现跨患者的共同生境模式，提高泛化能力
- **缺点**: 计算复杂度较高

**3. Direct Pooling 策略**

- **描述**: 将所有体素池化后一次性聚类
- **适用场景**: 数据量较小，或者希望快速获得结果
- **优点**: 计算速度快，实现简单
- **缺点**: 可能忽略个体差异，数据泄露风险较高

**策略选择建议：**

- **研究型项目**: 推荐使用 Two-Step 策略
- **快速原型**: 可以使用 Direct Pooling 策略
- **小样本研究**: 可以考虑 One-Step 策略

Pipeline 机制
-------------

HABIT 继承了 scikit-learn 的 Pipeline 机制，这是避免数据泄露的关键设计。

**什么是数据泄露？**

数据泄露是指在模型训练过程中，测试集的信息意外地泄露到训练集中，导致模型性能被高估。

**Pipeline 如何避免数据泄露？**

1. **训练阶段**: 在训练集上训练 Pipeline，包括特征选择、聚类、模型训练等步骤
2. **预测阶段**: 加载训练好的 Pipeline，应用于测试集，确保使用相同的处理流程
3. **严格分离**: 训练集和测试集完全分离，避免测试集信息泄露

**生境分析中的 Pipeline:**

.. code-block:: python

   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

   # 加载配置
   config = HabitatAnalysisConfig.from_file('./config_habitat.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config)

   # 创建生境分析对象
   habitat_analysis = configurator.create_habitat_analysis()

   # 运行生境分析
   habitat_analysis.run()

**机器学习中的 Pipeline:**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from habit.core.machine_learning import ModelFactory

   pipeline = Pipeline([
       ('feature_selection', feature_selector),
       ('model', ModelFactory.create_model('RandomForest', config))
   ])

   # 训练阶段
   pipeline.fit(X_train, y_train)

   # 预测阶段
   y_pred = pipeline.predict(X_test)

**关键要点：**

- 训练和测试必须使用相同的 Pipeline
- 不要在整个数据集上进行聚类或特征选择
- 使用交叉验证时，每个 fold 的训练和测试必须严格分离

特征提取
--------

HABIT 支持多种特征提取方法：

**1. 体素级特征**

- **描述**: 对每个体素提取特征
- **适用场景**: 需要精细的空间分析
- **示例**: 原始图像强度、纹理特征、局部熵等

**2. 超像素级特征**

- **描述**: 对超像素区域提取特征
- **适用场景**: 需要区域级别的分析
- **示例**: 平均强度、方差、形态学特征等

**3. 生境级特征**

- **描述**: 对每个生境提取特征
- **适用场景**: 研究生境的生物学特性
- **示例**: 生境大小、形状、强度分布等

**4. 传统影像组学特征**

- **描述**: 使用 pyradiomics 提取标准影像组学特征
- **适用场景**: 与传统影像组学研究对比
- **示例**: GLCM、GLRLM、GLSZM 等纹理特征

**5. 非影像组学特征**

- **描述**: 提取非标准的影像组学特征
- **适用场景**: 探索新的特征类型
- **示例**: MSI（多尺度图像）、ITH（肿瘤内异质性）等

**6. 整体生境特征**

- **描述**: 提取整个生境的特征
- **适用场景**: 研究生境的整体特性
- **示例**: 生境数量、生境分布、生境间关系等

配置驱动
--------

HABIT 使用 YAML 配置文件来控制所有参数，这是系统的核心设计理念。

**配置文件的优势：**

1. **无需修改代码**: 通过修改配置文件即可调整功能
2. **版本控制**: 配置文件可以纳入版本控制，便于追踪变更
3. **可重复性**: 相同的配置文件产生相同的结果
4. **易于分享**: 配置文件可以轻松分享给其他研究者

**配置文件结构：**

.. code-block:: yaml

   # 数据路径
   data_dir: ./data.yaml
   out_dir: ./results

   # 核心配置
   FeatureConstruction:
     voxel_level:
       method: concat(raw(T1), raw(T2), raw(FLAIR))

   HabitatsSegmention:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50

   # 通用设置
   processes: 2
   random_state: 42

**配置文件类型：**

HABIT 有两种类型的配置文件：

1. **数据配置文件**: 指定图像和掩码的路径
2. **功能配置文件**: 指定处理流程的参数

参考 :doc:`../data_structure_zh` 了解更多关于数据配置文件的信息。

双重接口
--------

HABIT 提供两种使用方式：CLI 和 Python API。

**CLI 接口：**

- **适用场景**: 批处理、自动化任务、快速原型
- **优点**: 简单易用，无需编写代码
- **示例**:

.. code-block:: bash

   habit preprocess --config config_preprocessing.yaml
   habitat get-habitat --config config_habitat.yaml --mode predict
   habit extract --config config_extract_features.yaml
   habitat model --config config_machine_learning.yaml --mode train

**Python API 接口：**

- **适用场景**: 集成到其他项目、定制化开发、复杂工作流
- **优点**: 灵活强大，可以深度定制
- **示例**:

.. code-block:: python

   from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
   from habit.core.machine_learning import MLWorkflow

   # 预处理
   processor = BatchProcessor(config_path='config_preprocessing.yaml')
   processor.process_batch()

   # 生境分析
   config = HabitatAnalysisConfig.from_file('config_habitat.yaml')
   configurator = ServiceConfigurator(config=config)
   habitat_analysis = configurator.create_habitat_analysis()
   habitat_analysis.run()

   # 机器学习
   workflow = MLWorkflow(config)
   workflow.run_pipeline()

**选择建议：**

- **初学者**: 从 CLI 开始，快速上手
- **研究者**: 根据需求选择，CLI 用于快速实验，API 用于定制开发
- **开发者**: 使用 Python API，集成到自己的项目中
