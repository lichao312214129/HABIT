生境分割
========

本节介绍如何使用 HABIT 进行生境分割，这是 HABIT 的核心功能。

概述
----

生境分割是将肿瘤分割为多个具有相似特征的区域（生境）的过程。HABIT 提供了三种聚类策略，支持灵活的特征提取和自定义扩展。

**生境分析的核心思想：**

1. **肿瘤异质性**: 肿瘤不是均匀的组织，而是由多个具有不同特征的亚区域组成
2. **功能分区**: 不同的生境可能对应不同的生物学过程（如血管生成、坏死、炎症等）
3. **临床意义**: 生境特征可以提供比整体肿瘤特征更丰富的诊断信息

**三种聚类策略：**

1. **One-Step 策略**: 个体级别聚类，每个肿瘤独立进行体素到生境的聚类
2. **Two-Step 策略（推荐）**: 两步聚类，先体素到超像素，再超像素到生境
3. **Direct Pooling 策略**: 直接池化策略，将所有体素池化后一次性聚类

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit get-habitat --config <config_file> [--mode <mode>] [--pipeline <pipeline_path>] [--debug]

**参数说明：**

- `--config`, `-c`: 配置文件路径（必需）
- `--mode`, `-m`: 运行模式（train 或 predict），覆盖配置文件中的设置
- `--pipeline`: Pipeline 文件路径，用于 predict 模式，覆盖配置文件中的设置
- `--debug`: 启用调试模式

**使用示例：**

.. code-block:: bash

   # 训练模式
   habit get-habitat --config ./config_habitat_train.yaml --mode train

   # 预测模式
   habit get-habitat --config ./config_habitat.yaml --mode predict

   # 使用指定的 Pipeline 文件
   habit get-habitat --config ./config_habitat.yaml --mode predict --pipeline ./custom_pipeline.pkl

   # 启用调试模式
   habit get-habitat --config ./config_habitat.yaml --debug

**输出：**

生境图将保存在配置文件中指定的输出目录。

Python API 使用方法
------------------

**基本用法：**

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

**详细示例：**

.. code-block:: python

   import logging
   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
   from habit.utils.log_utils import setup_logger
   from pathlib import Path

   # 设置日志
   output_dir = Path('./results/habitat')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='habitat_analysis',
       output_dir=output_dir,
       log_filename='habitat_analysis.log',
       level=logging.INFO
   )

   # 加载配置
   config = HabitatAnalysisConfig.from_file('./config_habitat.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))

   # 创建生境分析对象
   habitat_analysis = configurator.create_habitat_analysis()

   # 运行生境分析
   logger.info("开始生境分析")
   habitat_analysis.run(save_results_csv=True)
   logger.info("生境分析完成！")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 运行模式和 Pipeline 设置（首先检查这些！）
   run_mode: predict  # train（如果要训练新模型，设置为 train）或 predict（如果要使用预训练模型，设置为 predict）
   pipeline_path: ./results/habitat_pipeline.pkl  # predict 模式的 Pipeline 路径（predict 模式必需）

   # 数据路径
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/predict

   # 特征提取设置（仅在 train 模式需要）
   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

     preprocessing_for_subject_level:
       methods:
         - method: winsorize
           winsor_limits: [0.05, 0.05]
           global_normalize: true
         - method: minmax
           global_normalize: true

     preprocessing_for_group_level:
       methods:
         - method: binning
           n_bins: 10
           bin_strategy: uniform
           global_normalize: false

   # 生境分割设置（仅在 train 模式需要）
   HabitatsSegmention:
     # 聚类策略："one_step"、"two_step" 或 "direct_pooling"
     clustering_mode: two_step

     # 超像素聚类设置（第一步：个体级别聚类）
     supervoxel:
       algorithm: kmeans
       n_clusters: 50  # two_step 模式使用，或 one_step 模式的最大值
       random_state: 42
       max_iter: 300
       n_init: 10

       # one_step 模式设置：每个肿瘤的自动聚类数选择
       one_step_settings:
         min_clusters: 2           # 要测试的最小聚类数
         max_clusters: 10          # 要测试的最大聚类数
         selection_method: inertia  # 确定最佳聚类数的方法：silhouette、calinski_harabasz、davies_bouldin、inertia
         plot_validation_curves: true  # 为每个肿瘤绘制验证曲线

     # 生境聚类设置（第二步：群体级别聚类，仅在 two_step 模式使用）
     habitat:
       algorithm: kmeans  # kmeans 或 gmm
       max_clusters: 10
       # - 'silhouette'、'calinski_harabasz': 选择最大分数（越高越好）
       # - 'inertia'、'aic'、'bic': 使用二阶导数选择拐点（越低越好）
       habitat_cluster_selection_method:
         - inertia
         # - silhouette
         # - calinski_harabasz
         # - aic
         # - bic
         # - davies_bouldin

       best_n_clusters:    # 设置为 null 进行自动选择
       random_state: 42
       max_iter: 300
       n_init: 10

   # 通用设置
   processes: 2  # 并行进程数
   plot_curves: true  # 是否生成和保存图表
   save_results_csv: true  # 是否将结果保存为 CSV 文件
   random_state: 42  # 可重复性的随机种子
   debug: false  # 启用详细日志的调试模式

**字段说明：**

**run_mode**: 运行模式

- `train`: 训练新模型
- `predict`: 使用预训练模型进行预测

**pipeline_path**: Pipeline 文件路径

- predict 模式必需
- 指定训练好的 Pipeline 文件路径

**data_dir**: 数据目录路径

- 可以是文件夹或 YAML 配置文件
- 参考 :doc:`../data_structure_zh` 了解数据结构

**out_dir**: 输出目录路径

- 生境图和结果将保存在此目录

**FeatureConstruction**: 特征提取设置

- `voxel_level`: 体素级特征提取
- `supervoxel_level`: 超像素级特征提取
- `preprocessing_for_subject_level`: 个体级别预处理
- `preprocessing_for_group_level`: 群体级别预处理

**HabitatsSegmention**: 生境分割设置

- `clustering_mode`: 聚类策略（one_step、two_step、direct_pooling）
- `supervoxel`: 超像素聚类设置
- `habitat`: 生境聚类设置

聚类策略详解
----------------

**One-Step 策略**

**描述：**

对每个肿瘤独立进行体素到生境的聚类。

**适用场景：**
- 样本量较小
- 希望保留每个肿瘤的独特特征
- 不需要跨患者的泛化

**优点：**
- 简单直接
- 每个肿瘤独立分析
- 计算复杂度较低

**缺点：**
- 可能难以发现跨患者的共同模式
- 泛化能力较弱

**配置示例：**

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: one_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 10  # 最大聚类数
       random_state: 42
       one_step_settings:
         min_clusters: 2
         max_clusters: 10
         selection_method: silhouette
         plot_validation_curves: true

**Two-Step 策略（推荐）**

**描述：**

两步聚类：
- 第一步：体素到超像素（个体级别）
- 第二步：超像素到生境（群体级别）

**适用场景：**
- 样本量较大
- 希望发现跨患者的共同生境模式
- 需要更好的泛化能力

**优点：**
- 可以发现跨患者的共同生境模式
- 提高泛化能力
- 减少计算复杂度

**缺点：**
- 计算复杂度较高
- 需要更多的参数调优

**配置示例：**

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       best_n_clusters: null  # 自动选择
       random_state: 42

**Direct Pooling 策略**

**描述：**

将所有体素池化后一次性聚类。

**适用场景：**
- 数据量较小
- 希望快速获得结果
- 不需要个体差异

**优点：**
- 计算速度快
- 实现简单

**缺点：**
- 可能忽略个体差异
- 数据泄露风险较高

**配置示例：**

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: direct_pooling
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
       best_n_clusters: null
       random_state: 42

Pipeline 机制
------------

HABIT 继承了 scikit-learn 的 Pipeline 机制，这是避免数据泄露的关键设计。

**什么是数据泄露？**

数据泄露是指在模型训练过程中，测试集的信息意外地泄露到训练集中，导致模型性能被高估。

**Pipeline 如何避免数据泄露？**

1. **训练阶段**: 在训练集上训练 Pipeline，包括特征提取、聚类等步骤
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

**关键要点：**

- 训练和预测必须使用相同的 Pipeline
- 不要在整个数据集上进行聚类
- 使用交叉验证时，每个 fold 的训练和预测必须严格分离

特征提取详解
----------------

**体素级特征提取**

**描述：**

对每个体素提取特征。

**适用场景：**
- 需要精细的空间分析
- 研究体素级别的异质性

**配置示例：**

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

**内置特征提取器：**

- `raw`: 原始图像强度
- `kinetic`: 动力学特征
- `local_entropy`: 局部熵
- `mean_voxel_features`: 平均体素特征
- `supervoxel_radiomics`: 超像素影像组学特征

**超像素级特征提取**

**描述：**

对超像素区域提取特征。

**适用场景：**
- 需要区域级别的分析
- 减少计算复杂度

**配置示例：**

.. code-block:: yaml

   FeatureConstruction:
     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

**预处理方法**

**个体级别预处理：**

.. code-block:: yaml

   preprocessing_for_subject_level:
     methods:
       - method: winsorize
         winsor_limits: [0.05, 0.05]
         global_normalize: true
       - method: minmax
         global_normalize: true
       - method: log
         global_normalize: true

**群体级别预处理：**

.. code-block:: yaml

   preprocessing_for_group_level:
     methods:
       - method: binning
         n_bins: 10
         bin_strategy: uniform
         global_normalize: false

自定义特征提取器
----------------

HABIT 支持自定义特征提取器，您可以添加自己的特征提取方法。

**步骤 1: 创建自定义特征提取器**

.. code-block:: python

   from habit.core.habit_analysis.extractors.base_extractor import BaseClusteringExtractor
   from habit.core.habit_analysis.extractors.base_extractor import register_feature_extractor

   @register_feature_extractor('my_feature_extractor')
   class MyFeatureExtractor(BaseClusteringExtractor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.feature_names = ['feature1', 'feature2', 'feature3']

       def extract_features(self, image_data, **kwargs):
           # 实现特征提取逻辑
           n_samples = image_data.shape[0]
           features = np.random.random((n_samples, 3))
           return features

**步骤 2: 在配置文件中使用**

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       method: my_feature_extractor(delay2, delay3)
       params:
         param1: value1

**步骤 3: 运行生境分析**

.. code-block:: bash

   habit get-habitat --config config_with_custom_extractor.yaml

实际示例
--------

**示例 1: 训练模式**

.. code-block:: yaml

   run_mode: train
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/train

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

     preprocessing_for_group_level:
       methods:
         - method: binning
           n_bins: 10
           bin_strategy: uniform

   HabitatsSegmention:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       random_state: 42

   processes: 2
   plot_curves: true
   save_results_csv: true
   random_state: 42

**示例 2: 预测模式**

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./results/habitat/train/habit_pipeline.pkl
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/predict

   processes: 2
   random_state: 42

输出结构
--------

生境分析的输出结构：

.. code-block:: text

   results/habitat/
   ├── train/                    # 训练模式输出
   │   ├── habitat_pipeline.pkl  # 训练好的 Pipeline
   │   ├── habitats/              # 生境图
   │   │   ├── subj001/
   │   │   │   ├── delay2_habitats.nrrd
   │   │   │   ├── delay3_habitats.nrrd
   │   │   │   └── delay5_habitats.nrrd
   │   │   └── subj002/
   │   │       └── ...
   │   ├── supervoxels/           # 超像素图
   │   │   └── ...
   │   ├── features/               # 特征文件
   │   │   └── ...
   │   ├── plots/                 # 可视化图表
   │   │   └── ...
   │   └── habitat_analysis.log  # 日志文件
   └── predict/                  # 预测模式输出
       ├── habitats/
       │   └── ...
       ├── plots/
       │   └── ...
       └── habitat_analysis.log

常见问题
--------

**Q1: 如何选择聚类策略？**

A: 根据您的研究需求选择：
- **研究型项目**: 推荐使用 Two-Step 策略
- **快速原型**: 可以使用 Direct Pooling 策略
- **小样本研究**: 可以考虑 One-Step 策略

**Q2: 如何确定最佳聚类数？**

A: HABIT 提供了多种方法：
- **inertia**: 使用拐点方法
- **silhouette**: 使用轮廓系数
- **calinski_harabasz**: 使用 Calinski-Harabasz 指数
- **davies_bouldin**: 使用 Davies-Bouldin 指数

**Q3: 训练和预测模式有什么区别？**

A: 
- **训练模式**: 在训练集上训练新的 Pipeline，包括特征提取、聚类等步骤
- **预测模式**: 加载训练好的 Pipeline，应用于测试集，使用相同的处理流程

**Q4: 如何避免数据泄露？**

A: 遵循以下原则：
1. 训练和预测使用相同的 Pipeline
2. 不要在整个数据集上进行聚类
3. 使用交叉验证时，每个 fold 的训练和预测必须严格分离

**Q5: 生境分割失败怎么办？**

A: 检查以下几点：
1. 配置文件是否正确
2. 数据路径是否正确
3. 图像和掩码是否匹配
4. 特征提取是否成功
5. 查看日志文件了解详细错误信息

下一步
-------

生境分割完成后，您可以：

- :doc:`habitat_feature_extraction_zh`: 进行生境特征提取
- :doc:`machine_learning_modeling_zh`: 进行机器学习建模
- :doc:`../customization/index_zh`: 了解如何自定义特征提取器和聚类算法
