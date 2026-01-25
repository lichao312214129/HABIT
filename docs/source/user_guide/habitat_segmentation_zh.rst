生境分割
========

本节介绍如何使用 HABIT 进行生境分割，这是 HABIT 的核心功能。

概述
----

生境分割是将肿瘤分割为多个具有相似特征的区域（生境）的过程。HABIT 提供了三种聚类策略，支持灵活的特征提取和自定义扩展。

**生境分析的核心思想：**

1.  **肿瘤异质性**: 肿瘤不是均匀的组织，而是由多个具有不同特征的亚区域组成
2.  **功能分区**: 不同的生境可能对应不同的生物学过程（如血管生成、坏死、炎症等）
3.  **临床意义**: 生境特征可以提供比整体肿瘤特征更丰富的诊断信息

**三种聚类策略：**

1.  **One-Step 策略**: 个体级别聚类，每个肿瘤独立进行体素到生境的聚类
2.  **Two-Step 策略（推荐）**: 两步聚类，先体素到超像素，再超像素到生境
3.  **Direct Pooling 策略**: 直接池化策略，将所有体素池化后一次性聚类

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit get-habitat --config <config_file> [--mode <mode>] [--pipeline <pipeline_path>] [--debug]

**参数说明：**

*   `--config`, `-c`: 配置文件路径（必需）
*   `--mode`, `-m`: 运行模式（train 或 predict），覆盖配置文件中的设置
*   `--pipeline`: Pipeline 文件路径，用于 predict 模式，覆盖配置文件中的设置
*   `--debug`: 启用调试模式

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

结果解读指南
------------

运行 `get-habitat` 后，您会得到一系列结果文件。以下是如何解读这些结果的临床指南。

1. 生境地图 (*_habitats.nrrd)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这是最直观的结果。您可以使用 **ITK-SNAP** 或 **3D Slicer** 打开原始 MRI 图像，然后将此文件拖入作为 Segmentation（或 Overlay）。

*   **五颜六色的区域是什么？**
    每一个颜色代表一个“生境”（Habitat）。例如，红色可能代表 Cluster 1，蓝色代表 Cluster 2。
*   **如何赋予临床意义？**
    您需要结合原始影像的信号特征来解读。
    *   如果 Cluster 1 在 T1 增强序列上信号很高，在 ADC 图上信号很低，它可能代表 **活性肿瘤区**。
    *   如果 Cluster 2 在 T2 序列上信号高，无强化，可能代表 **坏死区** 或 **水肿区**。

2. 聚类验证图 (plots/validation_curves.png)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这张图帮助您回答：“为什么分成了 3 个生境，而不是 5 个？”

*   **横轴**：尝试的聚类数量（k=2, 3, 4...）。
*   **纵轴**：聚类评价指标（如 Inertia 或 Silhouette Score）。
*   **如何看？**
    *   **拐点法 (Elbow Method)**：看曲线在哪里出现明显的“肘部”弯曲。那个点通常就是最佳聚类数。
    *   **轮廓系数 (Silhouette)**：看哪个 k 对应的分数值最高。

3. 特征热图 (plots/feature_heatmap.png)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这张图展示了不同生境在各个影像序列上的特征表达。

*   **横轴**：影像序列（如 T1, T2, ADC）。
*   **纵轴**：生境类别（Habitat 1, Habitat 2...）。
*   **颜色**：代表特征值的高低（红色=高，蓝色=低）。
*   **解读示例**：如果 Habitat 1 在 "T1_post_contrast" 列显示为深红色，说明该生境是**高强化**区域。

Python API 使用方法
------------------

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

YAML 配置详解
--------------

(此处省略详细配置，请参考配置参考文档)
