基础 Habitat 分析教程
=======================

本教程将带你完成一个完整的 habitat 分析流程。

准备数据
----------

确保你的数据目录结构如下：

.. code-block:: bash

   data/
   ├── images/
   │   ├── patient001.nii.gz
   │   ├── patient002.nii.gz
   │   └── ...
   └── masks/
       ├── patient001.nii.gz
       ├── patient002.nii.gz
       └── ...

创建配置文件
------------

.. code-block:: yaml

   data_dir: ./data
   out_dir: ./output
   
   FeatureConstruction:
     voxel_level:
       method: mean_voxel_features
       params: {}
   
   HabitatsSegmention:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
     habitat:
       algorithm: kmeans
       min_clusters: 2
       max_clusters: 5
       habitat_cluster_selection_method: silhouette

运行分析
--------

.. code-block:: python

   from habit.core.habitat_analysis import HabitatAnalysis
   
   # 加载配置
   import yaml
   with open('config.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # 运行分析
   analysis = HabitatAnalysis(config)
   results = analysis.run()
   
   # 查看结果
   print(results.head())

结果解释
--------

分析完成后，你将得到：

* **supervoxel 图像**: 每个患者的超体素分割结果
* **habitat 图像**: 每个患者的生境分割结果
* **CSV 文件**: 包含每个超体素/生境的特征和标签

下一步
--------

* 学习 `自定义特征提取 <custom_features.html>`_
* 了解 `机器学习集成 <ml_integration.html>`_
