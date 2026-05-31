生境特征提取
============

从生境地图（及可选的原始影像）提取定量特征表（CSV），供机器学习或统计分析。

运行
----

.. code-block:: bash

   conda activate habit
   cd D:\HABIT-main
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

前置
----

配置中 ``habitats_map_folder`` 须指向生境分割的实际输出目录（two-step demo 一般为 ``./results/habitat_two_step``）。

输出
----

特征 CSV 保存在配置指定的输出目录（demo 为 ``demo_data/results/features/``）。

配置
----

特征类型、PyRadiomics 参数文件等见 :doc:`../configuration_zh` 中特征提取相关章节。Demo 模板：`config/feature_extraction/config_extract_features_demo.yaml`。
