生境分割
========

将肿瘤 ROI 内体素聚类为多个生境（亚区），输出生境地图供查看与后续特征提取。Demo 默认推荐 **Two-Step** 策略。

运行
----

.. code-block:: bash

   conda activate habit
   cd D:\HABIT-main
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

其它策略（改配置文件即可）：

- 一步法：``config/habitat/config_habitat_one_step_raw_concat_train.yaml``
- Direct pooling：``config/habitat/config_habitat_direct_pooling.yaml``

常用选项：``--mode train`` / ``predict``；中断后续跑可加 ``--resume``（详见 :doc:`../configuration_zh`）。

输出与查看
----------

- 生境地图：``*_habitats.nrrd``（及 two-step 时的 ``*_supervoxel.nrrd``）
- 在 **ITK-SNAP** 或 **3D Slicer** 中打开原始 MRI，将上述文件作为 Segmentation/Overlay 叠加；不同颜色代表不同生境，需结合各序列信号自行解读（如强化区、坏死区等）
- Demo 结果目录：``demo_data/results/habitat_two_step/``

配置
----

聚类策略、特征、并行、断点续训等全部参数见 :doc:`../configuration_zh` 中 **HabitatSegmentation** 相关章节。

完整 5 步流程见 :doc:`../getting_started/quickstart_zh`。
