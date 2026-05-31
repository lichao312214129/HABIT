图像预处理
==========

将 DICOM 转为 NIfTI，并完成重采样、配准、强度标准化等，使各病例影像可用于后续生境分析。

运行
----

.. code-block:: bash

   conda activate habit
   cd D:\HABIT-main
   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

仅整理 DICOM 目录（不转 NIfTI）：

.. code-block:: bash

   habit sort-dicom --config config/dicom_sort/config_sort_dicom.yaml

输出
----

预处理结果在配置 ``out_dir`` 下的 ``processed_images/``；日志为 ``processing.log``。

配置
----

YAML 步骤顺序、配准后端、路径等见 :doc:`../configuration_zh` 中 **Preprocessing** 相关章节。Demo 默认使用 SimpleITK（``config/preprocessing/config_preprocessing_demo.yaml``）；需 elastix 时可改用 ``config_preprocessing_demo_elastix.yaml``。
