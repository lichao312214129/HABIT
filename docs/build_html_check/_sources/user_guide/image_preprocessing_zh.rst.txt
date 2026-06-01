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

预处理结果在配置 ``out_dir`` 下的 ``processed_images/``；日志为 ``<out_dir>/processing.log``。Demo 写入 ``demo_data/results/preprocessed/processed_images/``。

配置
----

Demo 使用 ``config/preprocessing/config_preprocessing_demo.yaml``：重采样 → SimpleITK 配准 → Z-score；**输入** ``demo_data/preprocessed/processed_images``，**输出** ``demo_data/results/preprocessed/``。字段说明见 :doc:`../configuration_zh`。
