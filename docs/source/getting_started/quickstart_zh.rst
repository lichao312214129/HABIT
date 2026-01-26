快速入门
========

本指南将帮助您快速上手 HABIT，优先完成图像预处理流程。

前提条件
----------

确保您已经：

1. 安装了 HABIT（参考 :doc:`installation_zh`）
2. 准备了医学图像数据（DICOM 或 NIfTI 格式）

快速入门示例
-----------

我们将使用 demo_data 中的示例数据完成图像预处理，并说明如何准备数据。

数据准备（demo_data）
~~~~~~~~~~~~~~~~~~~~

**重要提示**: 使用前需要先解压 `demo_data` 目录中的 `demo_data.rar` 压缩包。

解压后会得到以下 demo 数据：

- **DICOM 数据**: ``demo_data/dicom/sub001``、``demo_data/dicom/sub002``
- **预处理配置**: ``demo_data/config_preprocessing.yaml``
- **文件列表**: ``demo_data/files_preprocessing.yaml``

如果使用自己的数据，请按"受试者/期相/序列"的结构整理 DICOM，
并参照 ``files_preprocessing.yaml`` 填写每个受试者对应的序列路径。

示例（节选）：

.. code-block:: yaml

   auto_select_first_file: false
   images:
     subj001:
       delay2: ./dicom/sub001/WATER_BHAxLAVA-Flex-2min_Series0012
       delay3: ./dicom/sub001/WATER_BHAxLAVA-Flex-3min_Series0014
       delay5: ./dicom/sub001/WATER_BHAxLAVA-Flex-5min_Series0016

步骤 1: 图像预处理
~~~~~~~~~~~~~~~~~~~~

首先，我们需要对原始 DICOM 图像进行预处理。

**使用 CLI:**

.. code-block:: bash

   habit preprocess --config ./demo_data/config_preprocessing.yaml

**使用 Python API:**

.. code-block:: python

   from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

   processor = BatchProcessor(config_path='./demo_data/config_preprocessing.yaml')
   processor.process_batch()

**输出:**

预处理后的图像将保存在 `./demo_data/preprocessed/processed_images/` 目录下。

下一步建议
~~~~~~~~~~

完成预处理后，可继续阅读用户指南，进入生境分割与特征提取流程：

- :doc:`../user_guide/habitat_analysis_zh`
- :doc:`../user_guide/habitat_feature_extraction_zh`

配置文件说明
-----------

HABIT 使用 YAML 配置文件来控制所有参数。配置文件的结构如下：

**预处理配置 (config_preprocessing.yaml):**

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]

   processes: 2
   random_state: 42

下一步
-------

恭喜您完成了快速入门！接下来您可以：

- 阅读 :doc:`../user_guide/index_zh` 了解详细的使用指南
- 查看 :doc:`../tutorials/index_zh` 学习更多教程
- 探索 :doc:`../customization/index_zh` 了解如何自定义扩展功能
