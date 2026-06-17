完整 Demo 教程
==============

本教程带您走完典型的影像组学研究全流程：从原始影像到预测模型与模型对比。

一、前提条件
------------

1. 已完成 :doc:`installation_zh` 中的安装
2. 已按下文第二节下载并解压 ``config`` 与 ``demo_data`` 到工作文件夹

**Windows 便携包**：完成安装（解压 + ``setup_habit.bat``）后，在本教程第二节下载 ``config.rar`` 与 ``demo_data.rar``。

**源码安装**：``conda activate habit`` 后 ``cd`` 到项目根目录；``config\`` 随源码自带，只需在第二节下载 ``demo_data.rar``。

跑 Demo 前先进入工作文件夹：

.. code-block:: bash

   cd /d D:\habit-cpu
   habit --version

二、下载配置与演示数据
----------------------

便携包 **不含** 示例配置与演示数据，须从网盘下载并解压到工作文件夹（与 ``python.exe`` 同级，如 ``D:\habit-cpu\``）。源码用户 ``config\`` 已随仓库自带，**只需下载 demo_data**。

+------------------+---------------------+------------------------------------------+--------+
| 网盘文件         | 解压后应出现        | 链接                                     | 提取码 |
+==================+=====================+==========================================+========+
| ``config.rar``   | ``config\`` 文件夹  | |config_pack_link|                       | |config_pack_code| |
+------------------+---------------------+------------------------------------------+--------+
| ``demo_data.rar``| ``demo_data\`` 文件夹| |demo_data_link|                         | |demo_data_code| |
+------------------+---------------------+------------------------------------------+--------+

解压时选 **解压到当前文件夹**。完成后目录类似：

.. code-block:: text

   D:\habit-cpu\
   ├── python.exe
   ├── setup_habit.bat
   ├── config\
   └── demo_data\

演示数据已去标识，**仅供学术研究与 Demo，禁止商业用途**。

三、完整研究流程（5 步）
------------------------

demo 已含预处理结果，**首次可跳过步骤 1** ，从步骤 2 开始。改参数见 :doc:`../configuration_zh`。
下文 YAML 均在 **`config/`** 下；完整场景索引见 ``config/README_CONFIG.md`` 。

**请先** ``cd`` **到工作根目录**（便携包为 pack 根，如 ``D:\habit-cpu`` ；源码为 ``HABIT-main`` ）。

**步骤 1 — 预处理** （约 2–5 分钟：重采样 → SimpleITK 配准 → Z-score）→ ``demo_data/results/preprocessed/processed_images/`` （输入可仍用 ``demo_data/preprocessed/processed_images/``）

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

**步骤 2 — 生境分割** （约 5–10 分钟）→ ``demo_data/results/habitat_two_step/``

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

将 ``demo_data/results/habitat_two_step/subj001_habitats.nrrd`` 拖入 ITK-SNAP，叠加在 ``demo_data/preprocessed/processed_images/images/subj001/delay2/`` 下的参考序列上查看。步骤 2–4 的输入影像均来自 ``demo_data/preprocessed/processed_images/``。

**步骤 3 — 特征提取** （约 3–8 分钟）→ ``demo_data/results/features/``

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**步骤 4 — 机器学习** （约 2–5 分钟）→ ``demo_data/results/ml/radiomics/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

**步骤 5 — 模型对比** （约 1–3 分钟，需先训练 radiomics 与 clinical 两个模型）→ ``demo_data/results/model_comparison/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

四、主要输出目录
----------------

- ``demo_data/preprocessed/processed_images/`` — 预处理影像与 mask
- ``demo_data/results/habitat_two_step/`` — 生境地图（ITK-SNAP 叠加查看）
- ``demo_data/results/features/`` — 特征 CSV（Excel / SPSS）
- ``demo_data/results/ml/`` — 模型训练输出（含 ROC 等 PDF）
- ``demo_data/results/model_comparison/`` — 多模型对比图与指标

下一步
------

- :doc:`../user_guide/index_zh` — 各步命令速查
- :doc:`../configuration_zh` — 修改 YAML 参数
