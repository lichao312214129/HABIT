完整 Demo 教程
==============

本教程带您走完典型的影像组学研究全流程：从原始影像到预测模型与模型对比。

前提条件
--------

1. 已完成 :doc:`installation_zh` 中的环境配置与 HABIT 安装
2. 每次运行前激活环境，并 ``cd`` 进入项目目录。ZIP 解压后目录名为 ``HABIT-main``（Git 克隆为 ``HABIT``）。若解压软件造成 ``HABIT-main/HABIT-main`` 嵌套，须 ``cd`` 到**最内层**且含 ``config/``、``habit/`` 的那一级（详见 :doc:`installation_zh` 中 ZIP 说明）。

   .. code-block:: bash

      conda activate habit
      cd "D:\HABIT-main"
      # 使用资源管理器地址栏复制的完整路径；macOS 可将文件夹拖入终端

下载演示数据
------------

**演示数据下载**

- **链接**: |demo_data_link|
- **提取码**: |demo_data_code|
- 解压到项目根目录下的 ``demo_data/``

**重要说明**：

- 所有隐私信息已完全去除
- **严禁商业用途，仅供学术研究和 Demo 演示使用**

解压后目录示意：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── demo_data/          <-- 演示数据解压到此
   │   ├── dicom/
   │   ├── preprocessed/processed_images/   <-- Demo 影像与 mask（步骤 2–4 的输入；可跳过步骤 1）
   │   └── ...
   └── habit/

完整研究流程（5 步）
--------------------

demo 已含预处理结果，**首次可跳过步骤 1**，从步骤 2 开始。改参数见 :doc:`../configuration_zh`。

**步骤 1 — 预处理**（约 2–5 分钟：重采样 → SimpleITK 配准 → Z-score）→ ``demo_data/results/preprocessed/processed_images/``（输入可仍用包内 ``demo_data/preprocessed/processed_images/``）

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

**步骤 2 — 生境分割**（约 5–10 分钟）→ ``demo_data/results/habitat_two_step/``

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

将 ``demo_data/results/habitat_two_step/subj001_habitats.nrrd`` 拖入 ITK-SNAP，叠加在 ``demo_data/preprocessed/processed_images/images/subj001/delay2/`` 下的参考序列上查看。步骤 2–4 的输入影像均来自 ``demo_data/preprocessed/processed_images/``。

**步骤 3 — 特征提取**（约 3–8 分钟）→ ``demo_data/results/features/``

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**步骤 4 — 机器学习**（约 2–5 分钟）→ ``demo_data/results/ml/radiomics/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

**步骤 5 — 模型对比**（约 1–3 分钟，需先训练 radiomics 与 clinical 两个模型）→ ``demo_data/results/model_comparison/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

主要输出目录
------------

- ``demo_data/preprocessed/processed_images/`` — 预处理影像与 mask
- ``demo_data/results/habitat_two_step/`` — 生境地图（ITK-SNAP 叠加查看）
- ``demo_data/results/features/`` — 特征 CSV（Excel / SPSS）
- ``demo_data/results/ml/`` — 模型训练输出（含 ROC 等 PDF）
- ``demo_data/results/model_comparison/`` — 多模型对比图与指标

下一步
------

- :doc:`../user_guide/index_zh` — 各步命令速查
- :doc:`../configuration_zh` — 修改 YAML 参数
