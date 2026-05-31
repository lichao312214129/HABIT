完整 Demo 教程
==============

本教程带您走完典型的影像组学研究全流程：从原始影像到预测模型与模型对比。

前提条件
--------

1. 已完成 :doc:`installation_zh` 中的环境配置与 HABIT 安装
2. 每次运行前激活环境：

   .. code-block:: bash

      conda activate habit
      cd D:\HABIT-main   # ZIP 解压后的目录；Git 克隆多为 HABIT

下载演示数据
------------

**演示数据下载**

- **链接**: |demo_data_link|
- **提取码**: |demo_data_code|
- 解压到项目根目录下的 ``demo_data/``（与 ``config/`` 同级）

**重要说明**：

- 所有隐私信息已完全去除
- **严禁商业用途，仅供学术研究和 Demo 演示使用**

解压后目录示意：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── demo_data/          <-- 演示数据解压到此（与 config 同级）
   │   ├── dicom/
   │   ├── preprocessed/   <-- 已含预处理结果，可跳过步骤 1
   │   └── ...
   └── habit/

完整研究流程（5 步）
--------------------

下面每一步均说明**临床意义**、**命令**、**预计用时**与**输出位置**。

步骤 1：影像预处理
~~~~~~~~~~~~~~~~~~

**临床意义**

- 确保所有病例的影像在同一"标尺"下分析，消除设备与扫描参数差异
- 类似于实验室检查前的标准化操作

**命令**

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo_elastix.yaml

**预计用时**：约 2–5 分钟

**结果位置**：``demo_data/preprocessed/``（最终产物在 ``processed_images/``）

**重要提示**

- demo 包内 ``preprocessed/`` 已含预处理影像与 ROI，**可直接进入步骤 2**
- 重新运行预处理会**覆盖**各阶段目录中的影像文件，但不会改变 ROI 勾画语义
- ROI mask 是后续生境分析的必需输入，请勿删除

步骤 2：生境聚类分析
~~~~~~~~~~~~~~~~~~~~

**临床意义**

- 自动识别肿瘤内部亚区（坏死区、活跃增殖区、缺氧区等）
- 不同亚区对治疗反应与预后可能不同

**命令**

.. code-block:: bash

   # 二步法（demo 默认：个体级超体素 + 群体级生境）
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

   # （可选）一步法 / direct-pooling
   # habit get-habitat --config config/habitat/config_habitat_one_step_raw_concat_train.yaml
   # habit get-habitat --config config/habitat/config_habitat_direct_pooling.yaml

**预计用时**：约 5–10 分钟

**结果位置**：``demo_data/results/habitat_two_step/``

**如何查看**：将 ``subj001_habitats.nrrd``、``subj001_supervoxel.nrrd`` 拖入 ITK-SNAP，叠加在原始影像上

步骤 3：特征提取
~~~~~~~~~~~~~~~~

**临床意义**

- 从影像中提取纹理、形状、强度等定量指标
- 量化肿瘤异质性，捕捉肉眼难以识别的信息

**命令**

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**预计用时**：约 3–8 分钟

**结果位置**：``demo_data/results/features/``（每类特征一个 CSV）

**前置条件**：配置中 ``habitats_map_folder`` 需指向步骤 2 实际输出目录（two-step 时为 ``./results/habitat_two_step``）

步骤 4：机器学习建模
~~~~~~~~~~~~~~~~~~~~

**临床意义**

- 从众多特征中筛选与预后/疗效相关的生物标志物
- 构建预测模型，辅助个体化治疗决策

**命令**

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

   # （可选）K 折交叉验证
   # habit model --config config/machine_learning/config_machine_learning_kfold_demo.yaml --mode train

**预计用时**：约 2–5 分钟

**结果位置**：``demo_data/ml_data/radiomics/``

**生成内容**：``prediction_results.csv``、``evaluation_metrics.csv``、ROC/校准/DCA/PR 曲线 PDF、混淆矩阵等

步骤 5：模型比较
~~~~~~~~~~~~~~~~

**临床意义**

- 比较不同模型（如 radiomics vs clinical）的预测性能
- 用于方法学对比与论文撰写

**命令**

运行前需分别用 ``config_machine_learning_radiomics.yaml`` 与 ``config_machine_learning_clinical.yaml`` 训练两个模型：

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

**预计用时**：约 1–3 分钟

**结果位置**：``demo_data/ml_data/model_comparison/``

一键运行全流程
--------------

熟悉各步输出后，可依次执行（约 15–30 分钟）：

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo_elastix.yaml && \
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml && \
   habit extract --config config/feature_extraction/config_extract_features_demo.yaml && \
   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train && \
   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train && \
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

建议首次使用时**逐步运行**，便于理解每步产物。

结果文件夹结构
--------------

运行完成后，``demo_data/`` 下形成三块输出：

.. code-block:: text

   demo_data/
   │
   ├── preprocessed/                    <-- 预处理产物
   │   ├── processing.log
   │   ├── dcm2nii_01/ ... zscore_normalization_04/
   │   └── processed_images/            <-- 供下游使用的最终产物
   │       ├── images/<subject>/<modality>/<modality>.nii.gz
   │       └── masks/<subject>/<modality>/<modality>.nii.gz
   │
   ├── results/
   │   ├── habitat_two_step/            <-- 生境地图、聚类图、habitats.csv
   │   └── features/                    <-- 各类特征 CSV
   │
   └── ml_data/
       ├── radiomics/ / clinical/       <-- 单模型训练结果
       └── model_comparison/            <-- 多模型对比图与 DeLong 检验

如何查看和使用结果
------------------

1. **生境地图（ITK-SNAP / 3D Slicer）**

   - 影像：``demo_data/preprocessed/processed_images/images/subj001/delay2/delay2.nii.gz``
   - 叠加：``demo_data/results/habitat_two_step/subj001_habitats.nrrd``

2. **特征数据（Excel / SPSS）**

   - ``demo_data/results/features/raw_image_radiomics.csv``
   - ``whole_habitat_radiomics.csv``、``msi_features.csv``、``ith_scores.csv`` 等

3. **模型性能**

   - 曲线：``demo_data/ml_data/radiomics/roc_curve.pdf`` 等
   - 指标：``evaluation_metrics.csv``（AUC > 0.8 通常视为较好）

4. **论文撰写**

   - 图表为 PDF 矢量格式，可直接插入稿件
   - ``delong_results.json`` 提供模型间 AUC 差异的统计学检验

结果解读小贴士
--------------

**对于医生**

- 生境地图直观展示肿瘤内部异质性，不同颜色区域可能对应不同生物学行为

**对于研究人员**

- 特征 CSV 与模型指标可用于进一步统计与方法学对比
- 各步骤均有日志，便于复现

下一步
------

- :doc:`../user_guide/habitat_segmentation_zh` — 生境分割详解
- :doc:`../user_guide/habitat_feature_extraction_zh` — 特征提取详解
- :doc:`../user_guide/machine_learning_modeling_zh` — 机器学习建模
- :doc:`../user_guide/index_zh` — 完整用户指南
