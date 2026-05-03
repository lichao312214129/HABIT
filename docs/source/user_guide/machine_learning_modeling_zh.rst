机器学习建模
============

本节介绍如何使用 HABIT 进行机器学习建模，包括特征选择、模型训练和模型评估。

使用场景
--------

在 HABIT 中，机器学习通常有两种主要使用场景：

**1. 训练场景 (Train Mode)**
   *   **目的**：基于已有的带标签数据（如良恶性、生存期等），通过特征筛选和算法拟合，构建预测模型。
   *   **操作**：使用训练配置（`MLConfig`）运行 `habit model --mode train`。

**2. 预测场景 (Predict Mode)**
   *   **目的**：将训练好的模型应用于全新的、未见过的数据，获取预测结果（如 Radscore）。
   *   **操作**：仍然使用 `MLConfig`，把 ``run_mode`` 设为 ``predict`` 并提供
       ``pipeline_path``（已训练的 ``*_final_pipeline.pkl``）；运行
       `habit model --mode predict`。CLI 上的 ``--mode`` 会覆盖 YAML 中
       的 ``run_mode``。

概述
----

机器学习建模是生境分析的最终步骤，使用提取的生境特征进行预测模型训练和评估。

.. note::

   在开始之前，请确保您的数据已整理为 CSV 或 Excel 格式，且您已清楚哪一列是 ID，哪一列是标签（Label）。HABIT 会根据文件后缀自动识别格式。

核心指标说明
------------

在 HABIT 的机器学习结果中，经常会遇到以下核心指标：

**1. Radscore (影像组学评分)**

- **定义**: Radscore 是机器学习模型（如 Logistic Regression, Random Forest 等）对每个样本计算出的预测概率或线性组合得分。
- **意义**: 它代表了影像特征与目标标签（如恶性程度）之间的相关性强度。Radscore 越高，通常代表样本属于阳性类别的概率越大。
- **输出**: 在预测结果文件中，概率列（由 `output_prob_col` 指定）可作为对应模型的 Radscore。

如何训练融合模型
----------------

在医学影像研究中，"融合模型"（Fusion Model）通常指结合了**影像特征（如 Radscore）**和**临床特征**的模型。

**训练融合模型通常有两种方式：**

**方式 1：早期融合 (Early Fusion) - 推荐**

将生境特征（Habitat Features）与临床特征（Clinical Features）在进入模型前直接进行表格合并。

1. **准备影像特征**: 运行生境特征提取，导出 CSV。
2. **准备临床特征**: 准备包含相同 `PatientID` 的临床特征 CSV/Excel。
3. **合并表格**: 使用 :doc:`../app_merge_csv_zh` 按 ID 合并，得到单一融合特征表。
4. **训练模型**: 在 `habit model` 配置中，将合并后的表写入 `input` 列表，按常规流程训练模型。

**方式 2：晚期融合 (Late Fusion)**

分别训练影像模型和临床模型，再对两者的预测概率进行融合（加权平均或训练 Meta-Model）。

1. **训练影像模型**: 得到影像模型在训练集/验证集的预测分数（Radscore）。
2. **训练临床模型**: 得到临床模型在训练集/验证集的预测分数。
3. **融合分数**: 将两个分数合并为新的特征表，再训练一个简单模型（如 Logistic Regression）作为 Meta-Model。

**严格的训练集控制（避免数据泄露）**

- 融合所用的特征或分数必须来自训练集内部的流程。
- 早期融合时，特征选择与模型训练应在训练集内完成；测试集仅用于最终评估。
- 晚期融合时，融合权重或 Meta-Model 只能用训练集拟合，测试集只能用已固定的权重或模型进行推断。

**推荐做法：**

大多数研究推荐使用**方式 1**，因为它能让模型自动学习影像特征与临床特征之间的交互关系。您可以利用 HABIT 的特征选择模块（如 LASSO）自动从融合后的特征空间中挑选出最有意义的特征子集。

**早期融合示例：多数据源就是多张表合并**

本质上和常规机器学习一样，只是把多个数据源（影像特征、临床信息、实验室指标等）合并成一张表再训练。

**示例数据文件：**

- `habitat_features.csv`: 影像/生境特征表
- `clinical_features.csv`: 临床特征表
- `lab_features.csv`: 实验室指标表

**表头示例（每张表都必须包含相同的 ID 列）：**

.. csv-table::
   :header: "PatientID", "Feature_A", "Feature_B", "..."
   :widths: 20, 20, 20, 40

   "sub-001", 1.23, 4.56, "..."
   "sub-002", 2.34, 5.67, "..."

**合并步骤：**

1. 使用 :doc:`../app_merge_csv_zh` 依次合并三张表，按 `PatientID` 对齐。
2. 得到融合后的 `fusion_features.csv`（包含所有特征 + Label）。

**融合后的表头示例：**

.. csv-table::
   :header: "PatientID", "Label", "Habitat_F1", "Habitat_F2", "Clinical_Age", "Lab_CRP", "..."
   :widths: 20, 10, 20, 20, 20, 20, 30

   "sub-001", 1, 0.12, 0.34, 56, 2.1, "..."
   "sub-002", 0, 0.45, 0.78, 63, 5.4, "..."

**训练配置示例（与常规机器学习完全一致）：**

.. code-block:: yaml

   input:
     - path: ./demo_data/ml_data/fusion_features.csv
       subject_id_col: PatientID
       label_col: Label
   output: ./results/ml/train

数据准备指南
------------

这是医生用户最关心的部分。您的输入数据应为一个标准的表格文件，HABIT 支持以下格式：

- **CSV 文件** (`.csv`)
- **Excel 文件** (`.xlsx`, `.xls`)

HABIT 会根据文件后缀自动识别格式。

**表格文件示例 (data.csv 或 data.xlsx):**

.. csv-table::
   :header: "PatientID", "Label", "Feature_1", "Feature_2", "...", "Feature_N"
   :widths: 15, 10, 15, 15, 5, 15

   "sub-001", 0, 12.5, 0.45, "...", 102.3
   "sub-002", 1, 14.2, 0.67, "...", 98.1
   "sub-003", 2, 11.8, 0.33, "...", 105.4

**关键要求：**

1.  **ID 列** (`PatientID`): 每一行代表一个病人，ID 必须唯一。
2.  **标签列** (`Label`): 您要预测的目标。
    *   **二分类**: 通常用 `0` (阴性/良性) 和 `1` (阳性/恶性) 表示。
    *   **多分类**: 支持 3 个及以上类别，如 `0` (良性), `1` (低度恶性), `2` (高度恶性)。HABIT 会自动检测并计算多分类指标（Macro-Average）。
    *   **回归**: 可以是连续数值（如生存时间）。
3.  **特征列**: 除了 ID 和 Label 外的其他列，HABIT 会自动将其识别为特征用于训练。
4.  **无中文**: 表头和内容尽量避免中文，以免出现编码错误。

**如何在配置文件中指定列名？**

您需要在数据配置文件（通常是 `files_ml.yaml` 或直接在主配置中）指定：

.. code-block:: yaml

   # 方式1：CSV文件
   - path: ./demo_data/ml_data/clinical_feature.csv
     subject_id_col: PatientID
     label_col: Label

   # 方式2：Excel文件
   - path: ./demo_data/ml_data/clinical_feature.xlsx
     subject_id_col: PatientID
     label_col: Label

HABIT 提供了完整的机器学习工作流程：

- **特征选择**: 选择最相关的特征
- **模型训练**: 训练各种机器学习模型
- **模型评估**: 评估模型性能
- **模型保存**: 保存训练好的模型
- **模型预测**: 使用训练好的模型进行预测

**支持的模型类型：**

- **传统机器学习模型**: 逻辑回归、随机森林、SVM、KNN 等
- **集成学习模型**: XGBoost、LightGBM、AutoGluon 等
- **深度学习模型**: 神经网络（可选）

**Pipeline 机制：**

HABIT 使用 scikit-learn 的 Pipeline 机制，确保特征选择、模型训练等步骤在交叉验证中正确应用，避免数据泄露。

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit model --config <config_file> [--mode <mode>]

**参数说明：**

- `--config`, `-c`: 配置文件路径（必需）
- `--mode`, `-m`: 运行模式（train 或 predict），覆盖配置文件中的设置

**使用示例：**

.. code-block:: bash

   # 训练模式
   habit model --config ./demo_data/config_machine_learning.yaml --mode train

   # 预测模式
   habit model --config ./demo_data/config_machine_learning.yaml --mode predict

**输出：**

训练好的模型和评估结果将保存在配置文件中指定的输出目录。

Python API 使用方法
------------------

**基本用法：**

.. code-block:: python

   from habit.core.common.configurators import MLConfigurator
   from habit.core.machine_learning.config_schemas import MLConfig

   # Load configuration
   config = MLConfig.from_file('./config_machine_learning.yaml')

   # Build the ML configurator
   configurator = MLConfigurator(config=config)

   # Build the ML workflow (covers train + predict via run_mode)
   workflow = configurator.create_ml_workflow()

   # Run
   workflow.run()

**详细示例：**

.. code-block:: python

   import logging
   from pathlib import Path
   from habit.core.common.configurators import MLConfigurator
   from habit.core.machine_learning.config_schemas import MLConfig
   from habit.utils.log_utils import setup_logger

   # Logging
   output_dir = Path('./results/ml')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='machine_learning',
       output_dir=output_dir,
       log_filename='machine_learning.log',
       level=logging.INFO
   )

   # Load configuration
   config = MLConfig.from_file('./config_machine_learning.yaml')

   # Build the ML configurator (logger + output_dir come from BaseConfigurator)
   configurator = MLConfigurator(config=config, logger=logger, output_dir=str(output_dir))

   # Build the ML workflow
   workflow = configurator.create_ml_workflow()

   # Run
   logger.info("开始训练模式")
   workflow.run()
   logger.info("模型训练完成！")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 训练模式配置（MLConfig）
   input:
     - path: ./demo_data/ml_data/clinical_feature.csv
       name: clinical_
       subject_id_col: PatientID
       label_col: Label
   output: ./results/ml/train
   random_state: 42

   split_method: stratified
   test_size: 0.3

   normalization:
     method: z_score
     params: {}

   feature_selection_methods:
     - method: variance
       params:
         threshold: 0.0
         before_z_score: true
     - method: correlation
       params:
         threshold: 0.95
         method: spearman
         before_z_score: false

   models:
     LogisticRegression:
       params:
         C: 1.0
         penalty: l2
         solver: lbfgs
         max_iter: 1000
         random_state: 42
     RandomForest:
       params:
         n_estimators: 100
         random_state: 42
     XGBoost:
       params:
         n_estimators: 100
         max_depth: 6
         learning_rate: 0.1
         random_state: 42
     SVM:
       params:
         C: 1.0
         kernel: rbf
         gamma: scale
         probability: true
         random_state: 42
     KNN:
       params:
         n_neighbors: 5
         weights: uniform
         algorithm: auto
     AutoGluonTabular:
       params:
         path: ./results/ml/autogluon_models
         label: Label
         time_limit: 3600
         presets: high_quality

   # 开关
   is_visualize: true
   is_save_model: true

   visualization:
     enabled: true
     plot_types: [roc, dca, calibration, pr, confusion, shap]
     dpi: 600
     format: pdf

**字段说明：**

**input**: 输入数据配置（训练模式）

- 列表结构，每个元素至少包含 `path`、`subject_id_col`、`label_col`
- 支持 CSV/Excel；可通过 `name` 给特征加前缀，避免多表融合时重名

**output**: 输出目录路径（训练模式）

- 训练结果、模型与图表会保存到该目录

**split_method / test_size**: 数据划分策略

- `split_method` 支持 `random`、`stratified`、`custom`
- 当 `split_method: custom` 时，需提供 `train_ids_file` 和 `test_ids_file`

**feature_selection_methods**: 特征选择设置

- 列表结构，可串联多个选择器
- 每个选择器格式为 `- method: <name> + params: {...}`
- 常见方法包括 `variance`、`correlation`、`anova`、`chi2`、`lasso`、`rfecv`、`vif` 等

**models**: 模型配置

- 支持配置多个模型，所有模型都会被训练和评估
- 每个模型以字典形式配置，包含 `params` 参数
- 支持的模型类型：LogisticRegression、RandomForest、XGBoost、SVM、KNN、AutoGluonTabular 等

**is_visualize / visualization**: 可视化设置

- `is_visualize` 控制是否执行可视化回调
- `visualization` 可进一步设置图类型、DPI、格式

**is_save_model**: 模型保存开关

- 控制是否保存训练得到的模型文件

特征选择方法详解
----------------

**Variance (variance)**

基于方差的特征选择，移除方差低于阈值的特征。

**适用场景：**
- 移除低方差特征
- 减少特征数量

**参数说明：**
- `threshold`: 方差阈值，低于此阈值的特征将被移除

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: variance
       params:
         threshold: 0.0

**Correlation (correlation)**

基于相关性的特征选择，移除高度相关的特征。

**适用场景：**
- 移除冗余特征
- 减少特征数量

**参数说明：**
- `threshold_correlation`: 相关性阈值，高于此阈值的特征对中一个将被移除

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: correlation
       params:
         threshold: 0.95
         method: spearman

**ANOVA (anova)**

基于 ANOVA F 值的特征选择，选择与目标变量最相关的特征。

**适用场景：**
- 分类任务
- 选择与目标变量最相关的特征

**参数说明：**
- `k_features`: 要选择的特征数量

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: anova
       params:
         p_threshold: 0.05

**Chi2 (chi2)**

基于卡方检验的特征选择，选择与目标变量最相关的特征。

**适用场景：**
- 分类任务
- 非负特征

**参数说明：**
- `k_features_chi2`: 要选择的特征数量

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: chi2
       params:
         p_threshold: 0.05

**LASSO (lasso)**

基于 LASSO 回归的特征选择，使用 L1 正则化进行特征选择。

**适用场景：**
- 线性模型
- 自动特征选择

**参数说明：**
- `alpha`: L1 正则化强度

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: lasso
       params:
         cv: 10

**RFECV (rfecv)**

递归特征消除与交叉验证，通过递归地移除特征并评估模型性能来选择最佳特征子集。

**适用场景：**
- 需要精确的特征选择
- 计算资源充足

**参数说明：**
- `estimator`: 估计器类型
- `cv`: 交叉验证折数
- `scoring`: 评分指标

**配置示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: rfecv
       params:
         estimator: LogisticRegression
         cv: 5
         scoring: accuracy

模型类型详解
----------------

**LogisticRegression**

逻辑回归，适用于二分类和多分类任务。

**适用场景：**
- 二分类任务
- 需要概率输出
- 解释性要求高

**优点：**
- 简单快速
- 可解释性强
- 提供概率输出

**缺点：**
- 假设线性关系
- 对异常值敏感

**参数说明：**
- `C`: 正则化强度的倒数
- `penalty`: 正则化类型（l1、l2、elasticnet）
- `solver`: 优化算法
- `max_iter`: 最大迭代次数

**配置示例：**

.. code-block:: yaml

   models:
     LogisticRegression:
       params:
         C: 1.0
         penalty: l2
         solver: lbfgs
         max_iter: 1000
         random_state: 42

**RandomForest**

随机森林，适用于分类和回归任务。

**适用场景：**
- 分类和回归任务
- 高维数据
- 非线性关系

**优点：**
- 性能强大
- 不易过拟合
- 提供特征重要性

**缺点：**
- 计算复杂度高
- 内存消耗大

**参数说明：**
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `min_samples_split`: 分裂节点所需的最小样本数
- `min_samples_leaf`: 叶节点所需的最小样本数

**配置示例：**

.. code-block:: yaml

   models:
     RandomForest:
       params:
         n_estimators: 100
         max_depth: null
         min_samples_split: 2
         min_samples_leaf: 1
         random_state: 42

**XGBoost**

梯度提升决策树，适用于分类和回归任务。

**适用场景：**
- 分类和回归任务
- 大规模数据
- 高性能要求

**优点：**
- 性能强大
- 计算效率高
- 支持并行计算

**缺点：**
- 参数较多
- 需要调优

**参数说明：**
- `n_estimators`: 树的数量
- `max_depth`: 树的最大深度
- `learning_rate`: 学习率

**配置示例：**

.. code-block:: yaml

   models:
     XGBoost:
       params:
         n_estimators: 100
         max_depth: 6
         learning_rate: 0.1
         random_state: 42

**SVM**

支持向量机，适用于分类和回归任务。

**适用场景：**
- 二分类任务
- 高维数据
- 小样本数据

**优点：**
- 理论基础扎实
- 泛化能力强
- 适合高维数据

**缺点：**
- 对大规模数据效率低
- 对参数敏感

**参数说明：**
- `C`: 正则化参数
- `kernel`: 核函数类型（linear、poly、rbf、sigmoid）
- `gamma`: 核函数系数

**配置示例：**

.. code-block:: yaml

   models:
     SVM:
       params:
         C: 1.0
         kernel: rbf
         gamma: scale
         random_state: 42

**KNN**

K 近邻，适用于分类和回归任务。

**适用场景：**
- 分类和回归任务
- 简单快速
- 非线性关系

**优点：**
- 简单易用
- 无训练过程
- 适合非线性关系

**缺点：**
- 预测速度慢
- 对异常值敏感
- 需要存储所有训练数据

**参数说明：**
- `n_neighbors`: 近邻数量
- `weights`: 权重计算方式（uniform、distance）
- `algorithm`: 算法类型（auto、ball_tree、kd_tree、brute）

**配置示例：**

.. code-block:: yaml

   models:
     KNN:
       params:
         n_neighbors: 5
         weights: uniform
         algorithm: auto

**AutoGluonTabular**

AutoML 框架，自动选择和优化模型。

**适用场景：**
- 快速原型
- 不需要调参
- 多模型比较

**优点：**
- 自动化程度高
- 性能优秀
- 易于使用

**缺点：**
- 计算资源消耗大
- 黑盒模型

**参数说明：**
- `time_limit`: 训练时间限制（秒）
- `presets`: 预设模式（best_quality、high_quality、good_quality、medium_quality）

**配置示例：**

.. code-block:: yaml

   models:
     AutoGluonTabular:
       params:
         path: ./results/ml/autogluon_models
         label: Label
         time_limit: 3600
         presets: high_quality

Pipeline 机制
------------

HABIT 使用 scikit-learn 的 Pipeline 机制，这是避免数据泄露的关键设计。

**什么是数据泄露？**

数据泄露是指在模型训练过程中，测试集的信息意外地泄露到训练集中，导致模型性能被高估。

**Pipeline 如何避免数据泄露？**

1. **特征选择**: 在交叉验证的每个 fold 内进行特征选择
2. **模型训练**: 在交叉验证的每个 fold 内进行模型训练
3. **严格分离**: 训练集和测试集完全分离，避免测试集信息泄露

**机器学习中的 Pipeline:**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from habit.core.machine_learning import ModelFactory

   # 创建 Pipeline，包含特征选择和模型训练
   pipeline = Pipeline([
       ('feature_selection', feature_selector),
       ('model', ModelFactory.create_model('RandomForest', config))
   ])

   # 训练阶段
   pipeline.fit(X_train, y_train)

   # 测试阶段：使用训练好的 Pipeline 进行预测
   y_pred = pipeline.predict(X_test)

**关键要点：**

- 特征选择和模型训练必须在同一个 Pipeline 中
- 不要在整个数据集上进行特征选择
- 使用交叉验证时，每个 fold 的训练和预测必须严格分离

实际示例
--------

**示例 1: 基本训练配置（MLConfig）**

.. code-block:: yaml

   input:
     - path: ./ml_data/clinical_feature.csv
       subject_id_col: PatientID
       label_col: Label
   output: ./results/ml/train

   split_method: stratified
   test_size: 0.3

   feature_selection_methods:
     - method: variance
       params:
         threshold: 0.0

   models:
     RandomForest:
       params:
         n_estimators: 100
         random_state: 42

   is_visualize: true
   is_save_model: true
   random_state: 42

**示例 2: 多模型训练配置（MLConfig）**

.. code-block:: yaml

   input:
     - path: ./ml_data/clinical_feature.csv
       subject_id_col: PatientID
       label_col: Label
   output: ./results/ml/train

   feature_selection_methods:
     - method: rfecv
       params:
         estimator: LogisticRegression
         cv: 5
         scoring: accuracy

   models:
     LogisticRegression:
       params:
         max_iter: 1000
         random_state: 42
     XGBoost:
       params:
         n_estimators: 100
         max_depth: 6
         learning_rate: 0.1
         random_state: 42

   is_visualize: true
   is_save_model: true
   random_state: 42

**示例 3: 预测配置（统一 MLConfig + run_mode='predict'）**

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./results/ml/train/models/LogisticRegression_final_pipeline.pkl
   output: ./results/ml/predict      # 输出目录
   input:
     - path: ./ml_data/new_data.csv  # 待预测的数据
       subject_id_col: PatientID
       label_col: Label              # 仅在 evaluate=true 时用作 ground truth
   evaluate: true
   output_label_col: predicted_label
   output_prob_col: predicted_probability

输出结构
--------

机器学习输出会保存在配置中指定目录，典型包括：

- 训练阶段：模型文件、训练/测试性能汇总、可视化图表、日志文件
- 预测阶段：`prediction_results.csv`（含预测标签与概率列）、可选评估结果、日志文件

常见问题
--------

**Q1: 如何选择特征选择方法？**

A: 根据您的数据特点选择：
- **快速原型**: 使用 variance 或 correlation
- **分类任务**: 使用 anova 或 chi2
- **线性模型**: 使用 lasso
- **精确选择**: 使用 rfecv

**Q2: 如何选择模型类型？**

A: 根据您的任务和数据特点选择：
- **二分类**: LogisticRegression、SVM
- **高维数据**: RandomForest、XGBoost
- **快速原型**: KNN
- **自动调优**: AutoGluon

**Q3: 训练和预测模式有什么区别？**

A: 
- **训练模式 (Train)**: 
    *   **输入**：带标签的特征表。
    *   **过程**：执行数据标准化、特征筛选（如 LASSO）、模型拟合（如 Random Forest）以及交叉验证评估。
    *   **输出**：模型性能指标（AUC, ACC 等）和保存的 **Pipeline 文件** (`.pkl`)。
- **预测模式 (Predict)**: 
    *   **输入**：新数据的特征表 + 已训练的 **Pipeline 文件**。
    *   **过程**：直接加载 Pipeline，对新数据应用完全相同的标准化和筛选逻辑，并输出预测得分。
    *   **输出**：每个样本的预测概率（Radscore）或类别。

**Q4: 如何避免数据泄露？**

A: 遵循以下原则：

1. 特征选择和模型训练必须在同一个 Pipeline 中。
2. 不要在整个数据集上进行特征选择。
3. 使用交叉验证时，每个 fold 的训练和预测必须严格分离。

**Q5: 模型训练失败怎么办？**

A: 检查以下几点：
1. 配置文件是否正确
2. 数据路径是否正确
3. 特征文件是否存在
4. 标签文件是否存在
5. 查看日志文件了解详细错误信息

**Q6: 如何提高模型性能？**

A: 可以尝试以下方法：
1. 尝试不同的特征选择方法
2. 尝试不同的模型类型
3. 调整模型参数
4. 增加训练数据
5. 特征工程

下一步
-------

机器学习建模完成后，您可以：

- :doc:`../customization/index_zh`: 了解如何自定义模型和特征选择器
- :doc:`../configuration_zh`: 了解配置文件的详细说明
- :doc:`../../design_philosophy_zh`: 了解 HABIT 的设计哲学
