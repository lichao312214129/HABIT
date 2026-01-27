机器学习建模
============

本节介绍如何使用 HABIT 进行机器学习建模，包括特征选择、模型训练和模型评估。

使用场景
--------

在 HABIT 中，机器学习通常有两种主要使用场景：

**1. 训练场景 (Train Mode)**
   *   **目的**：基于已有的带标签数据（如良恶性、生存期等），通过特征筛选和算法拟合，构建预测模型。
   *   **操作**：设置 `run_mode: train`。HABIT 会执行交叉验证，并保存最终的 Pipeline 文件（`.pkl`）。

**2. 预测场景 (Predict Mode)**
   *   **目的**：将训练好的模型应用于全新的、未见过的数据，获取预测结果（如 Radscore）。
   *   **操作**：设置 `run_mode: predict`，并指定 `pipeline_path`。

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
- **输出**: 在 `all_prediction_results.csv` 文件中，`<model_name>_prob` 列即为对应模型的 Radscore。

如何训练融合模型
----------------

在医学影像研究中，"融合模型"（Fusion Model）通常指结合了**影像特征（如 Radscore）**和**临床特征**的模型。

**训练融合模型通常有两种方式：**

**方式 1：早期融合 (Early Fusion) - 推荐**

将生境特征（Habitat Features）与临床特征（Clinical Features）在进入模型前直接进行表格合并。

1. **准备影像特征**: 运行生境特征提取，导出 CSV。
2. **准备临床特征**: 准备包含相同 `PatientID` 的临床特征 CSV/Excel。
3. **合并表格**: 使用 :doc:`../app_merge_csv_zh` 按 ID 合并，得到单一融合特征表。
4. **训练模型**: 在 `habit model` 配置中，将合并后的表作为 `data_dir` 输入，按常规流程训练模型。

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

   # data source
   data_dir: ./demo_data/ml_data/fusion_features.csv

   # columns
   subject_id_col: PatientID
   label_col: Label

   # training
   run_mode: train

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

   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.machine_learning.config_schemas import MLConfig

   # 加载配置
   config = MLConfig.from_file('./config_machine_learning.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config)

   # 创建机器学习工作流
   workflow = configurator.create_ml_workflow()

   # 运行工作流
   workflow.run_pipeline()

**详细示例：**

.. code-block:: python

   import logging
   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.machine_learning.config_schemas import MLConfig
   from habit.utils.log_utils import setup_logger
   from pathlib import Path

   # 设置日志
   output_dir = Path('./results/ml')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='machine_learning',
       output_dir=output_dir,
       log_filename='machine_learning.log',
       level=logging.INFO
   )

   # 加载配置
   config = MLConfig.from_file('./config_machine_learning.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))

   # 创建机器学习工作流
   workflow = configurator.create_ml_workflow()

   # 运行工作流
   logger.info("开始训练模式")
   workflow.run_pipeline()
   logger.info("模型训练完成！")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 运行模式和 Pipeline 设置（首先检查这些！）
   run_mode: train  # train（如果要训练新模型，设置为 train）或 predict（如果要使用预训练模型，设置为 predict）
   pipeline_path: ./results/ml/model_pipeline.pkl  # predict 模式的 Pipeline 路径（predict 模式必需）

   # 数据路径
   data_dir: ./files_ml.yaml
   output: ./results/ml/train

   # 特征选择设置
   FeatureSelection:
     enabled: true  # 是否启用特征选择
     method: variance  # 特征选择方法：variance、correlation、anova、chi2、lasso、rfecv
     params:
       # variance 方法参数
       threshold: 0.0

       # correlation 方法参数
       threshold_correlation: 0.95

       # anova 方法参数
       k_features: 10

       # chi2 方法参数
       k_features_chi2: 10

       # lasso 方法参数
       alpha: 0.01

       # rfecv 方法参数
       estimator: RandomForest
       cv: 5
       scoring: accuracy

   # 模型配置
   models:
     RandomForest:
       params:
         n_estimators: 100
         max_depth: null
         min_samples_split: 2
         min_samples_leaf: 1
         random_state: 42

     LogisticRegression:
       params:
         C: 1.0
         penalty: l2
         solver: lbfgs
         max_iter: 1000
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
         random_state: 42

     KNN:
       params:
         n_neighbors: 5
         weights: uniform
         algorithm: auto

     AutoGluon:
       params:
         time_limit: 3600
         presets: best_quality

   # 模型评估设置
   ModelEvaluation:
     enabled: true  # 是否启用模型评估
     metrics:
       - accuracy
       - precision
       - recall
       - f1
       - roc_auc
       - confusion_matrix
     cv: 5  # 交叉验证折数
     test_size: 0.2  # 测试集比例
     random_state: 42

   # 模型保存设置
   ModelSaving:
     enabled: true  # 是否启用模型保存
     save_path: ./results/ml/model_pipeline.pkl  # 模型保存路径
     save_format: pkl  # 保存格式：pkl、joblib

   # 通用设置
   processes: 2  # 并行进程数
   random_state: 42  # 可重复性的随机种子
   debug: false  # 启用详细日志的调试模式

**字段说明：**

**run_mode**: 运行模式

- `train`: 训练新模型
- `predict`: 使用预训练模型进行预测

**pipeline_path**: Pipeline 文件路径

- predict 模式必需
- 指定训练好的 Pipeline 文件路径

**data_dir**: 数据目录路径

- 可以是文件夹或 YAML 配置文件
- 参考 :doc:`../data_structure_zh` 了解数据结构

**output**: 输出目录路径

- 模型、评估结果和预测结果将保存在此目录
- 输出目录结构示例：

.. code-block:: text

   output/
   ├── models/                          # 模型文件目录
   │   ├── fold_1/
   │   │   ├── LogisticRegression_pipeline.pkl
   │   │   └── RandomForest_pipeline.pkl
   │   ├── fold_2/
   │   │   ├── LogisticRegression_pipeline.pkl
   │   │   └── RandomForest_pipeline.pkl
   │   ├── LogisticRegression_final_pipeline.pkl
   │   └── RandomForest_final_pipeline.pkl
   │
   ├── LogisticRegression_results.json      # 详细评估结果
   ├── LogisticRegression_summary.csv       # 汇总结果
   ├── RandomForest_results.json
   ├── RandomForest_summary.csv
   │
   ├── all_prediction_results.csv           # 所有预测结果
   ├── performance_table.csv                # 性能指标表
   ├── performance_detailed.csv             # 详细性能报告
   ├── merged_predictions.csv               # 合并预测结果
   ├── delong_comparison.json               # DeLong检验对比结果
   │
   ├── ROC_test.pdf                         # ROC曲线
   ├── ROC_train.pdf
   ├── DCA_test.pdf                         # DCA曲线
   ├── Calibration_test.pdf                 # 校准曲线
   └── PR_curve_test.pdf                    # PR曲线

**FeatureSelection**: 特征选择设置

- `enabled`: 是否启用特征选择
- `method`: 特征选择方法
- `params`: 特征选择方法的参数

**models**: 模型配置

- 支持配置多个模型，所有模型都会被训练和评估
- 每个模型以字典形式配置，包含 `params` 参数
- 支持的模型类型：LogisticRegression、RandomForest、XGBoost、SVM、KNN、AutoGluon

**ModelEvaluation**: 模型评估设置

- `enabled`: 是否启用模型评估
- `metrics`: 评估指标列表
- `cv`: 交叉验证折数
- `test_size`: 测试集比例

**ModelSaving**: 模型保存设置

- `enabled`: 是否启用模型保存
- `save_path`: 模型保存路径
- `save_format`: 保存格式

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

   FeatureSelection:
     enabled: true
     method: variance
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

   FeatureSelection:
     enabled: true
     method: correlation
     params:
       threshold_correlation: 0.95

**ANOVA (anova)**

基于 ANOVA F 值的特征选择，选择与目标变量最相关的特征。

**适用场景：**
- 分类任务
- 选择与目标变量最相关的特征

**参数说明：**
- `k_features`: 要选择的特征数量

**配置示例：**

.. code-block:: yaml

   FeatureSelection:
     enabled: true
     method: anova
     params:
       k_features: 10

**Chi2 (chi2)**

基于卡方检验的特征选择，选择与目标变量最相关的特征。

**适用场景：**
- 分类任务
- 非负特征

**参数说明：**
- `k_features_chi2`: 要选择的特征数量

**配置示例：**

.. code-block:: yaml

   FeatureSelection:
     enabled: true
     method: chi2
     params:
       k_features_chi2: 10

**LASSO (lasso)**

基于 LASSO 回归的特征选择，使用 L1 正则化进行特征选择。

**适用场景：**
- 线性模型
- 自动特征选择

**参数说明：**
- `alpha`: L1 正则化强度

**配置示例：**

.. code-block:: yaml

   FeatureSelection:
     enabled: true
     method: lasso
     params:
       alpha: 0.01

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

   FeatureSelection:
     enabled: true
     method: rfecv
     params:
       estimator: RandomForest
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

**AutoGluon**

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
     AutoGluon:
       params:
         time_limit: 3600
         presets: best_quality

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

**示例 1: 基本机器学习工作流**

.. code-block:: yaml

   run_mode: train
   data_dir: ./files_ml.yaml
   output: ./results/ml/train

   FeatureSelection:
     enabled: true
     method: variance
     params:
       threshold: 0.0

   ModelTraining:
     enabled: true
     model_type: RandomForest
     params:
       n_estimators: 100
       random_state: 42

   ModelEvaluation:
     enabled: true
     metrics:
       - accuracy
       - precision
       - recall
       - f1
     cv: 5
     test_size: 0.2
     random_state: 42

   ModelSaving:
     enabled: true
     save_path: ./results/ml/model_pipeline.pkl
     save_format: pkl

   processes: 2
   random_state: 42

**示例 2: 完整机器学习工作流**

.. code-block:: yaml

   run_mode: train
   data_dir: ./files_ml.yaml
   output: ./results/ml/train

   FeatureSelection:
     enabled: true
     method: rfecv
     params:
       estimator: RandomForest
       cv: 5
       scoring: accuracy

   ModelTraining:
     enabled: true
     model_type: XGBoost
     params:
       n_estimators: 100
       max_depth: 6
       learning_rate: 0.1
       random_state: 42

   ModelEvaluation:
     enabled: true
     metrics:
       - accuracy
       - precision
       - recall
       - f1
       - roc_auc
       - confusion_matrix
     cv: 5
     test_size: 0.2
     random_state: 42

   ModelSaving:
     enabled: true
     save_path: ./results/ml/model_pipeline.pkl
     save_format: pkl

   processes: 4
   random_state: 42

**示例 3: 预测模式**

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./results/ml/train/model_pipeline.pkl
   data_dir: ./files_ml.yaml
   output: ./results/ml/predict

   processes: 2
   random_state: 42

输出结构
--------

机器学习的输出结构：

.. code-block:: text

   results/ml/
   ├── train/                    # 训练模式输出
   │   ├── model_pipeline.pkl    # 训练好的 Pipeline
   │   ├── model/                # 模型文件
   │   │   └── ...
   │   ├── evaluation/           # 评估结果
   │   │   ├── metrics.csv
   │   │   ├── confusion_matrix.png
   │   │   ├── roc_curve.png
   │   │   └── ...
   │   ├── feature_importance/   # 特征重要性
   │   │   └── ...
   │   ├── predictions/          # 预测结果
   │   │   └── ...
   │   └── machine_learning.log  # 日志文件
   └── predict/                  # 预测模式输出
       ├── predictions/
       │   └── predictions.csv
       └── machine_learning.log

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
