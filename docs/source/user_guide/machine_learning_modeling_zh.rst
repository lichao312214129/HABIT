机器学习建模
==============

概述
----

机器学习建模是生境分析的最终步骤，使用提取的生境特征进行预测模型训练和评估。

.. note::

   在开始之前，请确保您的数据已整理为 CSV 格式，且您已清楚哪一列是 ID，哪一列是标签（Label）。

数据准备指南
------------

这是医生用户最关心的部分。您的输入数据应为一个标准的 CSV 文件（可以使用 Excel 另存为 CSV），格式如下：

**CSV 文件示例 (data.csv):**

.. csv-table::
   :header: "PatientID", "Label", "Feature_1", "Feature_2", "...", "Feature_N"
   :widths: 15, 10, 15, 15, 5, 15

   "sub-001", 0, 12.5, 0.45, "...", 102.3
   "sub-002", 1, 14.2, 0.67, "...", 98.1
   "sub-003", 0, 11.8, 0.33, "...", 105.4

**关键要求：**

1.  **ID 列** (`PatientID`): 每一行代表一个病人，ID 必须唯一。
2.  **标签列** (`Label`): 您要预测的目标。
    *   **二分类**: 通常用 `0` (阴性/良性) 和 `1` (阳性/恶性) 表示。
    *   **回归**: 可以是连续数值（如生存时间）。
3.  **特征列**: 除了 ID 和 Label 外的其他列，HABIT 会自动将其识别为特征用于训练。
4.  **无中文**: 表头和内容尽量避免中文，以免出现编码错误。

**如何在配置文件中指定列名？**

您需要在数据配置文件（通常是 `files_ml.yaml`）中告诉 HABIT 哪一列是 ID，哪一列是 Label：

.. code-block:: yaml

   # files_ml.yaml
   - path: ./demo_data/ml_data/clinical_feature.csv  # 您的 CSV 文件路径
     subject_id_col: PatientID                       # 对应 CSV 中的 ID 列名
     label_col: Label                                # 对应 CSV 中的标签 列名

HABIT 提供了完整的机器学习工作流程：

1. **数据加载与清洗**: 自动处理缺失值
2. **特征选择**: 使用 mRMR、LASSO 等算法筛选最有效的特征
3. **模型训练**: 支持多种机器学习算法（XGBoost, Random Forest, SVM, LR 等）
4. **模型评估**: 计算 AUC, Accuracy, Sensitivity, Specificity 等指标
5. **AutoGluon 集成**: 支持自动机器学习，自动搜索最佳模型

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit model --config <config_file> [--mode <train|predict>]

**常用参数：**

*   `--config`, `-c`: 配置文件路径（必需）。
*   `--mode`, `-m`: 运行模式，`train` (训练) 或 `predict` (预测)。默认为 `train`。

**示例：**

.. code-block:: bash

   # 训练模型
   habit model --config config_machine_learning.yaml --mode train

   # 使用训练好的模型进行预测
   habit model --config config_machine_learning.yaml --mode predict

Python API 使用方法
------------------

.. code-block:: python

   from habit.core.machine_learning.workflows.holdout_workflow import MachineLearningWorkflow
   from habit.core.machine_learning.config_schemas import MachineLearningConfig

   # 加载配置
   config = MachineLearningConfig.from_file('config_machine_learning.yaml')

   # 创建工作流
   workflow = MachineLearningWorkflow(config)

   # 运行
   workflow.run()

YAML 配置详解
--------------

.. code-block:: yaml

   # config_machine_learning.yaml

   # 实验名称
   experiment_name: "My_First_Experiment"
   output_dir: "./results/ml"

   # 数据设置
   input_data:
     - path: "./data.csv"
       subject_id_col: "PatientID"
       label_col: "Label"

   # 预处理
   preprocessing:
     imputation: "mean"  # 缺失值填充: mean, median, most_frequent
     normalization: "zscore" # 归一化: zscore, minmax

   # 特征选择
   feature_selection:
     method: "lasso"     # mrmr, lasso, rfe, anova
     k: 10               # 选择特征数量

   # 模型设置
   model:
     type: "xgboost"     # xgboost, rf, svm, lr, autogluon
     params:             # 模型参数 (可选)
       n_estimators: 100
       max_depth: 5

   # 评估设置
   evaluation:
     cv: 5               # 交叉验证折数 (如果使用 cv 命令)
     metrics: ["auc", "accuracy", "sensitivity", "specificity"]
