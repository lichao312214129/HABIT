模型比较工具 (habit compare)
==============================

概述
----

在完成机器学习训练后，您可能训练了多个模型（例如逻辑回归、随机森林、XGBoost）。**HABIT Compare** 工具可以帮助您将这些模型的性能画在同一张图上，方便直观比较。

它可以自动生成：
*   **ROC 曲线对比图**：比较不同模型的区分能力。
*   **校准曲线对比图**：比较不同模型的预测概率准确性。
*   **决策曲线对比图 (DCA)**：评估模型的临床净收益。

使用方法
--------

**1. 准备预测结果文件**

在使用 `habit model` 训练后，每个模型都会生成一个预测结果 CSV 文件。您需要创建一个配置文件（例如 `config_compare.yaml`），告诉工具这些文件在哪里。

**配置文件示例 (config_compare.yaml):**

.. code-block:: yaml

   # 输出目录
   out_dir: ./results/comparison

   # 要比较的模型列表
   models:
     - name: LogisticRegression          # 模型名称（显示在图例中）
       file: ./results/ml/lr/predictions.csv  # 预测结果文件路径
       # 告诉工具哪一列是真值，哪一列是预测概率
       label_col: Label
       prob_col: Probability

     - name: RandomForest
       file: ./results/ml/rf/predictions.csv
       label_col: Label
       prob_col: Probability

     - name: XGBoost
       file: ./results/ml/xgb/predictions.csv
       label_col: Label
       prob_col: Probability

**2. 运行命令**

.. code-block:: bash

   habit compare --config config_compare.yaml

输出结果
--------

运行完成后，您将在 `out_dir` 目录下看到：

*   `roc_curves.png`: 所有模型的 ROC 曲线对比。
*   `calibration_curves.png`: 校准曲线对比。
*   `dca_curves.png`: 决策曲线分析对比。
*   `metrics_comparison.csv`: 包含 AUC, Accuracy, F1-Score 等指标的汇总表格。
