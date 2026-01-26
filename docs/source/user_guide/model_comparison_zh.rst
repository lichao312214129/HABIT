模型对比
========

概述
----

在影像组学研究中，我们经常需要回答一个核心问题：**不同特征来源的模型哪个更好？**

HABIT 提供了模型对比工具，帮助您比较来自不同特征集或不同建模策略的预测模型：

- **临床模型** vs **影像组学模型**
- **单纯影像组学模型** vs **混合模型**（影像组学 + 临床特征）
- **不同影像组学特征集**的模型对比
- **不同预处理策略**的模型对比

为什么需要模型对比？

影像组学研究中，对比分析是发表论文的关键：

1. **验证影像组学的附加价值**: 影像组学特征能否提升纯临床模型的预测能力？
2. **特征贡献分析**: 哪些特征对预测贡献最大？
3. **临床净收益评估**: 通过 DCA 评估模型在实际临床决策中的价值
4. **模型稳定性验证**: 比较不同特征集下模型的稳定性

实际研究场景
------------

**场景1：临床模型 vs 影像组学模型**

这是最常见的研究场景，目的是验证影像组学特征的预测价值：

- 临床模型：仅使用年龄、肿瘤大小、分期等临床特征
- 影像组学模型：仅使用从医学图像提取的影像组学特征
- 问题：影像组学特征能否提供超越临床因素的预测价值？

**场景2：单纯影像组学模型 vs 混合模型**

评估临床因素与影像组学特征的互补性：

- 影像组学模型：仅使用影像组学特征
- 混合模型：同时使用影像组学特征和临床特征
- 问题：加入临床特征后，模型性能提升多少？

**场景3：不同特征集的影像组学模型**

比较不同特征选择策略或特征来源：

- 瘤内特征 vs 瘤周特征
- 不同聚类策略的特征
- 不同放射组学特征（形状、纹理、一阶统计等）

配置说明
--------

模型对比需要创建配置文件，指定要对比的模型预测结果文件。

**配置文件结构：**

.. code-block:: yaml

   # 输出目录
   output_dir: ./ml_data/model_comparison

   # 要比较的模型预测文件列表
   files_config:
     - path: ./ml_data/radiomics/all_prediction_results.csv
       model_name: radiomics
       subject_id_col: subject_id
       label_col: label
       prob_col: LogisticRegression_prob
       pred_col: LogisticRegression_pred
       split_col: dataset

     - path: ./ml_data/clinical/all_prediction_results.csv
       model_name: clinical
       subject_id_col: subject_id
       label_col: label
       prob_col: LogisticRegression_prob
       pred_col: LogisticRegression_pred
       split_col: dataset

     - path: ./ml_data/combined/all_prediction_results.csv
       model_name: combined
       subject_id_col: subject_id
       label_col: label
       prob_col: LogisticRegression_prob
       pred_col: LogisticRegression_pred
       split_col: dataset

   # 合并数据配置
   merged_data:
     enabled: true
     save_name: combined_predictions.csv

   # 数据分割配置
   split:
     enabled: true

   # 可视化配置
   visualization:
     roc:
       enabled: true
       save_name: roc_curves.pdf
       title: ROC Curves
     dca:
       enabled: true
       save_name: decision_curves.pdf
       title: Decision Curves
     calibration:
       enabled: true
       save_name: calibration_curves.pdf
       n_bins: 5
       title: Calibration Curves
     pr_curve:
       enabled: true
       save_name: precision_recall_curves.pdf
       title: Precision-Recall Curves

   # DeLong 检验配置
   delong_test:
     enabled: true
     save_name: delong_results.json

   # 性能指标配置
   metrics:
     basic_metrics:
       enabled: true
     youden_metrics:
       enabled: true
     target_metrics:
       enabled: true
       targets:
         sensitivity: 0.91
         specificity: 0.91

**字段说明：**

- **output_dir**: 结果输出目录
- **files_config**: 要对比的模型文件列表
  - **path**: 预测结果文件路径
  - **model_name**: 模型名称（用于图例显示）
  - **subject_id_col**: 患者ID列名
  - **label_col**: 真值列名
  - **prob_col**: 预测概率列名
  - **pred_col**: 预测类别列名
  - **split_col**: 数据分割列名（如 train/test）
- **merged_data**: 合并数据配置
  - **enabled**: 是否创建合并数据集
  - **save_name**: 合并文件名
- **split**: 数据分割配置
  - **enabled**: 是否分别分析不同分割
- **visualization**: 可视化配置
  - **roc**: ROC 曲线配置
  - **dca**: 决策曲线分析配置
  - **calibration**: 校准曲线配置
  - **pr_curve**: PR 曲线配置
- **delong_test**: DeLong 检验配置
- **metrics**: 性能指标配置

预测结果文件格式
----------------

预测结果文件应包含以下列：

.. csv-table::
   :header: "subject_id", "label", "LogisticRegression_prob", "LogisticRegression_pred", "dataset"
   :widths: 15, 10, 25, 25, 15

   "sub-001", 0, 0.12, 0, "test"
   "sub-002", 1, 0.78, 1, "test"
   "sub-003", 1, 0.65, 1, "train"

- **subject_id**: 患者ID
- **label**: 真实标签 (0 或 1)
- **LogisticRegression_prob**: 模型预测的概率值 (0-1)
- **LogisticRegression_pred**: 模型的预测类别 (0 或 1)
- **dataset**: 数据集分割标识 (train/test/validation)

使用方法
--------

**1. 创建配置文件**

根据上述格式创建配置文件 `config_model_comparison.yaml`。

**2. 运行命令**

.. code-block:: bash

   habit compare --config config_model_comparison.yaml

**3. 查看结果**

运行完成后，在 `output_dir` 目录下查看结果文件。

输出文件说明
------------

模型对比工具会生成以下文件：

**图表文件：**

- ``roc_curves.pdf``: 所有模型的 ROC 曲线对比图
- ``decision_curves.pdf``: 决策曲线分析图
- ``calibration_curves.pdf``: 校准曲线对比图
- ``precision_recall_curves.pdf``: PR 曲线对比图

**数据文件：**

- ``combined_predictions.csv``: 合并的预测结果
- ``delong_results.json``: DeLong 检验结果
- ``ml_standard_summary.csv``: 性能指标汇总表

**ml_standard_summary.csv 示例：**

.. csv-table::
   :header: "Model", "Dataset", "AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1-Score"
   :widths: 15, 10, 10, 10, 10, 10, 10, 10, 10

   "radiomics", "test", 0.81, 0.75, 0.78, 0.72, 0.74, 0.76, 0.76
   "clinical", "test", 0.72, 0.68, 0.70, 0.65, 0.67, 0.68, 0.68
   "combined", "test", 0.87, 0.80, 0.82, 0.78, 0.79, 0.81, 0.80

指标说明
--------

- **AUC (ROC曲线下面积)**: 衡量模型区分能力的综合指标，越接近 1 越好
- **Accuracy (准确率)**: 正确预测的比例
- **Sensitivity (灵敏度)**: 正确识别阳性的能力
- **Specificity (特异度)**: 正确识别阴性的能力
- **PPV (阳性预测值)**: 预测为阳性中真正阳性的比例
- **NPV (阴性预测值)**: 预测为阴性中真正阴性的比例
- **F1-Score**: 精确率和召回率的调和平均

图表解读
--------

**ROC 曲线对比图**

.. image:: ../images/roc_curves_example.png
   :alt: ROC曲线对比图示例
   :align: center

- X 轴：假阳性率 (1 - Specificity)
- Y 轴：真阳性率 (Sensitivity)
- 对角线：随机猜测的性能
- 曲线越靠近左上角，性能越好
- AUC 值越大，性能越好
- 在影像组学研究中，混合模型的 AUC 通常高于单一模型

**校准曲线对比图**

.. image:: ../images/calibration_curves_example.png
   :alt: 校准曲线对比图示例
   :align: center

- X 轴：预测概率的平均值
- Y 轴：实际阳性比例
- 对角线：完美校准
- 曲线越接近对角线，校准越好
- 校准曲线反映模型的概率预测是否可靠

**决策曲线分析 (DCA)**

.. image:: ../images/dca_curves_example.png
   :alt: 决策曲线分析图示例
   :align: center

- X 轴：阈值概率（临床决策的临界点）
- Y 轴：净收益
- 横线：全部预测为阴性的净收益
- 斜线：全部预测为阳性的净收益
- 曲线越高，表示在临床决策中净收益越大
- DCA 是评估模型临床价值的重要工具

研究建议
--------

1. **统计检验**: 使用 DeLong 检验比较不同模型的 AUC 差异
2. **净收益分析**: 关注临床决策曲线，评估模型的实用价值
3. **敏感性分析**: 尝试不同的特征选择策略，比较模型稳定性
4. **可视化展示**: ROC + 校准曲线 + DCA 三个图一起展示

实际应用建议
------------

1. **优先关注 AUC 提升**: 混合模型 vs 临床模型的 AUC 差值通常为 0.05-0.15
2. **关注临床指标**: 根据临床需求关注灵敏度、特异度等
3. **考虑校准程度**: 概率预测的校准程度对临床决策很重要
4. **综合考虑**: 不要只看单一指标，要综合考虑所有指标

与训练流程的结合
----------------

模型对比工具可以与机器学习训练流程结合使用：

1. 训练临床模型
2. 训练影像组学模型
3. 训练混合模型
4. 使用模型对比工具比较所有模型
5. 选择最佳模型进行后续分析

.. code-block:: bash

   # 1. 训练临床模型
   habit model --config config_machine_learning_clinical.yaml

   # 2. 训练影像组学模型
   habit model --config config_machine_learning_radiomics.yaml

   # 3. 训练混合模型
   habit model --config config_machine_learning_combined.yaml

   # 4. 对比所有模型
   habit compare --config config_model_comparison.yaml

注意事项
--------

1. 所有模型的预测结果应基于同一测试集
2. 预测结果文件格式需要保持一致
3. 确保列名（label_col, prob_col 等）正确匹配
4. 阈值选择会影响 DCA 结果的解释
5. 建议使用交叉验证以获得更稳定的结果
6. 使用 DeLong 检验评估 AUC 差异的统计显著性

常见问题
--------

**Q: 为什么混合模型的 AUC 不一定比单一模型高？**

A: 可能原因：
- 临床特征已经包含了主要的预测信息
- 影像组学特征与临床特征高度相关
- 特征维度太高导致过拟合
- 需要进行更严格的特征选择

**Q: 如何确定使用哪些特征组合？**

A: 建议采用逐步策略：
1. 先用临床特征建立基线模型
2. 添加影像组学特征看性能提升
3. 进行特征选择，去除冗余特征
4. 验证最终模型的稳定性

**Q: DCA 分析有什么意义？**

A: DCA 评估模型在临床决策中的净收益：
- 考虑不同阈值下的真阳性/假阳性权衡
- 评估模型是否值得临床使用
- 特别适用于筛查场景的模型评估

**Q: pred_col 和 prob_col 有什么区别？**

A:
- **prob_col**: 模型预测的概率值（连续值 0-1），用于绘制 ROC 曲线和 DCA
- **pred_col**: 基于默认阈值（0.5）的类别预测（0 或 1），用于计算混淆矩阵相关指标
