机器学习建模
============

使用表格特征（生境/影像组学 CSV、临床表等）训练与评估预测模型。

运行
----

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"    # 示例路径，请改为本机项目根目录
   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

融合临床特征时，另用 ``config/machine_learning/config_machine_learning_clinical.yaml`` 等模板；K 折示例见 ``config_machine_learning_kfold_demo.yaml``。

数据
----

- 输入为 **CSV / Excel** 等表格；在 YAML 中指定特征表路径、标签列、ID 列等（见配置参考）
- 多表融合：将影像特征与临床指标合并为一张表，或分别建模后在「模型对比」中比较

输出
----

模型、``all_prediction_results.csv`` 、``evaluation_metrics.csv`` 及 ROC/校准等图表 PDF，目录由配置 ``output`` 决定。Demo： ``demo_data/results/ml/radiomics/`` 、``demo_data/results/ml/clinical/`` （特征表仍从 ``demo_data/ml_data/`` 读取）。

配置
----

特征选择、模型类型、交叉验证等见 :doc:`../configuration_zh` 中 **MachineLearning** 相关章节。
