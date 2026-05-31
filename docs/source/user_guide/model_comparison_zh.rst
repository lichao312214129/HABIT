模型对比
========

比较多个已训练模型的预测结果（如纯临床 vs 影像组学），生成对比图表与统计结果。

运行
----

先分别训练待比较的模型，再执行对比：

.. code-block:: bash

   conda activate habit
   cd D:\HABIT-main
   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train
   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

输出
----

结果保存在配置 ``output_dir``（demo 为 ``demo_data/results/model_comparison/``）；需先完成步骤 4，使 ``demo_data/results/ml/radiomics/`` 与 ``clinical/`` 下各有 ``all_prediction_results.csv``。

配置
----

对比的预测文件路径、指标与作图选项见 :doc:`../configuration_zh` 中模型对比相关章节。Demo 模板：`config/model_comparison/config_model_comparison_demo.yaml`。
