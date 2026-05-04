machine_learning 模块
========================

.. automodule:: habit.core.machine_learning
   :members:
   :undoc-members:
   :show-inheritance:

开发者阅读入口
----------------

理解机器学习模块时，建议先从 ``MLConfig``、``BaseWorkflow``、
``PipelineBuilder``、``ModelFactory`` 和 ``run_selector`` 读起。整体代码架构
和模块边界见 :doc:`../development/module_architecture`。

工作流 (Workflows)
-------------------

工作流类封装了完整的机器学习过程，包括训练、验证和预测。

.. automodule:: habit.core.machine_learning.workflows.holdout_workflow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.workflows.kfold_workflow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.workflows.base
   :members:
   :undoc-members:
   :show-inheritance:

模型工厂 (Model Factory)
-------------------------

.. automodule:: habit.core.machine_learning.models.factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.models.ensemble
   :members:
   :undoc-members:
   :show-inheritance:

评估工具 (Evaluation)
----------------------

.. automodule:: habit.core.machine_learning.evaluation.model_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.evaluation.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.workflows.comparison_workflow
   :members:
   :undoc-members:
   :show-inheritance:

可视化 (Visualization)
-----------------------

.. automodule:: habit.core.machine_learning.visualization.plotting
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.visualization.km_survival
   :members:
   :undoc-members:
   :show-inheritance:
