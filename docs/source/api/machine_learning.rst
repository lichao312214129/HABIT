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

核心数据对象 (Core Contracts)
-----------------------------

.. automodule:: habit.core.machine_learning.core.plan
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.core.results
   :members:
   :undoc-members:
   :show-inheritance:

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

执行器层 (Runners)
------------------

从 2026-05 起，K-Fold 计算主循环已抽离到 ``runners`` 子包，
用于降低 workflow 类复杂度，同时保持外部调用方式不变。

.. automodule:: habit.core.machine_learning.runners.kfold
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.runners.holdout
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   ``MachineLearningKFoldWorkflow`` 仍然是对外入口，但其内部已委托给
   :py:class:`habit.core.machine_learning.runners.kfold.KFoldRunner` 执行折内训练与聚合。

输出组件说明
------------

``habit.core.machine_learning.callbacks`` 已从代码中移除。
模型保存、报表写入和可视化统一使用 ``reporting`` 子包中的显式组件。

.. automodule:: habit.core.machine_learning.reporting.model_store
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.reporting.report_writer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.reporting.plot_composer
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
