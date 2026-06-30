machine_learning module
=========================

.. automodule:: habit.core.machine_learning
   :no-members:

Developer entry point
---------------------

Start with ``MLConfig``, ``MLConfigurator``, ``BaseWorkflow``, ``PipelineBuilder``, ``ModelFactory``, and ``run_selector``. See :doc:`../development/index` for overall architecture.

.. automodule:: habit.core.machine_learning.configurator
   :members:
   :undoc-members:
   :show-inheritance:

Core contracts
--------------

.. automodule:: habit.core.machine_learning.contracts
   :members:
   :undoc-members:
   :show-inheritance:

Workflows
---------

Workflow classes wrap full ML pipelines: training, validation, and prediction.

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

Runners
-------

Since 2026-05, K-fold compute loops live in the ``runners`` subpackage to simplify workflow classes while keeping the public API unchanged.

.. automodule:: habit.core.machine_learning.runners.kfold
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.runners.holdout
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   ``KFoldWorkflow`` is the public entry point (legacy ``MachineLearningKFoldWorkflow`` remains as a deprecation subclass). It delegates fold training and aggregation to
   :py:class:`habit.core.machine_learning.runners.kfold.KFoldRunner`; persistence and plots use ``ReportWriter`` / ``ModelStore`` / ``PlotComposer``.

Reporting components
--------------------

``habit.core.machine_learning.callbacks`` has been removed. Model storage, report writing, and visualization use explicit components in the ``reporting`` subpackage.

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

Model factory
-------------

.. automodule:: habit.core.machine_learning.models.factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.models.ensemble
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
----------

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

Visualization
-------------

.. automodule:: habit.core.machine_learning.visualization.plotting
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.machine_learning.visualization.km_survival
   :members:
   :undoc-members:
   :show-inheritance:
