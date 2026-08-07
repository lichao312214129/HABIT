Advanced tabular ML: ordered selection and model comparison
===========================================================

:class:`~habit.spec.MLSpec` declares its table steps as ONE ordered list,
``steps``. The list order is the execution order, so preprocessors and
selectors interleave freely and position is the only thing that expresses
stage:

* a variance filter belongs **first**, on the raw table — after z-scoring
  every feature variance is 1.0, so the same step placed later selects
  nothing meaningful (v0.1 spelled this ``before_z_score: true``);
* stateful preprocessing (z-score, min-max, …) sits wherever the design puts
  it, including *between* two selectors;
* supervised selection (ANOVA, LASSO, …) normally follows the scaling.

The three predecessor fields ``pre_preprocessing_feature_selectors``,
``table_preprocessors`` and ``feature_selectors`` are deprecated aliases kept
for all of v1.x; they are folded into ``steps`` in that order.

:func:`~habit.recipes.compare_models` compares saved prediction CSVs and writes
ROC / calibration figures (the programmatic twin of ``habit compare``).

Script
------

.. literalinclude:: scripts/ml_advanced_demo.py
   :language: python

Output
------

::

   Table: 60 rows x 10 features

   --- Staged pipeline (pre-variance -> zscore -> k-best -> LR) ---
   Train metrics: {'accuracy': 1.0, 'auc': 1.0}
   Test metrics:  {'accuracy': 1.0, 'auc': 1.0}

   --- compare_models output: .../comparison ---
   Multi-model figures (ROC, DCA, calibration, PR) and metrics land under
   output_dir (e.g. roc_curves.pdf, metrics/metrics.json). Train/CV
   single-model figures from habit model / habit cv use
   output/visualizations/ with train_ / test_ / cv_ prefixes.

What to read next
-----------------

* :doc:`tabular_ml` — train / cross-validate / predict basics
* :doc:`../api/domain_table` — :class:`~habit.domain.TablePipeline` internals
* :doc:`visualization` — ``habit.viz`` figures for survival and regression
