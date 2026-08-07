Advanced tabular ML: staged selection and model comparison
==========================================================

:class:`~habit.spec.MLSpec` exposes three ordered table stages:

* ``pre_preprocessing_feature_selectors`` — on the **raw** table before any
  normalisation (v0.1 ``before_z_score: true``). Variance filtering must run
  here: after z-scoring every feature variance is 1.0.
* ``table_preprocessors`` — stateful preprocessing (z-score, min-max, …).
* ``feature_selectors`` — post-preprocessing selection (ANOVA, LASSO, …).

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
   Train metrics: {'accuracy': 0.978, 'auc': 1.0}
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
