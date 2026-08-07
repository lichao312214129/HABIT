Tabular machine learning: train, cross-validate, predict
=========================================================

Habitat (and radiomics) analyses bottom out in a per-subject feature table;
the v1 ML recipes model that table directly. This example covers:

1. declare the modelling definition as an :class:`~habit.spec.MLSpec`
   (``pre_preprocessing_feature_selectors`` for variance-style filters on
   the raw table, ``table_preprocessors``, optional post-preprocessing
   ``feature_selectors``, classifier, metrics),
2. hold-out evaluation with :func:`~habit.recipes.train_model`,
3. K-fold cross-validation with :func:`~habit.recipes.cross_validate`,
4. inference with :func:`~habit.recipes.predict_model`,
5. **TablePipeline** save/load (``.habitpipeline`` archive) — the tabular
   equivalent of ``HabitatModel.save/load``.

Two guarantees make these recipes leak-free by construction: under a split
or a fold the pipeline sees the training rows **only**, and at prediction
time the fitted preprocessing/selection state is replayed rather than
refitted.

Script
------

.. literalinclude:: scripts/tabular_ml_quickstart.py
   :language: python

Output
------

Real output of the script above. The synthetic table has one informative
feature, so the metrics are near-perfect by construction — the point is the
workflow, not the score::

   Table: 80 rows x 8 features, outcome=binary

   --- Hold-out split (75% train / 25% test) ---
   Train metrics: {'accuracy': 1.0, 'auc': 1.0}
   Test metrics:  {'accuracy': 1.0, 'auc': 1.0}

   --- 5-fold cross-validation ---
   Mean metrics: {'accuracy': 0.988, 'auc': 1.0}
   Std metrics:  {'accuracy': 0.025, 'auc': 0.0}

   --- TablePipeline round-trip ---
   Saved demo.habitpipeline (...)
   Reloaded classifier: LogisticRegression

What to read next
-----------------

* :doc:`ml_advanced` — staged selectors + ``compare_models``
* :doc:`../api/domain_table` — tabular building blocks
* :doc:`../configuration/machine_learning` — YAML equivalent (``habit model`` / ``habit cv``)
