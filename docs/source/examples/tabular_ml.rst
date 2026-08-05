Tabular machine learning: train, cross-validate, predict
========================================================

Habitat (and radiomics) analyses bottom out in a per-subject feature table;
the v1 ML recipes model that table directly. This example covers the full
tabular track on a synthetic :class:`~habit.contracts.FeatureTable`:

1. declare the modelling definition as an :class:`~habit.spec.MLSpec`
   (preprocessors, selector, classifier, metrics),
2. hold-out evaluation with :func:`~habit.recipes.train_model`,
3. K-fold cross-validation with :func:`~habit.recipes.cross_validate`,
4. inference on new rows with :func:`~habit.recipes.predict_model`.

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

   --- Prediction with the fitted pipeline ---
   Predictions: 80 rows
   Class probability columns: ['0', '1']
                0      1
   subject
   subj001  0.903  0.097
   subj002  0.109  0.891
   subj003  0.830  0.170

What to read next
-----------------

* :doc:`../api/domain_table` — the tabular building blocks (preprocessors,
  selectors, classifiers, metrics, :class:`~habit.domain.TablePipeline`)
* :class:`~habit.recipes.ModelResult` / :class:`~habit.recipes.CVResult` /
  :class:`~habit.recipes.PredictionResult` — the result contracts
* :doc:`../configuration/machine_learning` — the YAML equivalent
  (``habit model`` / ``habit cv``)
