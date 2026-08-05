Tabular ML API (train / CV / predict / compare)
===============================================

Recipes:

* :func:`~habit.recipes.train_model`
* :func:`~habit.recipes.cross_validate`
* :func:`~habit.recipes.predict_model`
* :func:`~habit.recipes.compare_models`

Persistence: :meth:`~habit.domain.pipeline.TablePipeline.save` /
:meth:`~habit.domain.pipeline.TablePipeline.load` (``.habitpipeline``).

``compare_models`` validates :class:`~habit.schemas.workflows.ml.ComparisonFileConfig`:
each file **must** declare ``prob_col`` (positive-class probability). Write
probabilities from ``PredictionResult.probabilities`` when exporting CSVs.

**Atomic** — call ``predict_model`` on a one-row
:class:`~habit.contracts.FeatureTable` (or any held-out id slice).

Script
------

.. literalinclude:: scripts/tabular_ml_api_demo.py
   :language: python

Coverage
--------

``demo_data/results/api/07_ml`` (train / CV / predict / compare on real
``demo_data/ml_data`` tables).

What to read next
-----------------

* :doc:`features_radiomics_api` — upstream feature tables
* :doc:`viz_parallel_extras_api` — ML coefficient forest and extras
