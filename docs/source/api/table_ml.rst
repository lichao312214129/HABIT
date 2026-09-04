Tabular ML (bookmark)
=====================

Supporting table-level modelling **after** a habitat
:class:`~habit.contracts.FeatureTable`. This is not the product core and
is not a wall of classifiers on :doc:`index`.

**User guide:** :doc:`domain_table`. Composer:
:class:`~habit.pipeline.TablePipeline` (documented on :doc:`pipeline`).
Construct components with each domain registry's ``create`` /
``constructor_signature`` (do not invent sklearn-shaped aliases).
Image-side feature scaling is :doc:`feature_preprocessing`.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   habit.table_preprocessing.TablePreprocessor
   habit.table_preprocessing.TablePreprocessorRegistry
   habit.feature_selection.FeatureSelector
   habit.feature_selection.FeatureSelectorRegistry
   habit.classification.Classifier
   habit.classification.ClassifierRegistry
   habit.evaluation.Metric
   habit.evaluation.MetricRegistry
   habit.recipes.ModelResult
   habit.recipes.CVResult
   habit.recipes.PredictionResult
   habit.recipes.SearchResult

Functions
---------

Recipe helpers exported from ``habit.recipes``:

.. autosummary::
   :toctree: generated
   :nosignatures:

   habit.recipes.train_model
   habit.recipes.cross_validate
   habit.recipes.predict_model
   habit.recipes.search_hyperparameters
   habit.recipes.compare_models
   habit.recipes.pairwise_delong_test
   habit.api.plugins.create_ml_model

Evaluation statistics
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   habit.evaluation.auc_confidence_interval
   habit.evaluation.calibration_tests
   habit.evaluation.delong_test
   habit.evaluation.icc_analysis
   habit.evaluation.repeat_measurement_matrix
   habit.evaluation.AucConfidenceInterval
   habit.evaluation.CalibrationResult
   habit.evaluation.DelongResult
