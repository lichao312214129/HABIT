Python API
============

Use HABIT programmatically from notebooks or scripts. The **stable public
contract** is the top-level ``habit`` package (see :doc:`../development/index`
and ``CHANGELOG.md``). CLI commands are thin wrappers around the same runners.

Typical pattern:

1. Load a typed config with ``XxxConfig.from_file(path)`` from ``habit``, or
   construct a Python dictionary and pass it directly to ``run_*``.
2. Optionally set up a logger with ``habit.setup_logger``.
3. Call the matching ``run_*`` helper and inspect its ``WorkflowResult``.

Migration from deep imports
---------------------------

Older examples imported from ``habit.core.*``. Those paths still work, but new
code should prefer the top-level names below.

Preprocessing
-------------

.. code-block:: python

   import logging
   from pathlib import Path

   from habit import PreprocessingConfig, run_preprocess, setup_logger

   config = PreprocessingConfig.from_file(
       "config/preprocessing/config_preprocessing_demo.yaml"
   )
   out = Path(config.out_dir)
   out.mkdir(parents=True, exist_ok=True)

   logger = setup_logger(
       name="my_script.preprocess",
       output_dir=out,
       log_filename="processing.log",
       level=logging.INFO,
   )
   result = run_preprocess(config, logger=logger)
   print(result.artifact("output_dir"))

In-memory configuration
-----------------------

Every public ``run_*`` workflow accepts its matching validated config class or
a plain Python dictionary. Dictionaries use the same Pydantic validation as
``XxxConfig.from_file`` and invalid values fail before the workflow starts.

.. code-block:: python

   from habit import run_preprocess

   result = run_preprocess(
       {
           "data_dir": "data/raw",
           "out_dir": "results/preprocessed",
           "processes": 4,
           "preprocessing": {
               "resample": {
                   "images": ["T1", "T2"],
                   "target_spacing": [1.0, 1.0, 1.0],
               }
           },
       }
   )
   processed_root = result.artifact("output_dir")

DICOM sort
----------

.. code-block:: python

   from habit import DicomSortConfig, run_dicom_sort

   config = DicomSortConfig.from_file("config/dicom_sort/config_sort_dicom.yaml")
   run_dicom_sort(config)

Habitat segmentation
--------------------

Train and predict are separate entry points. Predict mode requires
``pipeline_path`` on the config (or via overrides).

.. code-block:: python

   import logging

   from habit import (
       HabitatAnalysisConfig,
       apply_habitat_cli_overrides,
       run_habitat_analysis,
       setup_logger,
   )

   config = HabitatAnalysisConfig.from_file(
       "config/habitat/config_habitat_two_step.yaml"
   )
   apply_habitat_cli_overrides(config, debug=False, resume=False)

   logger = setup_logger(
       name="my_script.habitat",
       output_dir=config.out_dir,
       log_filename="habitat_analysis.log",
       level=logging.DEBUG if config.debug else logging.INFO,
   )
   result = run_habitat_analysis(config, logger=logger)
   results_df = result.data
   pipeline_path = result.artifacts.get("pipeline")

Feature extraction
------------------

.. code-block:: python

   from habit import load_feature_extraction_config, run_feature_extraction

   config, plugin_configs = load_feature_extraction_config(
       "config/feature_extraction/config_extract_features_demo.yaml"
   )
   result = run_feature_extraction(config, plugin_configs=plugin_configs)

``FeatureExtractionConfig.from_file(...)`` remains suitable when the YAML uses
only built-in feature types. Use ``load_feature_extraction_config`` when the
configuration includes plugin-specific sections, so programmatic execution
matches the ``habit extract-features`` CLI.

Traditional radiomics
---------------------

.. code-block:: python

   from habit import RadiomicsConfig, run_radiomics

   config = RadiomicsConfig.from_file(
       "config/radiomics/config_traditional_radiomics.yaml"
   )
   result = run_radiomics(config)

Machine learning
----------------

Holdout train/predict and K-fold share ``MLConfig``. Set ``run_mode`` in YAML
or override before calling the runner.

.. code-block:: python

   from habit import (
       MLConfig,
       apply_ml_mode_override,
       run_kfold,
       run_ml,
   )

   config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_radiomics.yaml"
   )
   config = apply_ml_mode_override(config, mode="train")
   result = run_ml(config)
   ml_run = result.data

   kfold_config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_kfold_demo.yaml"
   )
   kfold_result = run_kfold(kfold_config)

sklearn-compatible estimators
----------------------------

Use the estimators when a HABIT component needs to participate in a standard
scikit-learn workflow. ``SubjectFeatureAggregator`` converts a habitat long
table to one feature row per subject. Subject identifiers stay in the index by
default, preventing them from being treated as predictive features.

.. code-block:: python

   from habit import SubjectFeatureAggregator

   aggregator = SubjectFeatureAggregator(aggregation="mean")
   subject_features = aggregator.fit_transform(habitat_feature_table)
   assert subject_features.index.name == "subject"

``HabitClassifier`` accepts either a validated ``MLConfig`` or an in-memory
dictionary accepted by ``MLConfig``. It follows the standard
``fit`` / ``predict`` / ``predict_proba`` / ``score`` classifier protocol.

.. code-block:: python

   from habit import HabitClassifier

   classifier = HabitClassifier(
       config={
           "run_mode": "train",
           "input": [{"path": "train.csv", "subject_id_col": "subject", "label_col": "label"}],
           "output": "results/ml",
           "models": {"LogisticRegression": {"params": {"solver": "liblinear"}}},
           "feature_selection_methods": [],
           "is_visualize": False,
       }
   )
   classifier.fit(x_train, y_train)
   probabilities = classifier.predict_proba(x_test)
   classifier.save("models/classifier.joblib")

The saved estimator records HABIT and scikit-learn version metadata. Load only
artifacts from trusted sources because joblib uses Python pickle semantics.

.. code-block:: python

   loaded_classifier = HabitClassifier.load("models/classifier.joblib")
   predictions = loaded_classifier.predict(x_test)

Model comparison
----------------

.. code-block:: python

   from habit import ModelComparisonConfig, run_model_comparison

   config = ModelComparisonConfig.from_file(
       "config/model_comparison/config_model_comparison_demo.yaml"
   )
   result = run_model_comparison(config)
   metrics = result.data

ICC analysis
------------

.. code-block:: python

   from habit import ICCConfig, run_icc_analysis

   config = ICCConfig.from_file("config/auxiliary/config_icc_demo.yaml")
   result = run_icc_analysis(config)

Test-retest analysis
--------------------

``run_test_retest_analysis`` maps retest habitat labels to the corresponding
test labels, then writes the remapped images. The mapping is returned through
``WorkflowResult.data`` for callers that need to inspect or persist it::

   from habit import TestRetestConfig, run_test_retest_analysis

   config = TestRetestConfig.from_file(
       "config/auxiliary/config_test_retest.yaml"
   )
   habitat_mapping = run_test_retest_analysis(config).data

Top-level package exports
-------------------------

``import habit`` lazily exposes the runners, config classes, and helpers listed
above. The canonical symbol list is defined in ``habit.api.registry.PUBLIC_API_SYMBOLS``
and verified by ``tests/api/test_public_api.py``.

Optional dependencies can be probed without importing heavy backends::

   import habit

   if habit.is_available("radiomics"):
       ...

See also: :doc:`../api/index` (autodoc reference), :doc:`../configuration/index` (YAML fields).
